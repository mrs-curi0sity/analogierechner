import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import streamlit as st
from google.cloud import storage
from src.core.logger import logger
from typing import List, Tuple, Dict
import time

class EmbeddingHandler:
    """
    Handler für Word Embeddings mit direkter Analogieberechnung.
    Verwendet FastText für deutsche und GloVe für englische Embeddings.
    """
    
    # Cloud path prefix - leer für lokale Entwicklung
    CLOUD_PREFIX = "gs://analogierechner-models/" if os.getenv("ENVIRONMENT") == "cloud" else ""
    
    MODEL_CONFIGS = {
        'de': {
            'type': 'fasttext',
            'model_path': 'data/cc.de.100.bin',
            'word_list_path': 'data/de_50k_most_frequent.txt'
        },
        'en': {
            'type': 'glove',
            'model_path': 'data/glove.6B.100d.txt',
            'word_list_path': 'data/en_50k_most_frequent.txt'
        }
    }

    def __init__(self, language='de'):
        logger.info(f"Initializing EmbeddingHandler for {language}")
        self.language = language
        self.config = self.MODEL_CONFIGS[language]
        self.model_type = self.config['type']
        
        # Pfade auflösen
        self.model_path = self._get_file_path(self.config['model_path'])
        self.word_list_path = self._get_file_path(self.config['word_list_path'])
        
        self._model = None
        self._word_list = None
        self._embedding_cache = {}
        
    def hybrid_similarity(target_vector: np.ndarray, candidate_vector: np.ndarray) -> float:
        """
        Berechnet eine hybride Ähnlichkeit aus Cosine Similarity und euklidischer Distanz.
        """
        # Cosine Similarity berechnen
        cos_sim = float(cosine_similarity([target_vector], [candidate_vector])[0][0])
        
        # Euklidische Distanz berechnen und normalisieren
        eucl_dist = np.linalg.norm(target_vector - candidate_vector)
        max_dist = np.linalg.norm(target_vector) + np.linalg.norm(candidate_vector)
        norm_dist = eucl_dist / max_dist
        
        # Gewichtete Kombination
        similarity = 0.7 * cos_sim - 0.3 * norm_dist
        
        return similarity
    
    def find_analogy(self, word1: str, word2: str, word3: str, n: int = 5):
        try:
            logger.info(f"Input words: {word1}, {word2}, {word3}")
            
            # Wörter normalisieren
            word1 = word1.replace(',', '').strip().lower()
            word2 = word2.replace(',', '').strip().lower()
            word3 = word3.replace(',', '').strip().lower()
            
            # Embeddings holen
            emb1 = np.array(self.get_embedding(word1))
            emb2 = np.array(self.get_embedding(word2))
            emb3 = np.array(self.get_embedding(word3))
            
            # Differenzvektor berechnen
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            logger.info("Starting analogy search...")
            similarities = []
            exclude_words = {word1, word2, word3}
            
            # Wortliste durchgehen
            for word in self.word_list:
                word = word.lower()
                if word not in exclude_words:
                    try:
                        word_embedding = np.array(self.get_embedding(word))
                        
                        # Cosine Similarity direkt nutzen
                        sim = float(cosine_similarity([target_vector], [word_embedding])[0][0])
                        
                        # Differenzvektor-Check optional machen
                        new_diff = word_embedding - emb3
                        diff_similarity = float(cosine_similarity([diff_vector], [new_diff])[0][0])
                        
                        # Weniger strenge Filterung
                        if diff_similarity > 0.0:  # Akzeptiere alle positiven Ähnlichkeiten
                            similarities.append((word, sim, diff_similarity))
                            
                    except Exception as e:
                        logger.debug(f"Skipping word {word}: {str(e)}")
                        continue
            
            # Nach Gesamtähnlichkeit sortieren
            results = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Top-N Ergebnisse
            n = min(n, len(results))
            final_results = [(word, sim) for word, sim, _ in results[:n]]
            
            if final_results:
                best_result = final_results[0][0]
                debug_info = {
                    'input_similarity': float(cosine_similarity([emb1], [emb2])[0][0]),
                    'output_similarity': float(cosine_similarity([emb3], [self.get_embedding(best_result)])[0][0]),
                    'vector_norm': float(np.linalg.norm(diff_vector))
                }
            else:
                debug_info = None
            
            return final_results, None, debug_info
            
        except Exception as e:
            logger.error(f"Error in analogy calculation: {str(e)}")
            raise

    
    def find_analogy_vectorized(self, word1: str, word2: str, word3: str, n: int = 5):
        """
        Vektorisierte Version der Analogieberechnung.
        """
        try:
            start_time = time.time()
            logger.info(f"Input words: {word1}, {word2}, {word3}")
            
            # Wörter normalisieren
            word1 = word1.replace(',', '').strip().lower()
            word2 = word2.replace(',', '').strip().lower()
            word3 = word3.replace(',', '').strip().lower()
            
            # Cache erstellen falls noch nicht vorhanden
            if not hasattr(self, '_all_embeddings'):
                logger.info("Creating embeddings cache...")
                self._all_embeddings = np.array([self.get_embedding(w) for w in self.word_list])
                logger.info("Cache creation completed")
            
            # Embeddings für Eingabewörter
            emb1 = np.array(self.get_embedding(word1))
            emb2 = np.array(self.get_embedding(word2))
            emb3 = np.array(self.get_embedding(word3))
            
            # Vektorisierte Berechnung
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            # Alle Similarities auf einmal berechnen
            similarities = cosine_similarity([target_vector], self._all_embeddings)[0]
            
            # Ausschlusswörter
            exclude_indices = [i for i, w in enumerate(self.word_list) 
                             if w.lower() in {word1, word2, word3}]
            similarities[exclude_indices] = -1
            
            # Top-N finden
            top_indices = np.argsort(similarities)[-n:][::-1]
            results = [(self.word_list[i], float(similarities[i])) 
                      for i in top_indices]
            
            computation_time = time.time() - start_time
            
            return results, None, {
                'input_similarity': float(cosine_similarity([emb1], [emb2])[0][0]),
                'computation_time': computation_time
            }
            
        except Exception as e:
            logger.error(f"Error in vectorized analogy calculation: {str(e)}")
            raise
    
    def find_similar_words(self, word: str, n: int = 10) -> List[Tuple[str, float]]:
        """Findet ähnliche Wörter basierend auf Cosine-Similarity"""
        try:
            input_embedding = self.get_embedding(word)
            
            similarities = []
            for w in self.word_list:
                if w.lower() != word.lower():
                    try:
                        w_embedding = self.get_embedding(w)
                        sim = cosine_similarity([input_embedding], [w_embedding])[0][0]
                        similarities.append((w, sim))
                    except:
                        continue
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
            
        except Exception as e:
            raise ValueError(f"Fehler beim Finden ähnlicher Wörter: {str(e)}")

    def get_embedding(self, word: str) -> np.ndarray:
        """Holt Embedding für ein Wort mit Caching"""
        word_lower = word.lower()
        
        if word_lower in self._embedding_cache:
            return self._embedding_cache[word_lower]
                
        try:
            if self.model_type == 'glove':
                if word_lower not in self.model:
                    raise ValueError(f"Word '{word}' not found in vocabulary")
                embedding = self.model[word_lower]
            else:  # fasttext
                # Prüfen ob das Wort in der Wortliste ist
                if word_lower not in self.word_list:
                    raise ValueError(f"Word '{word}' not found in vocabulary")
                embedding = self.model.get_word_vector(word)
                    
            self._embedding_cache[word_lower] = embedding
            return embedding
                    
        except Exception as e:
            raise ValueError(f"Word '{word}' not found in vocabulary")

    @property
    def model(self):
        """Lazy loading des Models"""
        if self._model is None:
            logger.info(f"Loading model for type: {self.model_type}")
            if self.model_type == 'fasttext':
                self._model = fasttext.load_model(self.model_path)
            elif self.model_type == 'glove':
                self._model = self._load_glove(self.model_path)
        return self._model
    
    @property
    def word_list(self):
        """Lazy loading der Wortliste"""
        if self._word_list is None:
            if self.model_type == 'fasttext':
                self._word_list = self.load_word_list(self.word_list_path)
            else:  # glove
                self._word_list = list(self.model.keys())
        return self._word_list

    def load_word_list(self, path: str) -> List[str]:
        """Lädt Wortliste aus Textdatei"""
        try:
            st.write(f"Loading word list from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                words = [line.strip().split(" ")[0].lower() for line in f]
            st.write(f"Loaded {len(words)} words")
            return words
        except Exception as e:
            st.error(f"Error loading word list: {str(e)}")
            raise

    def _get_file_path(self, path: str) -> str:
        """Handhabt Dateizugriff basierend auf Umgebung"""
        if self.CLOUD_PREFIX:
            cloud_path = self.CLOUD_PREFIX + path
            local_path = f"/tmp/{os.path.basename(path)}"
            return self._download_from_gcs(cloud_path, local_path)
        return path

    def _download_from_gcs(self, gs_path: str, local_path: str) -> str:
        """Downloads a file from Google Cloud Storage"""
        try:
            logger.info(f"Downloading {gs_path} to {local_path}")
            
            bucket_name = gs_path.replace("gs://", "").split("/")[0]
            blob_name = "/".join(gs_path.replace(f"gs://{bucket_name}/", "").split("/"))
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                raise FileNotFoundError(f"Blob does not exist: {gs_path}")
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            
            if os.path.exists(local_path):
                return local_path
            else:
                raise FileNotFoundError(f"File not found after download: {local_path}")
                
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise

    def _load_glove(self, path: str) -> Dict[str, np.ndarray]:
        """Lädt GloVe Embeddings"""
        embeddings = {}
        word_list = set()
        
        # Wortliste laden
        try:
            with open(self.word_list_path, 'r', encoding='utf-8') as f:
                word_list = {line.strip().split()[0].lower() for line in f}
        except Exception as e:
            logger.error(f"Error loading word list: {str(e)}")
            raise
        
        # GloVe Embeddings laden
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if not word_list or word in word_list:
                        vector = np.asarray(values[1:], dtype='float32')
                        embeddings[word] = vector
            return embeddings
        except Exception as e:
            logger.error(f"Error loading GloVe embeddings: {str(e)}")
            raise