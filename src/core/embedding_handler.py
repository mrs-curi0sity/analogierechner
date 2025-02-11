import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import streamlit as st
from google.cloud import storage
from core.logger import logger
from typing import List, Tuple, Dict

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
    
    def find_analogy(self, word1: str, word2: str, word3: str, n: int = 5):
        try:
            logger.info(f"Input words: {word1}, {word2}, {word3}")
            logger.info("Getting embeddings...")
            
            # Sicherstellen dass n ein Integer ist
            n = int(n)
            
            # Kommas aus den Eingabewörtern entfernen
            word1 = word1.replace(',', '').strip()
            word2 = word2.replace(',', '').strip()
            word3 = word3.replace(',', '').strip()
            
            emb1 = self.get_embedding(word1)
            emb2 = self.get_embedding(word2)
            emb3 = self.get_embedding(word3)
            
            logger.info("Embedding shapes:")
            logger.info(f"emb1: {emb1.shape}")
            logger.info(f"emb2: {emb2.shape}")
            logger.info(f"emb3: {emb3.shape}")
            
            # Arrays explizit als numpy arrays casten
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            emb3 = np.array(emb3)
            
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            logger.info("Starting analogy search...")
            similarities = []
            exclude_words = {word1.lower(), word2.lower(), word3.lower()}
            
            for word in self.word_list:
                if word.lower() not in exclude_words:
                    try:
                        word_embedding = np.array(self.get_embedding(word))
                        sim = float(cosine_similarity([target_vector], [word_embedding])[0][0])
                        similarities.append((word, sim))
                    except Exception as e:
                        logger.debug(f"Skipping word {word}: {str(e)}")
                        continue
            
            # Explizit nach dem zweiten Element (similarity) sortieren
            results = sorted(similarities, key=lambda x: float(x[1]), reverse=True)
            
            # Sicherstellen dass n nicht größer ist als die Anzahl der Ergebnisse
            n = min(n, len(results))
            results = results[:n]
            
            logger.info(f"Found {len(results)} results")
            
            if results:
                best_result = results[0][0]
                debug_info = {
                    'input_similarity': float(cosine_similarity([emb1], [emb2])[0][0]),
                    'output_similarity': float(cosine_similarity([emb3], [self.get_embedding(best_result)])[0][0]),
                    'vector_norm': float(np.linalg.norm(diff_vector)),
                    'norm_word1': float(np.linalg.norm(emb1)),
                    'norm_word2': float(np.linalg.norm(emb2)),
                    'norm_word3': float(np.linalg.norm(emb3)),
                    'norm_result': float(np.linalg.norm(self.get_embedding(best_result)))
                }
            else:
                debug_info = None
            
            return results, None, debug_info
            
        except Exception as e:
            logger.error(f"Error in analogy calculation: {str(e)}")
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
            else:
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