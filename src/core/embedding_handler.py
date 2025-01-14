import os
import time
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util
import streamlit as st
from google.cloud import storage
from src.core.logger import logger
from datasketch import MinHashLSH, MinHash
from typing import List, Tuple, Dict


class EmbeddingHandler:

    """
    LSH (Locality-Sensitive Hashing) für Word Embeddings:
    1. Beim Start werden alle Wort-Embeddings in LSH-Buckets einsortiert
    2. Ähnliche Vektoren landen mit hoher Wahrscheinlichkeit im gleichen Bucket
    3. Bei der Analogiesuche müssen nur noch Wörter aus relevanten Buckets verglichen werden
    4. Dies reduziert die Suche von O(n) auf O(b) wo b die Bucket-Größe ist
    """
    
    # Cloud path prefix - leer für lokale Entwicklung
    CLOUD_PREFIX = "gs://analogierechner-models/" if os.getenv("ENVIRONMENT") == "cloud" else ""
    LSH_NUM_PERM = 128
    LSH_THRESHOLD = 0.5
    
    MODEL_CONFIGS = {
        'de': {
            'type': 'fasttext',
            'model_path': 'data/cc.de.100.bin',
            'word_list_path': 'data/de_50k_most_frequent.txt',
            'lsh_path': '/tmp/lsh_index_de.pkl'
        },
        'en': {
            'type': 'glove',
            'model_path': 'data/glove.6B.100d.txt',
            'word_list_path': 'data/en_50k_most_frequent.txt',
            'lsh_path': '/tmp/lsh_index_en.pkl'
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
        
        # LSH Komponenten - beide mit gleichem num_perm
        self.lsh = MinHashLSH(threshold=self.LSH_THRESHOLD, num_perm=self.LSH_NUM_PERM)
        self.word_to_minhash: Dict[str, MinHash] = {}
        
        self._model = None
        self._word_list = None
        self._embedding_cache = {}
        
        # LSH Index initial aufbauen
        start_time = time.time()
        self._initialize_lsh()
        logger.info(f"LSH initialization took {time.time() - start_time:.2f} seconds")



    def _vector_to_minhash(self, vector: np.ndarray) -> MinHash:
        """Konvertiert Embedding-Vektor zu MinHash"""
        # MinHash direkt mit dem gleichen num_perm erstellen wie bei LSH
        mh = MinHash(num_perm=self.LSH_NUM_PERM)  # Fester Wert, muss gleich sein wie bei LSH Init
        bins = (vector * 100).astype(int)
        for i, value in enumerate(bins):
            mh.update(f"{i}:{value}".encode('utf-8'))
        return mh
    
    def _get_file_path(self, path):
        """Handhabt Dateizugriff basierend auf Umgebung"""
        if self.CLOUD_PREFIX:
            cloud_path = self.CLOUD_PREFIX + path
            local_path = f"/tmp/{os.path.basename(path)}"
            return self._download_from_gcs(cloud_path, local_path)
        return path



    def _initialize_lsh(self):
        """Initialisiert oder lädt LSH Index"""
        if os.path.exists(self.config['lsh_path']):
            self._load_lsh_index()
        else:
            logger.info("Creating new LSH index")
            self._build_lsh_index()


    def _build_lsh_index(self):
        """Baut LSH Index aus Wortliste"""
        logger.info("Building LSH index...")
        for word in self.word_list:
            try:
                embedding = self.get_embedding(word)
                mh = self._vector_to_minhash(embedding)
                self.word_to_minhash[word] = mh
                self.lsh.insert(word, mh)
            except Exception as e:
                logger.warning(f"Couldn't add word to LSH index: {word}, {str(e)}")
        self._save_lsh_index()

    def _save_lsh_index(self):
        """Speichert LSH Index"""
        try:
            with open(self.config['lsh_path'], 'wb') as f:
                pickle.dump({
                    'lsh': self.lsh,
                    'word_to_minhash': self.word_to_minhash
                }, f)
        except Exception as e:
            logger.error(f"Failed to save LSH index: {str(e)}")

    def _load_lsh_index(self):
        """Lädt gespeicherten LSH Index"""
        try:
            with open(self.config['lsh_path'], 'rb') as f:
                data = pickle.load(f)
                self.lsh = data['lsh']
                self.word_to_minhash = data['word_to_minhash']
        except Exception as e:
            logger.error(f"Failed to load LSH index: {str(e)}")
            self._build_lsh_index()

    # Bestehende Methoden bleiben größtenteils gleich...

    def find_analogy(self, word1, word2, word3, expected_result="", n=5):
        """Optimierte Analogieberechnung mit LSH"""
        try:
            # Embeddings berechnen
            emb1 = self.get_embedding(word1)
            emb2 = self.get_embedding(word2)
            emb3 = self.get_embedding(word3)
            
            # Vektordifferenz und Zielvektor
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            # LSH für Kandidatenauswahl
            target_mh = self._vector_to_minhash(target_vector)
            candidates = self.lsh.query(target_mh)
            
            # Kandidaten filtern
            exclude_words = {word1.lower(), word2.lower(), word3.lower()}
            candidates = [w for w in candidates if w.lower() not in exclude_words]
            
            # Ähnlichkeiten nur für LSH-Kandidaten berechnen
            similarities = []
            for word in candidates:
                try:
                    word_embedding = self.get_embedding(word)
                    sim = cosine_similarity([target_vector], [word_embedding])[0][0]
                    similarities.append((word, sim))
                except:
                    continue
            logger.info(f"LSH found {len(candidates)} candidates to evaluate")
            if similarities:
                logger.info(f"Similarity range: {min(s[1] for s in similarities):.3f} to {max(s[1] for s in similarities):.3f}")
                            
            # Top-N Results
            results = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

            if not results:
                logger.warning("No results found via LSH, falling back to full search")
                return self._find_analogy_full_search(word1, word2, word3, n)
            
            return results, None, None
        
            
        except Exception as e:
            logger.error(f"Error in analogy calculation: {str(e)}")
            raise

    def _find_analogy_full_search(self, word1, word2, word3, n=5):
        """Fallback zur vollständigen Suche"""
        try:
            # Embeddings berechnen
            emb1 = self.get_embedding(word1)
            emb2 = self.get_embedding(word2)
            emb3 = self.get_embedding(word3)

            # Debug: Ähnlichkeit zwischen Eingabewörtern
            input_similarity = cosine_similarity([emb1], [emb2])[0][0]
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            norm3 = np.linalg.norm(emb3)
            
            # Vektordifferenz und Zielvektor
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            # Kandidaten filtern
            exclude_words = {word1.lower(), word2.lower(), word3.lower()}
            candidates = [w for w in self.word_list if w.lower() not in exclude_words]
            
            # Similarities berechnen
            similarities = []
            for word in candidates:
                try:
                    word_embedding = self.get_embedding(word)
                    sim = cosine_similarity([target_vector], [word_embedding])[0][0]
                    similarities.append((word, sim))
                except:
                    continue
            
            # Top-N Results
            results = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
                
            debug_info = {
                'input_similarity': input_similarity,
                'output_similarity': cosine_similarity([emb3], [self.get_embedding(results[0][0])])[0][0],
                'vector_norm': np.linalg.norm(diff_vector),
                'norm_word1': norm1,
                'norm_word2': norm2,
                'norm_word3': norm3,
                'norm_result': np.linalg.norm(self.get_embedding(results[0][0]))
            }
            
            return results, None, debug_info
            
        except Exception as e:
            logger.error(f"Error in full analogy search: {str(e)}")
            raise


    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading model for type: {self.model_type}")
            logger.info(f"Using path: {self.model_path}")
            if self.model_type == 'fasttext':
                logger.info("Starting FastText model load...")
                self._model = fasttext.load_model(self.model_path)
                logger.info("FastText model loaded successfully")
            elif self.model_type == 'glove':
                self._model = self._load_glove(self.model_path)
        return self._model
    
    @property
    def word_list(self):
        """Brauchen wir für LSH Index Building und Kandidatenfilterung"""
        if self._word_list is None:
            logger.info("Loading word list...")
            if self.model_type == 'fasttext':
                self._word_list = self.load_word_list(self.word_list_path)
            else:  # glove
                self._word_list = list(self.model.keys())
        return self._word_list

  
    def load_word_list(self, path):
        """Lädt eine Wortliste aus einer Textdatei"""
        try:
            st.write(f"Loading word list from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                words = [line.strip().split(" ")[0].lower() for line in f]
            st.write(f"Loaded {len(words)} words")
            return words
        except Exception as e:
            st.error(f"Error loading word list: {str(e)}")
            raise
        
    
    def _download_from_gcs(self, gs_path, local_path):
        """Downloads a file from Google Cloud Storage"""
        try:
            logger.info("=== Starting GCS Download ===")
            logger.info(f"From GCS path: {gs_path}")
            logger.info(f"To local path: {local_path}")
            
            # Parse bucket and blob
            bucket_name = gs_path.replace("gs://", "").split("/")[0]
            blob_name = "/".join(gs_path.replace(f"gs://{bucket_name}/", "").split("/"))
            
            logger.debug(f"Parsed bucket name: {bucket_name}")
            logger.debug(f"Parsed blob name: {blob_name}")
            
            # Initialize client
            logger.info("Initializing storage client...")
            try:
                storage_client = storage.Client()
                logger.debug("Storage client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize storage client: {str(e)}")
                raise
                
            # Get bucket
            logger.debug(f"Getting bucket {bucket_name}...")
            try:
                bucket = storage_client.bucket(bucket_name)
                logger.debug("Got bucket successfully")
            except Exception as e:
                logger.error(f"Failed to get bucket: {str(e)}")
                raise
                
            # Get blob
            logger.debug(f"Getting blob {blob_name}...")
            try:
                blob = bucket.blob(blob_name)
                logger.debug("Got blob successfully")
            except Exception as e:
                logger.error(f"Failed to get blob: {str(e)}")
                raise
                
            # Check if blob exists
            logger.debug("Checking if blob exists...")
            if not blob.exists():
                logger.error(f"Blob does not exist: {gs_path}")
                raise FileNotFoundError(f"Blob does not exist: {gs_path}")
            logger.debug("Blob exists!")
            
            # Create directory if needed
            logger.debug(f"Creating directory {os.path.dirname(local_path)}...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download
            logger.info("Starting download...")
            blob.download_to_filename(local_path)
            
            # Verify download
            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                logger.info(f"Download successful! File size: {size} bytes")
                return local_path
            else:
                logger.error("Download seemed to succeed but file not found!")
                raise FileNotFoundError(f"File not found after download: {local_path}")
                
        except Exception as e:
            logger.error("=== Download Failed ===")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            raise


    
    def get_embedding(self, word):
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
    

    def _load_glove(self, path):
        """Lädt GloVe Embeddings aus einer Textdatei"""
        embeddings = {}
        word_list = set()  # Set für schnellere Lookups
        
        st.write("=== Loading GloVe Embeddings ===")
        st.write(f"Model path: {path}")
        st.write(f"Word list path: {self.word_list_path}")
        
        # Erst Wortliste laden
        try:
            st.write("Loading word list...")
            with open(self.word_list_path, 'r', encoding='utf-8') as f:
                word_list = {line.strip().split()[0].lower() for line in f}
            st.write(f"Loaded {len(word_list)} words from word list")
        except Exception as e:
            st.error(f"Error loading word list: {str(e)}")
            raise
        
        # Dann nur diese Wörter aus GloVe laden
        try:
            st.write("Loading GloVe embeddings...")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if not word_list or word in word_list:  # wenn keine Liste da ist oder Wort in Liste
                        vector = np.asarray(values[1:], dtype='float32')
                        embeddings[word] = vector
            st.write(f"Loaded {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            st.error("=== GloVe Loading Failed ===")
            st.error(f"Error type: {type(e)}")
            st.error(f"Error message: {str(e)}")
            raise Exception(f"Fehler beim Laden der GloVe Embeddings: {str(e)}")

  

    def find_similar_words(self, word: str, n: int = 10):
        """Findet ähnliche Wörter"""
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