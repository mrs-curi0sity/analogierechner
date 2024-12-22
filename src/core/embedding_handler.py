import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util
import streamlit as st
from google.cloud import storage
from src.core.logger import logger


class EmbeddingHandler:
    # Basis-Pfade basierend auf Umgebung
    BASE_PATH = "gs://analogierechner-models/data" if os.getenv("ENVIRONMENT") == "cloud" else "data"
    
    def _download_from_gcs(self, gs_path, local_path):
        """Downloads a file from Google Cloud Storage"""
        try:
            bucket_name = gs_path.replace("gs://", "").split("/")[0]
            blob_name = "/".join(gs_path.replace(f"gs://{bucket_name}/", "").split("/"))
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            return local_path
        except Exception as e:
            st.error(f"Download error: {str(e)}")
            raise e

    MODEL_CONFIGS = {
        'de': {
            'type': 'fasttext',
            'local_path': 'data/cc.de.300.bin',
            'cloud_path': 'gs://analogierechner-models/data/cc.de.300.bin',
            'local_word_list': 'data/de_50k_most_frequent.txt',
            'cloud_word_list': 'gs://analogierechner-models/data/de_50k_most_frequent.txt'
        },
        'en': {
            'type': 'glove',
            'local_path': 'data/glove.6B.100d.txt',
            'cloud_path': 'gs://analogierechner-models/data/glove.6B.100d.txt',
            'local_word_list': 'data/en_50k_most_frequent.txt',
            'cloud_word_list': 'gs://analogierechner-models/data/en_50k_most_frequent.txt'
        }
    }
    
    def _get_file_path(self, path):
        """Handhabt Dateizugriff basierend auf Umgebung"""
        environment = os.getenv("ENVIRONMENT", "local")
        st.write(f"Environment: {environment}")
        st.write(f"Original path: {path}")
        
        if environment == "cloud":
            # Für Cloud-Umgebung: Wenn lokaler Pfad übergeben, ersetze ihn durch Cloud-Pfad
            for lang_config in self.MODEL_CONFIGS.values():
                if path == lang_config['local_path']:
                    path = lang_config['cloud_path']
                elif path == lang_config['local_word_list']:
                    path = lang_config['cloud_word_list']
            
            local_path = f'/tmp/{os.path.basename(path)}'
            st.write(f"Downloading {path} to {local_path}")
            return self._download_from_gcs(path, local_path)
        
        return path

    def __init__(self, language='de'):
        self.language = language
        self.config = self.MODEL_CONFIGS[language]
        self.model_type = self.config['type']
        
        # Korrekte Schlüssel verwenden
        self.model_path = self._get_file_path(self.config['local_path'])
        self.word_list_path = self._get_file_path(self.config['local_word_list'])
        
        self._model = None
        self._word_list = None
        self._embedding_cache = {}

    def _load_glove(self, path):
        """Lädt GloVe Embeddings aus einer Textdatei"""
        embeddings = {}
        word_list = set()  # Set für schnellere Lookups
        
        # Korrekten Schlüssel verwenden
        if self.config['local_word_list']:
            with open(self.word_list_path, 'r', encoding='utf-8') as f:
                word_list = {line.strip().split()[0].lower() for line in f}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if not word_list or word in word_list:  # wenn keine Liste da ist oder Wort in Liste
                        vector = np.asarray(values[1:], dtype='float32')
                        embeddings[word] = vector
            return embeddings
        except Exception as e:
            raise Exception(f"Fehler beim Laden der GloVe Embeddings: {str(e)}")

    @property
    def model(self):
        if self._model is None:
            # Debug-Ausgabe
            st.write(f"Loading model for type: {self.model_type}")
            st.write(f"Using path: {self.model_path}")
            if self.model_type == 'fasttext':
                self._model = fasttext.load_model(self.model_path)
            elif self.model_type == 'glove':
                self._model = self._load_glove(self.model_path)
        return self._model
    

    @property
    def word_list(self):
        if self._word_list is None:
            if self.model_type == 'fasttext' and self.config['word_list_path']:
                self._word_list = self.load_word_list(self.config['word_list_path'])
            else:  # glove
                self._word_list = list(self.model.keys())
        return self._word_list


    
    def load_word_list(self, path):
       """Lädt eine Wortliste aus einer Textdatei"""
       try:
           with open(path, 'r', encoding='utf-8') as f:
               return [line.strip().split(" ")[0].lower() for line in f]
       except FileNotFoundError:
           print(f"Wortliste nicht gefunden unter {path}")
           return []
    
    def get_embedding(self, word):
        """Holt Embedding für ein Wort"""
        word_lower = word.lower()
        
        # Prüfe Cache
        if word_lower in self._embedding_cache:
            return self._embedding_cache[word_lower]
            
        try:
            if self.model_type == 'glove':
                # Für englische Wörter
                if word_lower not in self.model:
                    raise ValueError(f"Das Wort '{word}' wurde nicht im englischen Vokabular gefunden")
                embedding = self.model[word_lower]
            else:
                # Für deutsche Wörter
                embedding = self.model.get_word_vector(word)
                
            self._embedding_cache[word_lower] = embedding
            return embedding
                
        except Exception as e:
            if self.language == 'en':
                raise ValueError(f"Das Wort '{word}' wurde nicht im englischen Vokabular gefunden")
            else:
                raise ValueError(f"Das Wort '{word}' wurde nicht im deutschen Vokabular gefunden")


    def find_analogy(self, word1, word2, word3, expected_result="", n=5):
       """Berechnet Wort-Analogien"""
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
           
           # Debug: Ähnlichkeit zwischen Eingabe- und Ausgabewort
           if results:
               result_emb = self.get_embedding(results[0][0])
               output_similarity = cosine_similarity([emb3], [result_emb])[0][0]
               
               # Zusätzliche Debug-Info
               debug_info = {
                    'input_similarity': input_similarity,  # Ähnlichkeit word1:word2
                    'output_similarity': output_similarity,  # Ähnlichkeit word3:result
                    'vector_norm': np.linalg.norm(diff_vector),  # Größe des Differenzvektors
                    'norm_word1': norm1,  # Länge Vektor word1
                    'norm_word2': norm2,  # Länge Vektor word2
                    'norm_word3': norm3,  # Länge Vektor word3
                    'norm_result': np.linalg.norm(result_emb)  # Länge Vektor result
                }
               
               if expected_result:
                   expected_emb = self.get_embedding(expected_result)
                   expected_similarity = cosine_similarity([target_vector], [expected_emb])[0][0]
                   return results, expected_similarity, debug_info
               return results, None, debug_info
               
           return results, None, None
       
       except Exception as e:
           raise Exception(f"Fehler bei der Analogieberechnung: {str(e)}")

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