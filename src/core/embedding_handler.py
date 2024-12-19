import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util

class EmbeddingHandler:
    def __init__(self, language='de'):
        self.language = language
        self.word_list_path = f'data/{language}_50k_most_frequent.txt'
        
        if language == 'de':
            self.model_path = f'data/cc.{language}.300.bin'
        else: # englisch
            self.model_path = f'data/glove.6B.100d.txt'
        
        # Lazy loading für Model und Wortliste
        self._model = None
        self._word_list = None
        self._embedding_cache = {}
        self._case_mapping = {}
        self._proper_case_word_list = None

    @property
    def model(self):
        """Lazy loading des Models"""
        if self._model is None:
            self._model = fasttext.load_model(self.model_path)
        return self._model

    @property
    def word_list(self):
        """Lazy loading der Wortliste"""
        if self._word_list is None:
            self._word_list = self.load_word_list()
            self._proper_case_word_list = [
                self._get_best_case_variant(word) for word in self._word_list
            ]
        return self._word_list

    def load_word_list(self, path=None):
        """Lädt die Wortliste der gewählten Sprache"""
        path = path or self.word_list_path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip().split(" ")[0].lower() for line in f]
        except FileNotFoundError:
            print(f"Wortliste nicht gefunden unter {path}")
            return []

    def _get_best_case_variant(self, word):
        """Findet die beste Schreibweise eines Wortes"""
        variants = [
            word.lower(),          # frau
            word.capitalize(),     # Frau
            word.upper(),          # FRAU
            word                   # Original-Eingabe
        ]
        variants = list(set(variants))  # Entferne Duplikate
        
        best_variant = word
        best_norm = 0
        
        for variant in variants:
            try:
                embedding = self.model.get_word_vector(variant)
                norm = np.linalg.norm(embedding)
                if norm > best_norm:
                    best_norm = norm
                    best_variant = variant
            except:
                continue
                
        return best_variant


    def _is_acronym(self, word):
        """Erweiterte Akronym-Erkennung"""
        if word.upper() in GERMAN_ACRONYMS:
            return True
            
        # Zusätzliche Heuristiken für unbekannte Akronyme
        if (word.isupper() and 
            len(word) >= 2 and 
            len(word) <= 5 and
            not any(c.isdigit() for c in word) and
            all(c.isalpha() for c in word)):
            return True
            
        return False
    
    def _calculate_similarities(self, target_vector, candidates):
        """Berechnet Cosine-Similarities zwischen Zielvektor und Kandidaten"""
        similarities = []
        for word in candidates:
            try:
                word_embedding = self.get_embedding(word)
                sim = cosine_similarity([target_vector], [word_embedding])[0][0]
                similarities.append(sim)
            except:
                similarities.append(-1)  # Fallback für nicht gefundene Wörter
        return similarities
        

    def get_embedding(self, word):
        """Verbesserte Version mit Case-Handling"""
        if not self._case_mapping:
            self._initialize_case_mapping()
        
        word_lower = word.lower()
        
        # Prüfe Cache
        if word_lower in self._embedding_cache:
            return self._embedding_cache[word_lower]
        
        # Wenn das Wort nicht in unserem Mapping ist,
        # finde die beste Variante on-the-fly
        if word_lower not in self._case_mapping:
            self._case_mapping[word_lower] = self._get_best_case_variant(word)
        
        best_case = self._case_mapping[word_lower]
        
        try:
            embedding = self.model.get_word_vector(best_case)
            self._embedding_cache[word_lower] = embedding
            return embedding
        except Exception as e:
            raise ValueError(f"Konnte kein Embedding für '{word}' ({best_case}) finden: {str(e)}")

    
    def _initialize_case_mapping(self):
        """Initialisiert das Case-Mapping beim ersten Bedarf"""
        if not self._case_mapping:
            print("Initialisiere Case-Mapping...")
            for word in self.word_list:
                lower = word.lower()
                if lower not in self._case_mapping:
                    best_variant = self._get_best_case_variant(word)
                    self._case_mapping[lower] = best_variant

    

    def find_analogy(self, word1, word2, word3, expected_result="", n=5):
        """Verbesserte Analogieberechnung mit korrekter Schreibweise"""
        try:
            # Embeddings berechnen
            emb1 = self.get_embedding(word1)
            emb2 = self.get_embedding(word2)
            emb3 = self.get_embedding(word3)
            
            # Vektordifferenz und Zielvektor
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            # Kandidaten filtern mit korrekter Schreibweise
            exclude_words = {word1.lower(), word2.lower(), word3.lower()}
            candidates_with_case = [
                (proper_case, word.lower()) 
                for proper_case, word in zip(self._proper_case_word_list, self.word_list)
                if word.lower() not in exclude_words
            ]
            
            # Similarities berechnen
            similarities = [
                cosine_similarity([target_vector], [self.get_embedding(proper_case)])[0][0]
                for proper_case, _ in candidates_with_case
            ]
            
            # Top-N Results mit korrekter Schreibweise
            sorted_indices = np.argsort(similarities)[-n:][::-1]
            results = [
                (candidates_with_case[idx][0], similarities[idx])  # Nutze proper_case
                for idx in sorted_indices
            ]
            
            if expected_result:
                expected_emb = self.get_embedding(expected_result)
                expected_similarity = cosine_similarity([target_vector], [expected_emb])[0][0]
                return results, expected_similarity
            return results, None
            
        except Exception as e:
            raise Exception(f"Fehler bei der Analogieberechnung: {str(e)}")

  


    def find_similar_words(self, word: str, n: int = 10):
        """
        Findet die semantisch ähnlichsten Wörter.
        """
        input_embedding = self.get_embedding(word)
        
        candidates = [
            w for w in self.word_list 
            if w != word
        ]
        
        similarities = [
            cosine_similarity([input_embedding], [self.get_embedding(w)])[0][0] 
            for w in candidates
        ]
        
        top_indices = np.argsort(similarities)[-n:][::-1]
        results = [(candidates[idx], similarities[idx]) for idx in top_indices]
        
        return results
