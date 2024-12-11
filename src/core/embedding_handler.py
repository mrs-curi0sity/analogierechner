import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random


class EmbeddingHandler:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        Initialisiert den EmbeddingHandler mit einem mehrsprachigen Modell.
        """
        self.model = SentenceTransformer(model_name)
        self.word_list = self.load_word_list()

    def load_word_list(self, path='data/de_50k_most_frequent.txt'):
        """
        Lädt die Liste der häufigsten deutschen Wörter.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip().split(" ")[0] for line in f]
        except FileNotFoundError:
            print(f"Wortliste nicht gefunden unter {path}")
            return []

    def find_analogy(self, word1: str, word2: str, word3: str, n: int = 5):
        """
        Findet Analogien wie: word1 : word2 = word3 : ?
        """
        try:
            # Überprüfen, ob Eingabewörter im Modell vorhanden sind
            if word1.lower() not in self.word_list or word2.lower() not in self.word_list or word3.lower() not in self.word_list:
                print("Eines der Eingabewörter ist nicht in der Wortliste.")
                return []
    
            # Embeddings berechnen
            emb1 = self.model.encode(word1)
            emb2 = self.model.encode(word2)
            emb3 = self.model.encode(word3)
            
            # Vektordifferenz berechnen
            diff_vector = emb2 - emb1
            target_vector = emb3 + diff_vector
            
            # Alle Kandidaten außer Eingabewörter
            candidates = [w for w in self.word_list if w not in [word1, word2, word3]]
            
            # Embeddings der Kandidaten
            candidate_embeddings = self.model.encode(candidates)
            
            # Ähnlichkeiten berechnen
            similarities = cosine_similarity([target_vector], candidate_embeddings)[0]
            
            # Top-Kandidaten finden und sortieren
            sorted_indices = similarities.argsort()[-n:][::-1]
            results = [(candidates[idx], similarities[idx]) for idx in sorted_indices]
            
            return results
        
        except Exception as e:
            print(f"Fehler in find_analogy: {e}")
            return []


    def find_similar_words(self, word: str, n: int = 5):
        """
        Findet die semantisch ähnlichsten Wörter zu einem gegebenen Wort.
        
        Args:
            word (str): Das Wort, zu dem Ähnlichkeiten gesucht werden
            n (int): Anzahl der zurückzugebenden ähnlichen Wörter
        
        Returns:
            List[Tuple[str, float]]: Liste von (Wort, Ähnlichkeitsscore)
        """
        try:
            # Überprüfen, ob das Eingabewort in der Wortliste ist
            word = word.lower()
            if word not in self.word_list:
                print(f"Wort '{word}' nicht in der Wortliste gefunden.")
                return []
            
            # Embedding des Eingabeworts
            input_embedding = self.model.encode(word)#self.word_embeddings.get(word)
            
            # Kandidaten (außer das Eingabewort selbst)
            candidates = [w for w in self.word_list if w != word]
            
            # Ähnlichkeiten berechnen
            similarities = [
                cosine_similarity([input_embedding], [self.model.encode(candidate)])[0][0] 
                for candidate in candidates
            ]
            
            # Top-Kandidaten finden
            sorted_indices = np.argsort(similarities)[-n:][::-1]
            results = [(candidates[idx], similarities[idx]) for idx in sorted_indices]
            
            return results
        
        except Exception as e:
            print(f"Fehler in find_similar_words: {e}")
            return []



# Beispielverwendung
if __name__ == "__main__":
    handler = EmbeddingHandler()
    
    # Beispiel-Analogien
    analogies = [
        ("Berlin", "Deutschland", "Paris"),
        ("Mutter", "Frau", "Vater"),
        ("groß", "größer", "klein")
    ]
    
    for word1, word2, word3 in analogies:
        print(f"\nAnalogiesuche: {word1} : {word2} = {word3} : ?")
        results = handler.find_analogy(word1, word2, word3)
        for word, score in results:
            print(f"{word}: {score:.3f}")