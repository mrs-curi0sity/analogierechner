import unittest
import numpy as np
from core.embedding_handler import EmbeddingHandler

class TestEmbeddingHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt"""
        cls.handler_de = EmbeddingHandler(language='de')
        cls.handler_en = EmbeddingHandler(language='en')

    def test_vocabulary_coverage(self):
        """Test ob häufige deutsche Wörter im Vokabular sind"""
        common_words = [
            'der', 'die', 'das', 
            'mann', 'frau', 
            'junge', 'mädchen',
            'könig', 'königin',
            'berlin', 'münchen',
            'deutschland', 'frankreich'
        ]
        
        for word in common_words:
            with self.subTest(word=word):
                try:
                    embedding = self.handler_de.get_embedding(word)
                    self.assertIsNotNone(embedding, f"Embedding für '{word}' sollte nicht None sein")
                except ValueError as e:
                    self.fail(f"Wort '{word}' nicht im Vokabular gefunden")

    def test_umlaut_handling(self):
        """Test ob Wörter mit Umlauten korrekt behandelt werden"""
        umlaut_words = [
            'mädchen', 'größe', 'schön', 
            'münchen', 'köln', 'düsseldorf',
            'österreich', 'übung', 'ärzte'
        ]
        
        for word in umlaut_words:
            with self.subTest(word=word):
                try:
                    embedding = self.handler_de.get_embedding(word)
                    self.assertIsNotNone(embedding)
                except ValueError as e:
                    self.fail(f"Umlaut-Wort '{word}' nicht gefunden")

    def test_case_sensitivity(self):
        """Test ob Groß-/Kleinschreibung korrekt gehandhabt wird"""
        word_pairs = [
            ('Berlin', 'berlin'),
            ('DEUTSCHLAND', 'deutschland'),
            ('König', 'könig'),
            ('Österreich', 'österreich')
        ]
        
        for upper, lower in word_pairs:
            with self.subTest(upper=upper, lower=lower):
                try:
                    emb_upper = self.handler_de.get_embedding(upper)
                    emb_lower = self.handler_de.get_embedding(lower)
                    # Vektoren sollten identisch sein
                    np.testing.assert_array_almost_equal(emb_upper, emb_lower)
                except ValueError as e:
                    self.fail(f"Problem bei Groß-/Kleinschreibung: {str(e)}")

    def test_embedding_dimensions(self):
        """Test ob Embeddings die richtige Dimension haben"""
        test_words = ['test', 'beispiel', 'python']
        expected_dim = 100  # FastText/GloVe Dimension
        
        for word in test_words:
            with self.subTest(word=word):
                embedding = self.handler_de.get_embedding(word)
                self.assertEqual(embedding.shape, (expected_dim,))

    def test_basic_analogies(self):
        """Test von grundlegenden Analogien die funktionieren sollten"""
        test_cases = [
            {
                'word1': 'könig', 
                'word2': 'königin', 
                'word3': 'mann',
                'expected': 'frau',
                'min_score': 0.7
            },
            {
                'word1': 'deutschland', 
                'word2': 'berlin', 
                'word3': 'frankreich',
                'expected': 'paris',
                'min_score': 0.7
            },
            {
                'word1': 'mann', 
                'word2': 'frau', 
                'word3': 'junge',
                'expected': 'mädchen',
                'min_score': 0.6
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                results, _, _ = self.handler_de.find_analogy(
                    case['word1'], 
                    case['word2'], 
                    case['word3']
                )
                
                # Prüfen ob erwartetes Wort in Top-5 ist
                result_words = [word for word, score in results]
                self.assertIn(
                    case['expected'], 
                    result_words, 
                    f"'{case['expected']}' sollte in den Top-5 sein für {case}"
                )
                
                # Prüfen ob Score über Minimum
                result_dict = dict(results)
                score = result_dict.get(case['expected'], 0)
                self.assertGreater(
                    score, 
                    case['min_score'],
                    f"Score für {case['expected']} zu niedrig: {score}"
                )

    def test_model_loading(self):
        """Test ob die Modelle korrekt geladen werden"""
        # Prüfe ob Modell geladen wurde
        self.assertIsNotNone(self.handler_de.model)
        self.assertIsNotNone(self.handler_en.model)
        
        # Prüfe ob Wortliste geladen wurde
        self.assertIsNotNone(self.handler_de.word_list)
        self.assertIsNotNone(self.handler_en.word_list)
        
        # Prüfe ob Wortliste nicht leer ist
        self.assertGreater(len(self.handler_de.word_list), 0)
        self.assertGreater(len(self.handler_en.word_list), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)