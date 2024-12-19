import streamlit as st
from src.core.embedding_handler import EmbeddingHandler

# Initialisiere den Embedding Handler
@st.cache_resource
def load_embedding_handler():
    return EmbeddingHandler()

def main():
    st.title("Analogierechner: Semantische Analyse")
    
    # Sidebar für Navigations-Optionen
    app_mode = st.sidebar.selectbox("Wähle eine Funktion", 
        ["Analogie-Suche", "Ähnliche Wörter", "Über das Projekt"])

    language = st.sidebar.selectbox(
        "Sprache / Language",
        ["Deutsch", "English"],
        index=0
    )

        
    @st.cache_resource
    def load_embedding_handler():
        lang = 'de' if language == 'Deutsch' else 'en'
        return EmbeddingHandler(language=lang)
    
    # Embedding Handler laden
    embedding_handler = load_embedding_handler()
    
    if app_mode == "Analogie-Suche":
        st.header("Analogie-Finder")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            word1 = st.text_input("Erstes Wort", "Sandkorn")
        with col2:
            word2 = st.text_input("Zweites Wort", "Wüste")
        with col3:
            word3 = st.text_input("Zielwort", "Tropfen")
            expected_result = st.text_input("Erwartetes Ergebnis (optional)", "")

        if st.button("Analogie berechnen"):
            try:
                results, expected_similarity = embedding_handler.find_analogy(word1, word2, word3, expected_result)
                
                st.subheader("Ergebnisse:")
                for word, score in results:
                    st.write(f"{word}: {score:.3f}")
                
                if expected_result:
                    st.write(f"Erwartetes Ergebniswort: {expected_result}")
                    st.write(f"Erwartete Ähnlichkeit: {expected_similarity:.3f}")
            except Exception as e:
                st.error(f"Fehler bei der Berechnung: {str(e)}")

    
    
    elif app_mode == "Ähnliche Wörter":
        st.header("Wort-Ähnlichkeits-Suche")
        
        search_word = st.text_input("Suche ähnliche Wörter zu:", "Auto")
        top_n = st.slider("Anzahl der Ergebnisse", 3, 10, 5)
        
        if st.button("Suchen"):
            try:
                similar_words = embedding_handler.find_similar_words(search_word, top_n)
                
                st.subheader("Ähnliche Wörter:")
                for word, score in similar_words:
                    st.metric(label=word, value=f"{score:.3f}")
            
            except Exception as e:
                st.error(f"Fehler bei der Suche: {e}")
    
    else:
        st.header("Über Word Wanderer")
        st.markdown("""
        ### 🧠 Semantische Analyse mit Word Embeddings
        
        Word Wanderer nutzt moderne Machine-Learning-Techniken, um:
        - Semantische Beziehungen zwischen Wörtern zu analysieren
        - Analogien zu berechnen
        - Ähnliche Wörter zu finden
        
        #### Technologien:
        - Sentence Transformers
        - Python
        - Streamlit
        """)

if __name__ == "__main__":
    main()