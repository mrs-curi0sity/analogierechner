import streamlit as st
from streamlit.components.v1 import html
from src.core.embedding_handler import EmbeddingHandler
import sys
import os
from src.core.logger import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_resource
def load_embedding_handler(language):
    # Debug-Ausgabe
    st.write(f"Loading handler for language: {language}")
    lang = 'de' if language == 'Deutsch' else 'en'
    st.write(f"Using language code: {lang}")
    return EmbeddingHandler(language=lang)



def main():
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    st.title("Analogierechner: Semantische Analyse")
    
    language = st.sidebar.selectbox(
        "Sprache / Language",
        ["Deutsch", "English"],
        index=1,
        key='language_selector'
    )
    
    app_mode = st.sidebar.selectbox(
        "Wähle eine Funktion", 
        ["Analogie-Suche", "Ähnliche Wörter", "Über das Projekt"]
    )
    
    # Language direkt übergeben
    embedding_handler = load_embedding_handler(language)
        
    if app_mode == "Analogie-Suche":
        st.header("Wörter-Rechner")
        
        st.info("""
        Der Wörter-Rechner findet Beziehungen zwischen Wörtern.
        Beispiel: Germany ↔ Berlin = Spain ↔ ?
        """)


        # Eingabefelder in zwei Spalten
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Erste Beziehung")
            word1 = st.text_input("Startwort (z.B. Germany)", "Germany")
            word2 = st.text_input("Zielwort (z.B. Berlin)", "Berlin")
            st.markdown(f"**{word1}** ↔ **{word2}**")
            
        with col2:
            st.markdown("### Zweite Beziehung")
            word3 = st.text_input("Startwort (z.B. Spain)", "Spain")
            # Platzhalter mit gleicher Höhe wie das Eingabefeld
            st.text_input("Ergebnis", "", disabled=True, key="result_placeholder")
            if 'results' in locals() and results:
                st.markdown(f"**{word3}** ↔ **{results[0][0]}**")
            else:
                st.markdown(f"**{word3}** ↔ **?**")


    if st.button("Analogie berechnen"):
        with st.spinner('Berechne Analogie...'):
            try:
                results, _ = embedding_handler.find_analogy(word1, word2, word3, "")

                # Logging
                logger.log(
                    'ui',
                    language,  # 'en' oder 'de' aus dem Selector
                    word1,
                    word2,
                    word3,
                    best_result
                )

                
                # Große, klare Ergebnisaussage
                st.markdown("---")
                st.markdown(f"## 🎯 Ergebnis:")
                st.markdown(f"""
                ### {word1} verhält sich zu {word2} wie {word3} zu {results[0][0]}
                """)
                st.markdown("---")

                
                # Weitere Details ausklappbar
                with st.expander("Details und weitere Vorschläge", expanded=True):
                    st.markdown("#### Alternative Vorschläge:")
                    for word, score in results[1:5]:
                        st.markdown(f"- {word} (Score: {score:.3f})")
                    
            except Exception as e:
                st.error(f"Fehler bei der Berechnung: {str(e)}")
    
        # Beispiele am Ende
        with st.expander("📚 Beispiele für Wortbeziehungen"):
            st.markdown("""
            - Mann ↔ König = Frau ↔ Königin
            - Frankreich ↔ Paris = Italien ↔ Rom
            - Auto ↔ Straße = Zug ↔ Schiene
            """)
        
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