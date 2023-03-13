import streamlit as st
from scripts import prompt_matching
import pandas as pd
import sys
sys.path.append("../")

books = pd.read_csv('data/duke_books.csv')

# Define function to search for relevant book summaries
def search_books(prompt):
    pm = prompt_matching.PromptMatching()
    pm.keyword_matching(prompt, books)
    pm.cosine_similarity(prompt, books)
    matched = books[(books['cosine_similarity']>0.75) | (books['keywords_match'])]
    matched['Summary'] = matched['Summary'].str[:100] + '...'
    return matched

# Define Streamlit app
def app():
    st.title("Book Search")
    prompt = st.text_area("Enter a prompt:", height=100)
    if st.button("Search", key="search_button"):
        if prompt:
            results = search_books(prompt)
            st.write("")
            st.write("")
            st.write("")
            st.dataframe(results.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))


if __name__ == "__main__":
    app()