import streamlit as st
from scripts import prompt_matching
import pandas as pd
import sys
sys.path.append("../")

books = pd.read_csv('data/duke_books.csv')
books = books.drop_duplicates(subset=['Title'])
books = books.dropna(subset=['Summary'])

# Define function to search for relevant book summaries using keyword matching
def search_books_by_keyword(prompt):
    pm = prompt_matching.PromptMatching()
    pm.keyword_matching(prompt, books)
    matched = books[books['keywords_match']]
    matched['Summary'] = matched['Summary'].str[:100] + '...'
    return matched

# Define function to search for relevant book summaries using cosine similarity
def search_books_by_cosine_similarity(prompt):
    pm = prompt_matching.PromptMatching()
    pm.cosine_similarity(prompt, books)
    matched = books[books['cosine_similarity']>0.75]
    matched['Summary'] = matched['Summary'].str[:100] + '...'
    return matched

# Define function to search for relevant book summaries using bert similarity
def search_books_by_bert_similarity(prompt):
    pm = prompt_matching.PromptMatching()
    pm.bert_matching(prompt, books)
    matched = books[books['bert_cosine_similarity']>0.75]
    matched['Summary'] = matched['Summary'].str[:100] + '...'
    return matched

# Define Streamlit app
def app():
    st.title("Book Search")
    
    # Create two columns for the buttons
    col1, col2, col3 = st.columns(3)
    
    # Create empty container for the search results
    results_container = st.empty()
    
    prompt = st.text_area("Enter a prompt:", height=100)
    
    # Add search button for keyword matching
    with col1:
        if st.button("Search by Keyword Matching", key="search_by_keyword_button"):
            if prompt:
                results = search_books_by_keyword(prompt)
                results_container.dataframe(results.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))
    
    # Add search button for cosine similarity
    with col2:
        if st.button("Search by Cosine Similarity", key="search_by_cosine_button"):
            if prompt:
                results = search_books_by_cosine_similarity(prompt)
                results_container.dataframe(results.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))

    # Add search button for bert similarity
    with col3:
        if st.button("Search by Bert Similarity", key="search_by_bert_button"):
            if prompt:
                results = search_books_by_bert_similarity(prompt)
                results_container.dataframe(results.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))


if __name__ == "__main__":
    app()