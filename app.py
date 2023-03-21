import streamlit as st
from scripts import prompt_matching
import pandas as pd
import sys
sys.path.append("../")

# Set page config
st.set_page_config(page_title="Book Search", page_icon=":books:", layout="wide")

books = pd.read_csv('data/books_with_summaries.csv')

# Define function to search for relevant book summaries using keyword matching
def search_books_by_keyword(prompt):
    """Searches for the most relevant books using keyword matching

    Args:
        prompt (str): any prompt to search for books

    Returns:
        matched (pd.DataFrame): A dataframe containing the books that have the necessary keywords
    """
    pm = prompt_matching.PromptMatching()
    pm.keyword_matching(prompt, books)
    matched = books[books['keywords_match']]
    return matched

# Define function to search for relevant book summaries using cosine similarity
def search_books_by_cosine_similarity(prompt,col, num_books=3):
    """Searches for the most relevant books using cosine similarity

    Args:
        prompt (str): any prompt to search for books
        col (str): the column containing the summary of interest
        num_books (int, optional): The number of books to return. Defaults to 3.

    Returns:
        pd.DataFrame: A dataframe containing the books that have the highest cosine similarity scores
    """
    pm = prompt_matching.PromptMatching()
    pm.cosine_similarity(prompt, books,col)
    matched = books
    matched = matched.sort_values(by='cosine_similarity', ascending=False)
    return matched.head(num_books)

def display_results(results):
    """Displays the results of the search in the streamlit app

    Args:
        results (pd.DataFrame): The top books resulting from the search
    """
    top_3_books = results.head(3)
    count = 0
    for index, row in top_3_books.iterrows():
        count+=1
        st.markdown(f"<h4 style='color: green'>Cosine Similarity: {row['cosine_similarity']}</h4>", unsafe_allow_html=True)
        st.markdown(f"### {count}: **{row['Title']}**")
        st.write(f"**Library Location:** {row['Location']} **Authors:** {row['Authors']}")
        with st.expander("Click to view summaries"):
            st.write(f"**Summary:** {row['Summary']}")
            st.write(f"**Extractive Summary:** {row['extractive_summary']}")
            st.write(f"**Abbreviated Summary:** {row['abbreviated_summary']}")


# Define Streamlit app
def app():
    st.title("Duke Libraries Book Search (NLP)")

    prompt = st.text_area("Enter a prompt:", height=100)

    # Create two columns for the buttons
    col1, col2, col3, col4 = st.columns(4)

    # Create a section for recommendations
    st.write("## Recommendations")
    # Create empty container for the search results
    results_container = st.empty()

    # Add search button for keyword matching
    with col1:
        if st.button("Keyword Matching", key="search_by_keyword_button"):
            if prompt:
                results = search_books_by_keyword(prompt)
                
    # Add search button for cosine similarity
    with col2:
        if st.button("Duke Summary Matching", key="search_by_summary"):
            if prompt:
                results = search_books_by_cosine_similarity(prompt,'Summary')

    with col3:
        if st.button("Extractive Summary Matching", key="search_by_extractive_summary"):
            if prompt:
                results = search_books_by_cosine_similarity(prompt,'extractive_summary')

    with col4:
        if st.button("Abbreviated Summary Matching", key="search_by_abbreviated_summary"):
            if prompt:
                results = search_books_by_cosine_similarity(prompt,'abbreviated_summary')
    
    # Display results under Recommendations
    if 'results' in locals():
        display_results(results)
        results_container.dataframe(results.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))
        

if __name__ == "__main__":
    app()