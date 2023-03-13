# create a class for matching prompt to book
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PromptMatching:

    def keyword_matching(self,prompt,books):
        # Prompt and relevant keywords
        vectorizer = CountVectorizer()
        keywords = vectorizer.fit_transform([prompt]).toarray()[0]

        # Loop through each book summary
        keywords_match = []
        for summary in books['Summary']:
            if isinstance(summary, str):
                # Check if summary contains all relevant keywords
                book_keywords = vectorizer.transform([summary]).toarray()[0]
                match = all(book_keywords[i] >= keywords[i] for i in range(len(keywords)))
            else:
                # If summary is NaN, set match to False
                match = False
            keywords_match.append(match)

        # Add keywords_match column to books dataframe
        books['keywords_match'] = keywords_match
        
    def cosine_similarity(self,prompt,books):
        vectorizer = TfidfVectorizer()
        prompt_vector = vectorizer.fit_transform([prompt])

        # Loop through each book summary
        cosine_similarity_scores = []
        for summary in books['Summary']:
            if isinstance(summary, str):
                # Calculate cosine similarity between summary and prompt
                book_vector = vectorizer.transform([summary])
                similarity = cosine_similarity(prompt_vector, book_vector)[0][0]
            else:
                # If summary is NaN, set similarity score to NaN
                similarity = np.nan
            cosine_similarity_scores.append(similarity)

        # Add cosine_similarity column to books dataframe
        books['cosine_similarity'] = cosine_similarity_scores
        
# create main for this class

if __name__ == "__main__":
    # initialize this class
    prompt_matching = PromptMatching()
    books = pd.read_csv('data/duke_books.csv')
    #prompt = "Find a book about a detective solving a murder mystery in a small town."
    prompt = "Looking for a book that explores the changing role of religion in the 20th century. Specifically, how certain religious groups redefined what it meant to be religious and allowed their members the choice of what kind of God to believe in, or the option to not believe in God at all."

    prompt_matching.keyword_matching(prompt,books)
    prompt_matching.cosine_similarity(prompt,books)
    
    # check if cosine similarity is > 0.6 or keywords_match is true
    matched = books[(books['cosine_similarity']>0.75) | (books['keywords_match'])]
    print(matched)
        


