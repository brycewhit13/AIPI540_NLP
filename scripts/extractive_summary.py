#load imports 
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

#download necessary library from nltk 
nltk.download('stopwords')
nltk.download('punkt')

class ExtractiveSummaryGenerator: 

    def __init__(self, filename):
        '''Initializes the class for extractive summaries
        - reads book data in from CSV file
        - drops books with summary NA
        
        Returns: Book pandas data frame with the summry dropped if value is NA
        '''

        #reads in the books data file 
        self.books = pd.read_csv(filename)

        #drop NA values for now
        self.books = self.book_data.dropna(subset = ['Summary']).reset_index(drop = True)
    
    def word_count(self, text):
        '''
        Returns the number of words in a sentence
        
        '''
        words = text.split()
        return len(words)
    
    def generate_summary(self, input_text): 
        '''
        Creates extractive summaries from the input text
        - tockenizes input text 
        - strips alpha numeric characters and stopwords from input text 
        - instantiates TF-IDF feature vectorizer, fit/transforms stripped sentences 
        - creates adjacency matrix and populates it with the sentence vectors 
        - populates the adjacency matrix using cosine distance as similarity metric
        - creates a corresponding document graph
        - applies PageRank algorithim to the document graph to rank sentences
        - outputs the ranked sentence - either 5 or the maximum number of sentences in the summary
        - stores count failures of the Page Rank algorithim

        Returns: Extracted Summary (str), Count failure, 0 or 1 (int)
        
        '''
        #tokenize sentences
        sentences = sent_tokenize(input_text)

        #strip alpha numeric characters and stopwords 
        sentences_processed = []
        for sentence in sentences:
            sentence_reduced = sentence.replace("[^a-zA-Z0-9_]", '')
            sentence_reduced = [word.lower() for word in sentence_reduced.split(' ') if word.lower() not in stopwords.words('english')]
            sentences_processed.append(' '.join(word for word in sentence_reduced))

        #create TFIDF feature vecs
        vectorizer = TfidfVectorizer()
        feature_vecs = vectorizer.fit_transform(sentences_processed)
        feature_vecs = feature_vecs.todense().tolist()

        # Create empty adjacency matrix
        adjacency_matrix = np.zeros((len(feature_vecs), len(feature_vecs)))
 
        # Populate the adjacency matrix using the similarity of all pairs of sentences
        for i in range(len(feature_vecs)):
            for j in range(len(feature_vecs)):
                if i == j: #ignore if both are the same sentence
                    continue 
                adjacency_matrix[i][j] = 1 - cosine_distance(feature_vecs[1], feature_vecs[j])

        # Create the graph representing the document
        document_graph = nx.from_numpy_array(adjacency_matrix)

        #initialize count failures so we can keep track of them 
        count_failures = 0
    
        # Apply PageRank algorithm to get centrality scores for each node/sentence
        try:
            scores = nx.pagerank(document_graph)
            scores_list = list(scores.values())

            # Sort and pick top sentences
            ranking_idx = np.argsort(scores_list)[::-1]
            ranked_sentences = [sentences[i] for i in ranking_idx]   

            summary = []
            top_n = min(len(ranked_sentences), 5)
            for i in range(top_n):
                summary.append(ranked_sentences[i])

            summary = " ".join(summary)
        except nx.PowerIterationFailedConvergence:
            count_failures += 1
            summary = input_text

        return summary, count_failures
    

    def extractive_summary(self,books):
        '''
        Applies the extractive summary function to the data frame
        - generates word count column to ID longer summaries 
        - filters data frame for only books where the summary is longer than 100 words
        - drops everything from data frame besides title and summary and creates full title column 
        - applies generate summary function to full summary column and creates column to track count failures 
        - returns data frame extracted summary as a column 

        Returns: Data frame with the extracted summary as a column 
        '''
        books['word_count'] = books['Summary'].apply(self.word_count)
        #filter for books with longer summaries
        book_long = books[books['word_count'] >= 100] 
        
        #drop everything but title and summary 
        book_long = book_long.loc[:, ['Title', 'Summary']]

        #generate full text by concatenating Title and Summary columns
        book_long['full_text'] = book_long['Title'] + ' ' + book_long['Summary']

        #generate extractive summary and count failure (see how many times page rank failed to converge and we just filled with original)
        book_long['extractive_summary'], book_long['count_failures'] = zip(*book_long['full_text'].apply(self.generate_summary))

        return book_long
    
if __name__ == "__main__":
    summary_generator = ExtractiveSummaryGenerator('data/duke_books.csv')

