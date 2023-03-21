from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


class SummaryGenerator:
    
    def __init__(self, filename):
        """Constructor for SummaryGenerator class

        Args:
            filename (str): the path to the csv file containing the book data
        """
        self.book_data = pd.read_csv(filename)
        #drop NA values for now
        self.book_data = self.book_data.dropna(subset = ['Summary']).reset_index(drop = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        
    def word_count(self, text):
        """Returns the number of words in a string.

        Args:
            text (str): any string

        Returns:
            int: the number of words in the string
        """
        words = text.split()
        return len(words)

    def truncate_summary(self,input_text):
        """Truncates a summary to a maximum of 200 words.

        Args:
            input_text (str): any string to be used as a summary

        Returns:
            summary (str): the truncated summary
        """
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(inputs["input_ids"], max_length=200, min_length=100, length_penalty=1.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(outputs[0])
        summary = summary.split('</s>')[-2].split('<s>')[-1].strip()
        return summary 

    def abstractive_summary(self,books):
        """Generates an abstractive summary for each book in the dataframe.

        Args:
            books (pd.DataFrame): a dataframe containing the book data

        Returns:
            book_long (pd.DataFrame): a dataframe containing the book data and the new abstractive summaries
        """
        books['word_count'] = books['Summary'].apply(self.word_count)
        #filter for books with longer summaries
        book_long = books[books['word_count'] >= 100] 
        
        #drop everything but title and summary 
        book_long = book_long.loc[:, ['Title', 'Summary']]

        # generate full text by concatenating Title and Summary columns
        book_long['full_text'] = book_long['Title'] + ' ' + book_long['Summary']

        book_long['abbreviated_summary'] = book_long['full_text'].apply(self.truncate_summary)
        return book_long
    
    def save_abstractive_summaries(self):
        """Saves the abstractive summaries to a csv file.
        """
        # Assuming self.book_data is a pandas dataframe
        num_records = len(self.book_data)
        batch_size = 100

        for i in range(0, num_records, batch_size):
            batch = self.book_data.iloc[i:i+batch_size]
            results = self.abstractive_summary(batch)
            results.to_csv('data/duke_books_abstractive.csv', mode='a', header=(i==0))

if __name__ == "__main__":
    # initialize TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        # initialize this class
        summary_generator = SummaryGenerator('data/duke_books.csv')
        summary_generator.save_abstractive_summaries()
        
            