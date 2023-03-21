# create a class for matching prompt to book
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import warnings
warnings.filterwarnings('ignore')


class PromptMatching:

    def keyword_matching(self,prompt,books):
        """Takes a prompt and compares the keywords in the prompt to each book summary in the data. 
        If all keywords in the prompt are present in the summary, the book is considered a match.
        Otherwise, the book is not considered a match.

        Args:
            prompt (str): a prompt to match to books
            books (pd.DataFrame): a dataframe consiting of the book data
        """
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
        
    def cosine_similarity(self,prompt,books,col):
        """Calculates the cosine similarity between the prompt and each book summary in the data.

        Args:
            prompt (str): a prompt to match to books
            books (pd.DataFrame): a dataframe consiting of the book data
            col (str): The column containing the summary of interest
        """
        vectorizer = TfidfVectorizer()
        prompt_vector = vectorizer.fit_transform([prompt])

        # Loop through each book summary
        cosine_similarity_scores = []
        for summary in books[col]:
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
        
    def bert_matching(self,prompt, books):
        """Calculates the cosine similarity between the prompt and each book summary in the data using
        BERT embeddings instead of TF-IDF vectors.

        Args:
            prompt (str): a prompt to match to books
            books (pd.DataFrame): a dataframe consiting of the book data
        """
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        book_summaries = books['Summary'].tolist()
        book_ids = books.index.tolist()

        book_embeddings = []
        for summary in book_summaries:
            summary_tokens = tokenizer(summary, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            with torch.no_grad():
                summary_embedding = model(summary_tokens['input_ids'], summary_tokens['attention_mask'])[0][:, 0, :]
                book_embeddings.append(summary_embedding)

        book_embeddings = torch.cat(book_embeddings, dim=0)
        book_embeddings = book_embeddings.reshape(len(book_summaries), -1)
        print(book_embeddings)
        prompt_tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        with torch.no_grad():
            prompt_embedding = model(prompt_tokens['input_ids'], prompt_tokens['attention_mask'])[0][:, 0, :]

        similarities = cosine_similarity(prompt_embedding.reshape(1, -1), book_embeddings).squeeze().tolist()
        books['bert_cosine_similarity'] = similarities
        
    def combine_summaries(self):
        """Combines the summaries from the different models into one dataframe.
        """
        books_df = pd.read_csv('../data/duke_books.csv')
        abs_df = pd.read_csv('../data/duke_books_abstractive.csv')
        ext_df = pd.read_csv('../data/extractive_summary_df.csv')
        
        merged_data = books_df.copy()
        merged_data = pd.merge(merged_data, abs_df[['Title','abbreviated_summary']], how='inner', on='Title')
        merged_data = pd.merge(merged_data, ext_df[['Title','extractive_summary']], how='inner', on='Title')

        # drop duplicates based on the 'Title' column
        merged_data.drop_duplicates(subset=['Title'], inplace=True)
        merged_data.dropna(subset=['Summary'], inplace=True)

        # save the merged data to a new CSV file
        merged_data.to_csv('../data/books_with_summaries.csv', index=False)
        
    def get_matched_prompt_results(self,prompt,books,col):
        """Returns the cosine similarity score for the best matched book summary.

        Args:
            prompt (str): a prompt to match to books
            books (pd.DataFrame): a dataframe consiting of the book data
            col (str): The column containing the summary of interest
            
        Returns:
            pd.DataFrame: a dataframe containing the book that matches the prompt best

        """
        self.cosine_similarity(prompt,books,col)
        matched = books.sort_values(by='cosine_similarity', ascending=False)
        return matched.head(1)['cosine_similarity'].values[0]
        
    def run_validation_prompts(self):
        """Runs the large scale validation using the generated prompts.
        The results are saved to a CSV file.
        """
        validation_prompts = pd.read_csv('../data/validation_prompts.csv')
        validation_prompts = validation_prompts['prompt'].tolist()
        books = pd.read_csv('../data/books_with_summaries.csv')
        # create empty dataframe to store results
        results_df = pd.DataFrame(columns=['prompt', 'summary_cs', 'abb_summary_cs', 'ex_summary_cs'])
    
        for i, prompt in enumerate(validation_prompts):
            cs = self.get_matched_prompt_results(prompt,books,'Summary')
            ab_cs = self.get_matched_prompt_results(prompt,books,'abbreviated_summary')
            ext_cs = self.get_matched_prompt_results(prompt,books,'extractive_summary')
            
            # create a new row for the results dataframe
            new_row = {'prompt': prompt, 'summary_cs': cs, 'abb_summary_cs': ab_cs, 'ex_summary_cs': ext_cs}
        
            # append the row to the results dataframe
            results_df = results_df.append(new_row, ignore_index=True)
            
        results_df.to_csv('../data/prompts_with_cs.csv', index=False) 
        
    def calculate_summary_metrics(self):
        """Calculates the average cosine similarity for each summary type and prints the results.
        A boxplot is also created for each summary type and saved into the imgs folder. 
        """
        # Load the data
        data = pd.read_csv('../data/prompts_with_cs.csv')
        
        # Calculate the average cosine similarity for each prompt
        avg_base_cs = data.groupby('prompt')['summary_cs'].mean()
        avg_abb_cs = data.groupby('prompt')['abb_summary_cs'].mean()
        avg_extract_cs = data.groupby('prompt')['ex_summary_cs'].mean()
        
        # Print the results
        print(f"Average Cosine Similarity for Base Summary: {np.round(avg_base_cs.mean(), 4)}")
        print(f"Average Cosine Similarity for Abstractive Summary: {np.round(avg_abb_cs.mean(), 4)}")
        print(f"Average Cosine Similarity for Extractive Summary: {np.round(avg_extract_cs.mean(), 4)}")
        
        # Create a boxplot for each
        plt.boxplot([avg_base_cs, avg_abb_cs, avg_extract_cs], labels=['Base Summary', 'Abstractive Summary', 'Extractive Summary'])
        plt.title("Cosine Similarity for each Summary Type")
        plt.ylabel("Cosine Similarity")
        plt.savefig('../imgs/summary_boxplot.png') 
 
# create main for this class

if __name__ == "__main__":
    # initialize this class
    prompt_matching = PromptMatching()
    #prompt_matching.combine_summaries()
    #prompt_matching.run_validation_prompts()
    prompt_matching.calculate_summary_metrics()
    
    #prompt = "Find a book about a detective solving a murder mystery in a small town."
    #prompt = "Looking for a book that explores the changing role of religion in the 20th century. Specifically, how certain religious groups redefined what it meant to be religious and allowed their members the choice of what kind of God to believe in, or the option to not believe in God at all."
    
        


