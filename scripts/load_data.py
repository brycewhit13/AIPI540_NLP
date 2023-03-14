# Imports and Constants
import os
import pandas as pd

DATA_FOLDER_PATH = os.path.join("..", "data")

##### Functions #####

def load_duke_library_data():
    duke_book_data = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "duke_books.csv")).drop_duplicates()
    return duke_book_data

def load_dpl_data():
    dpl_book_data = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "dpl_book_data.tsv"), sep='\t').drop_duplicates()
    return dpl_book_data

def load_combined_data():
    # Load the data separately
    duke_data = load_duke_library_data()
    dpl_data = load_dpl_data()
    
    # Get the columns of interest
    duke_data_subset = duke_data[['Title', 'Authors', 'Summary', 'System ID']].rename(columns={'Title': 'title', 'Authors': 'author', 'Summary': 'summary'})
    dpl_data_subset = dpl_data[['title', 'author', 'description', 'url']].rename(columns={'description': 'summary'})

    # Combine the data
    combined_data = pd.concat([duke_data_subset, dpl_data_subset], ignore_index=True).drop_duplicates()
    
    # Return the combined data
    return combined_data