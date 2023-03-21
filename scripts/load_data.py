# Imports and Constants
import os
import pandas as pd

DATA_FOLDER_PATH = os.path.join("..", "data")

##### Functions #####
def load_duke_library_data():
    """Loads the Duke University Library data

    Returns:
        duke_book_data (pd.DataFrame): A dataframe containing the Duke book data
    """
    duke_book_data = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "duke_books.csv")).drop_duplicates()
    return duke_book_data

def load_dpl_data():
    """Loads the Durham Public Library data

    Returns:
        dpl_book_data (pd.DataFrame): A dataframe containing the DPL book data
    """
    dpl_book_data = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "dpl_book_data.tsv"), sep='\t').drop_duplicates()
    return dpl_book_data