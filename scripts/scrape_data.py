# Imports
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Constants
DPL_SEARCH_RESULTS_URL = "https://durhamcounty.bibliocommons.com/v2/search?custom_edit=false&query=isolanguage%3A%22eng%22%20audience%3A%22adult%22%20formatcode%3A(BK%20)&searchType=bl&suppress=true"
URL_PREFIX = "https://durhamcounty.bibliocommons.com"

#############
# FUNCTIONS #
#############

def get_urls_to_scrape():
    # Scrape the search results page for the urls to each book
    results = requests.get(DPL_SEARCH_RESULTS_URL)
    soup = BeautifulSoup(results.content, "html.parser")
    
    # Extract the urls and book titles
    urls = []
    book_titles = []
    for book in soup.find_all('h2', class_='cp-title'):
        urls.append(URL_PREFIX + book.find('a').get('href'))
        book_titles.append(book.find('span', class_='title-content').text)
        
    # Convert lists to pandas dataframes
    book_df = pd.DataFrame(book_titles, columns=['title'])
    book_df['url'] = urls
    
    # Save the df to a csv file
    book_df.to_csv('book_urls.csv', index=False)
    
#################
# MAIN FUNCTION #
#################

if __name__ == "__main__":
    get_urls_to_scrape()