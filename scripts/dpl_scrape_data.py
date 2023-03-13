# Imports
import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import unicodedata

# Constants
DPL_SEARCH_RESULTS_URL = "https://durhamcounty.bibliocommons.com/v2/search?custom_edit=false&query=isolanguage%3A%22eng%22%20audience%3A%22adult%22%20formatcode%3A(BK%20)&searchType=bl&suppress=true"
URL_PREFIX = "https://durhamcounty.bibliocommons.com"
DATA_FOLDER = os.path.join('..', 'data')

TITLE_JARGON = {'Read', 'an', 'excerpt', 'from', 'an', 'overlay', 'opens'}

#############
# FUNCTIONS #
#############
    
def get_urls_to_scrape(start_page=0, end_page=10_000):
    """Obtains all the urls to the books on the Durham Public Library website.

    Args:
        start_page (int, optional): Starting page of the search results. Defaults to 0.
        end_page (int, optional): Ending page of the search results. Defaults to 10_000.
    """
    
    # Initialize the webdriver
    driver = webdriver.Chrome()
    driver.get(DPL_SEARCH_RESULTS_URL + f"&page={start_page}")
    
    # For each page, extract the urls to each book
    for current_page in range(start_page, end_page): # Extract 118,600 Urls
        # Scroll to the bottom of the page until bottom is reached
        _scroll_down(driver)
            
        # Extract the URLs for each book
        try:
            elements = driver.find_elements(By.XPATH, "//h2[@class='cp-title']")
            urls = [element.find_element(By.XPATH, ".//a").get_attribute("href") for element in elements]
        except:
            continue
        
        # Append the urls to the csv file
        _append_to_csv_file(urls)
        
        # Scroll back up
        _scroll_up(driver)
        
        # Calculate what the next page will be
        next_page = driver.find_element(By.XPATH, "//li[@class='cp-pagination-item pagination__next-chevron']/a[@class='cp-link pagination-item__link']") # Extract link
        
        # Move to the next page and repeat
        next_page.click()
        time.sleep(8)
        
        # Print updates periodically
        if(current_page % 100 == 0):
            print(f"Extracted {current_page*10} urls")

def scrape_from_urls(starting_idx = 0, save_file_path=os.path.join(DATA_FOLDER, 'dpl_book_data.tsv'), save_mode='w'):
    """Scrapes the individual book data from the urls stored in book_urls.txt. 
    This includes information such as the title, author, rating, description, and subjects/genres.

    Args:
        save_mode (str, optional): Either 'a' for append or 'w' for write from scratch. Defaults to 'w'.
        starting_idx (int, optional): Starting url index. Defaults to 0.
        save_file_path (str, optional): Path to the tsv for storing data. Defaults to '../data/book_data.csv'.
    """
    # Load the data from the txt file
    with(open(os.path.join(DATA_FOLDER, 'dpl_book_urls.txt'), 'r')) as f:
        urls = f.readlines()
        
    # Add headers to the tsv file
    if(save_mode == 'w'):
        with(open(save_file_path, 'w')) as f:
            f.write("title\tauthor\trating\tnum_ratings\tdescription\turl\n")
        
    # Initialize the webdriver
    options = Options().add_argument("--headless")
    driver = webdriver.Chrome(options=options)
        
    # Loop through the urls and scrape from each one
    for i in range(starting_idx, len(urls)):
        # Make request to the url and extract the html content
        url = urls[i]
        try:
            driver.get(url)
        except:
            continue # Not worth the hassle if url has issues, just skip it and move on
        
        # Expand the description if possible
        try:
            read_more_element = driver.find_element(By.XPATH, "//a[@class='cp-link cp-expand-link expandable-html__expand-button']")
            read_more_element.click()
        except:
            pass # Button may not exist and thats fine
        
        try:
            # Extract Title
            title = driver.find_element(By.XPATH, "//div[@class='cp-bib-title']//span[@class='cp-screen-reader-message']").text
            title = ' '.join([word for word in title.split() if word not in TITLE_JARGON])
            if(title[-1] == ','):
                title = title[:-1]
            
            # Extract Author
            author = driver.find_element(By.XPATH, "//div[@class='cp-author-link']//span[@class='cp-screen-reader-message']").text

            # Extract Rating
            rating_text = driver.find_element(By.XPATH, "//span[@class='cp-rating-stars rating-stars']/span[@class='cp-screen-reader-message']").text
            rating  = float(rating_text.split()[2])
            num_ratings = int(rating_text.split()[-2].replace(',', ''))
            
            # Extract Description
            description = driver.find_element(By.XPATH, "//div[@class='cp-bib-description']//div[@class='expandable-html__text']").text
            description = description.replace('\n', ' ')
            
            # Append the results to the tsv file
            string_to_save = f"{title}\t{author}\t{rating}\t{num_ratings}\t{description}\t{url}"
            string_to_save = _strip_accents_from_str(string_to_save)
            
            with(open(save_file_path, 'a')) as f:
                f.write(string_to_save)
        except:
            continue
            
        # Print updates periodically
        if(i % 100 == 0):
            print(f"Scraped {i} books")
            
####################
# HELPER FUNCTIONS #
####################
def _scroll_down(driver, pause_time=0.5):
    """Scrolls from the top of the page to the bottom of the page. This is used to load all of the books on the page.

    Args:
        driver (webdriver): Selenium webdriver object to handle the browser.
        pause_time (float, optional): amount of time to pause while scrolling. Defaults to 0.5.
    """
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
def _scroll_up(driver, pause_time=0.5):
    """Scrolls from the bottom of the page to the top of the page.
    This allows us to move onto the next page autmoatically.

    Args:
        driver (_type_): Selenium webdriver object to handle the browser.
        pause_time (float, optional): amount of time to pause while scrolling. Defaults to 0.5.
    """
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, 0);")

        # Wait to load page
        time.sleep(pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        

def _append_to_csv_file(urls, filename=os.path.join(DATA_FOLDER,'book_urls.txt'), mode='a'):
    """Appends the urls to the csv file.
    
    Args:
        urls (list): list of urls to append to the csv file.
        filename (str, optional): name of the file to append to. Defaults to ../data/book_urls.txt.
        mode (str, optional): mode to open the file in. Defaults to append mode 'a'.
    """
    # Join the filenames with a newline
    url_string = "\n".join(urls) + "\n"

    # Append the urls to the file
    with(open(filename, mode)) as f:
        f.write(url_string)
        

def _strip_accents_from_str(text):
    """Removes accents from a string so it can follow utf-8 format.
    
    Args:
        text (str): string to remove accents from.
    
    Returns:
        text_normalized (str): string with accents removed.
    """
    text_normalized = unicodedata.normalize('NFKD', text)
    text_normalized = text_normalized.encode('ASCII', 'ignore')
    text_normalized = text_normalized.decode("utf-8")
    return text_normalized
        
#################
# MAIN FUNCTION #
#################

if __name__ == "__main__":
    # Download the urls for the books
    #start_page = 0
    #end_page = 3145
    #print(f"Scraping the urls starting from page {start_page} to page {end_page}. This may take a long time...")
    #get_urls_to_scrape(start_page=start_page, end_page=end_page)
    
    # Scrape the individual book data from the urls
    start_url_idx = 333
    save_mode = 'a'
    print("Scraping the individual book data from the urls. This may take a long time...")
    scrape_from_urls(starting_idx=start_url_idx, save_mode=save_mode, save_file_path=os.path.join(DATA_FOLDER, 'dpl_book_data.tsv'))