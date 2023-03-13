import requests
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf

# create class for scraping
class DukeLibrariesScraper:

    def get_doc_ids(self,pages):
        url = 'https://find.library.duke.edu/?f%5Bresource_type_f%5D%5B%5D=Book&page='
        perPage = '&per_page=100'
        
        doc_ids = []
        for page in range(101,pages+1):
            duke_url = url + str(page) + perPage
            results = requests.get(duke_url)
            soup = BeautifulSoup(results.content, "html.parser")
            h3_tags = soup.find_all('h3', class_='index_title')
            for h3_tag in h3_tags:
                id_value = h3_tag.get('id').split('-')[0]
                doc_ids.append(id_value)
                
        return doc_ids
    
    def get_book_details(self,doc_ids):
        
        # create an empty book dataframe with column names
        df = pd.DataFrame(columns=['Title', 'Location', 'Authors','Summary','Published','Language','System Details',
                           'Notes','Description','Description Details','Genre','OCLC','Other Identifiers',
                           'System ID'])

        for id in doc_ids['doc_id']:
            url = 'https://find.library.duke.edu/catalog/' + id
            results = requests.get(url)
            soup = BeautifulSoup(results.content, "html.parser")

            title = soup.find('title').text.strip() if soup.find('title') else ''
            location = soup.find('span', {'class': 'loc_b__DOCS'})
            location_text = location.text.strip() if location else ''
            authors_section = soup.find('div', {'id': 'authors'})
            authors = authors_section.find('a').text.strip() if authors_section and authors_section.find('a') else ''
            summary_section = soup.find('section', {'id': 'summary'})
            summary = summary_section.find('p').text.strip() if summary_section and summary_section.find('p') else ''
            meta_data = soup.find('dl', {'class': 'document-metadata'})
            published = meta_data.find('dd', {'class':'blacklight-imprint_main_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-imprint_main_a'}) else ''
            language = meta_data.find('dd', {'class':'blacklight-language_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-language_a'}) else ''
            system_details = meta_data.find('dd', {'class':'blacklight-note_system_details_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-note_system_details_a'}) else ''
            notes = meta_data.find('dd', {'class':'blacklight-note_general_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-note_general_a'}) else ''
            description = meta_data.find('dd', {'class':'blacklight-physical_description_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-physical_description_a'}) else ''
            description_details = meta_data.find('dd', {'class':'blacklight-physical_description_details_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-physical_description_details_a'}) else ''
            genre = meta_data.find('dd', {'class':'blacklight-genre_headings_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-genre_headings_a'}) else ''
            oclc = meta_data.find('dd', {'class':'blacklight-oclc_number'}).text.strip() if meta_data.find('dd', {'class':'blacklight-oclc_number'}) else ''
            other_identifiers = meta_data.find('dd', {'class':'blacklight-misc_id_a'}).text.strip() if meta_data.find('dd', {'class':'blacklight-misc_id_a'}) else ''
            system_id = meta_data.find('dd', {'class':'blacklight-local_id'}).text.strip() if meta_data.find('dd', {'class':'blacklight-local_id'}) else ''

            df = df.append({'Title':title, 'Location':location_text, 'Authors':authors,'Summary':summary,'Published':published,
                            'Language':language,'System Details':system_details,'Notes':notes,'Description':description,
                            'Description Details':description_details,'Genre':genre,'OCLC':oclc,
                            'Other Identifiers':other_identifiers,'System ID':system_id}, ignore_index=True)

        return df
            
    
# create main
if __name__ == "__main__":
    # check if GPU is available
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    # set up a TensorFlow session to use the GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('Memory growth enabled')

    # run the scraper to get doc IDs and save to a CSV file
    scraper = DukeLibrariesScraper()
    doc_ids = scraper.get_doc_ids(65702)
    df = pd.DataFrame({'doc_id': doc_ids})
    df.to_csv('duke_doc_ids.csv', index=False)
    
    '''doc_ids = pd.read_csv('data/duke_doc_ids.csv')
    print(doc_ids)
    df = scraper.get_book_details(doc_ids)
    df.to_csv('data/duke_books.csv', index=False)'''
