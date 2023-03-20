# Imports
import os
import openai
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_dewey_topics():
    all_topics = []
    url = "https://en.wikipedia.org/wiki/List_of_Dewey_Decimal_classes#Class_000_%E2%80%93_Computer_science,_information_and_general_works"

    # Make the request to the website
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Get the different topics
    macro_topics = soup.find("div", {"class": "mw-parser-output"}).find_all("ul")
    for macro_topic_html in macro_topics:
        # Get and clean the topics
        topics = macro_topic_html.text.split("\n")[1:]
        topic_categories = []
        for i, topic in enumerate(topics):
            if(topic[0].isdigit()==True and "unassigned" not in topic.lower()):
                if(topic[2] == '0' and topics[i+1][2] == '0'):
                    continue
                topic_categories.append(topic[3:].lower().strip())        
                
        all_topics.extend(topic_categories)
        
    # Convert the list to a dataframe
    topic_df = pd.DataFrame(all_topics, columns=["topic"])
    topic_df = topic_df.drop_duplicates().reset_index(drop=True)

    # Return the topics
    return topic_df
    
def generate_prompts(topics_df):
    topics_df['prompt'] = topics_df.apply(lambda x: _create_prompt(x['topic']), axis=1)    
    return topics_df
    
def load_validation_prompts():
    topics_df = pd.read_csv("../data/validation_prompts.csv")
    return topics_df
    
def _create_prompt(topic):
    openings = ["Find a book about ", "I'm looking for a book about ", "I want to read a book about ", "I need a recommendation for a book about "]
    
    # Select a random opening
    random_number = np.random.randint(0, len(openings))
    opening = openings[random_number]

    # Use the gpt-3.5 api
    prompt = f"generate a prompt with the following opening looking for a book with the following topic\nopening: {opening}\ntopic: {topic}"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message
        )
    
        # Return the response
        return response['choices'][0]['message']['content']
    except:
        return ""
    
    
def clean_prompts(topics_df):
    # Load the prompts
    topics_df = load_validation_prompts()
    
    # Drop any prompts that are not of type string
    topics_df_cleaned = topics_df[topics_df['prompt'].apply(lambda x: isinstance(x, str))]
    
    # Drop any prompts that are empty strings
    topics_df_cleaned = topics_df_cleaned[topics_df_cleaned['prompt'] != ""]
    
    # If a prompt contains a newline, remove everything before and including the newline
    topics_df_cleaned["prompt"] = topics_df_cleaned.apply(lambda x: x['prompt'].split("\n")[-1], axis=1)
    
    # Drop na
    topics_df_cleaned = topics_df_cleaned.dropna(inplace=False)
    
    # Ensure there is only one quote on each side of the prompt
    topics_df_cleaned["prompt"] = topics_df_cleaned.apply(lambda x: x['prompt'].replace("\"", ""), axis=1)
    
    # Remove rows where the topic contains no longer used
    topics_df_cleaned = topics_df_cleaned[topics_df_cleaned['topic'].apply(lambda x: "no longer used" not in x)]
    
    return topics_df_cleaned
    
def save_prompts(topics_df):
    topics_df.to_csv("../data/validation_prompts.csv", index=False)
    
if __name__ == "__main__":
    # Generate the prompts
    topics_df = get_dewey_topics()
    topics_df = generate_prompts(topics_df)
    
    # Clean and save the prompts
    topics_df = clean_prompts(topics_df)
    save_prompts(topics_df)