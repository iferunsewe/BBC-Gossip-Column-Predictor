import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import calendar
import random
from urllib.parse import urlparse
from datetime import datetime

# Create or find the transfer_news_data.csv
def create_csv():
    if not os.path.isfile("../../data/transfer_news_data.csv"):
        with open("../../data/transfer_news_data.csv", "w") as file:
            file.write("date,raw_text,source,source_url\n")
    else:
        print("transfer_news_data.csv already exists")

# Read articles.csv and return the list of dates and links
def get_articles(start_date=None):
    articles = pd.read_csv("../../data/transfer_rumours_articles.csv")
    articles["Date"] = pd.to_datetime(articles["Date"])  # Convert the "Date" column to datetime64 data type

    if start_date:
        articles = articles[articles["Date"] > start_date]
    return articles["Date"].tolist(), articles["Link"].tolist()

# Scrape the content from a given URL
def scrape_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
    }
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    return soup

def extract_data(soup, url):
    story_body = soup.find(class_="qa-story-body")
    transfer_rumours_header = soup.find("h1", text=lambda t: t and "Transfer rumours:" in t)

    data_list = []
    if story_body and transfer_rumours_header:
        for p in story_body.find_all("p"):
            a_tag = p.find("a")
            if a_tag:
                source = urlparse(a_tag["href"]).netloc
                source_url = a_tag["href"]
                a_tag.extract()  # Remove the "a" tag from the paragraph
            
            text = p.text.strip()  # Get the text without the "a" tag
            if text:  # Check if the text is not empty
                data_list.append((text, source, source_url))
    else:
        print(f"No qa-story-body element found on {url}")

    return data_list


# Save the data to the csv file
def save_data_to_csv(date, data_list):
    with open("../../data/transfer_news_data.csv", "a", encoding='utf-8') as file:
        for text, source, source_url in data_list:
            cleaned_text = '"' + text.replace('\n', ' ').replace('"', '""') + '"'
            quoted_source_url = '"' + source_url.replace('"', '""') + '"'
            file.write(f"{date},{cleaned_text},{source},{quoted_source_url}\n")

def get_latest_date_from_csv():
    if os.path.isfile("../../data/transfer_news_data.csv"):
        transfer_news_data = pd.read_csv("../../data/transfer_news_data.csv")
        if not transfer_news_data.empty:
            latest_date = max(pd.to_datetime(transfer_news_data["date"]))
            return latest_date
    return None

def main(start_date=None):
    create_csv()

    if start_date:
        print(f"Start date provided: {start_date}")
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    latest_date = get_latest_date_from_csv()
    if latest_date and (not start_date or latest_date > start_date):
        print(f"Latest date from csv: {latest_date.date()}")
        start_date = latest_date
    
    if start_date:
        print(f"Start date: {start_date.date()}")

    # Create a list of month names
    month_names = [calendar.month_name[i] for i in range(1, 13)]

    dates, links = get_articles(start_date=start_date)
        
    for date, link in zip(dates, links):
        if not link:
            continue
        print(f"Scraping {date.date()}...")
        soup = scrape_content(link)
        data_list = extract_data(soup, link)

        formatted_date = f"{date.day} {month_names[date.month - 1]} {date.year}"
        save_data_to_csv(formatted_date, data_list)

        sleep_time = random.uniform(3, 10)

        print(f"Scraped {date.date()} : {link}. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main("2021-10-07")