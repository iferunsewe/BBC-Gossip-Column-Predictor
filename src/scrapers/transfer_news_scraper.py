import pandas as pd
import os
import time
import random
from urllib.parse import urlparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import scrape_content, create_csv, get_data_file_path, read_last_date_from_csv

def get_articles(start_date=None):
    articles = pd.read_csv(get_data_file_path("transfer_rumours_articles.csv"))
    articles["date"] = pd.to_datetime(articles["date"])

    if start_date:
        start_date = pd.Timestamp(start_date)  # Convert start_date to a datetime64[ns] object
        articles = articles[articles["date"] > start_date]
    return articles["date"].tolist(), articles["link"].tolist()


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
                a_tag.extract()
            
            text = p.text.strip()
            if text:
                data_list.append((text, source, source_url))
    else:
        print(f"No qa-story-body element found on {url}")

    return data_list

def save_data_to_csv(date, data_list):
    formatted_date = date.strftime("%-d %B %Y")
    with open(get_data_file_path("transfer_news_data.csv"), "a", encoding='utf-8') as file:
        for text, source, source_url in data_list:
            cleaned_text = '"' + text.replace('\n', ' ').replace('"', '""') + '"'
            quoted_source_url = '"' + source_url.replace('"', '""') + '"'
            file.write(f"{formatted_date},{cleaned_text},{source},{quoted_source_url}\n")

def scrape_and_save_data(dates, links):
    for date, link in zip(dates, links):
        if not link:
            continue
        print(f"Scraping {date.date()}...")
        soup = scrape_content(link)
        data_list = extract_data(soup, link)

        save_data_to_csv(date.date(), data_list)

        sleep_time = random.uniform(2, 4)

        print(f"Scraped {date.date()} : {link}. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

def main():
    create_csv("transfer_news_data.csv", ["date", "raw_text", "source", "source_url"])

    latest_date = read_last_date_from_csv("transfer_news_data.csv")
    start_date = latest_date + pd.Timedelta(days=1) if latest_date else None

    dates, links = get_articles(start_date=start_date)

    scrape_and_save_data(dates, links)

if __name__ == "__main__":
    main()
