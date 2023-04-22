from googleapiclient.discovery import build
import csv
import datetime
import calendar
import time
import random
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import your module from the parent directory
from utils import get_data_file_path, create_csv, read_last_date_from_csv, load_api_key

def enter_date():
    date = input("> ")
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        return date
    except ValueError:
        print("Invalid date format. Please enter a date in the format YYYY-MM-DD")
        return enter_date()
    

def enter_start_date():
    while True:
        print("1. Use last date in transfer_rumours_articles.csv 2. Enter a start date in the format YYYY-MM-DD")
        option = input("> ")
        if option == "1":
            last_date = read_last_date_from_csv('transfer_rumours_articles.csv')
            if last_date:
                start_date = last_date + datetime.timedelta(days=1)
                return start_date
            else:
                print("transfer_rumours_articles.csv is empty. Please enter a start date in the format YYYY-MM-DD")
        elif option == "2":
            start_date = enter_date()
            return start_date
        else:
            print("Invalid option")

  
def enter_end_date(start_date):
    while True:
        print("Enter the end date in the format YYYY-MM-DD")
        end_date = enter_date()
        if end_date >= start_date:
            return end_date
        else:
            print("End date must be after start date. Please try again.")

def generate_article_dates():
    start_date = enter_start_date()
    end_date = enter_end_date(start_date)
    article_dates = build_article_dates_list(start_date, end_date)
    return article_dates


def build_article_dates_list(start_date, end_date):
    # Create a list of month names
    month_names = [calendar.month_name[i] for i in range(1, 13)]

    # Generate the list of dates
    article_dates = []
    current_date = start_date
    while current_date <= end_date:
        # Format the date as "day month year" and append it to the list
        formatted_date = f"{current_date.day} {month_names[current_date.month - 1]} {current_date.year}"
        article_dates.append(formatted_date)

        # Increment the date by one day
        current_date += datetime.timedelta(days=1)

    return article_dates

def perform_search(service, query, cx_id):
    search_params = {"q": query, "cx": cx_id, "num": 1}
    response = service.cse().list(**search_params).execute()
    return response


def process_search_result(response):
    if "items" in response:
        item = response["items"][0]
        return item["link"]
    else:
        return None

def sleep_between_searches(iteration):
    if iteration % 8 == 7 or iteration % 9 == 8 or iteration % 10 == 9:
        sleep_time = random.randint(180, 360)
    else:
        sleep_time = random.randint(5, 60)
    print(f"Sleeping for {sleep_time} seconds...")
    time.sleep(sleep_time)

def search_for_articles(api_key, cx_id, query, article_dates):
    service = build("customsearch", "v1", developerKey=api_key)

    with open(get_data_file_path('transfer_rumours_articles.csv'), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)

        for i, date in enumerate(article_dates):
            print(f"Searching for articles published on {date}...")
            query_with_date = f"{query} \"{date}\""
            response = perform_search(service, query_with_date, cx_id)
            result_link = process_search_result(response)

            if result_link:
                writer.writerow([date, result_link])
                print(f"Article found: {result_link}")
            else:
                print(f"No articles found for {date}.")

            sleep_between_searches(i)

    print("CSV file generation complete.")

def main():
    # Set up the search parameters
    google_api_key = load_api_key("GOOGLE_API_KEY")
    cx_id = load_api_key("CX_ID")
    
    create_csv('transfer_rumours_articles.csv', ['date', 'link'])

    query = "transfer rumours"
    article_dates = generate_article_dates()

    # Search for articles and write results to CSV file
    search_for_articles(google_api_key, cx_id, query, article_dates)

if __name__ == "__main__":
  main()
