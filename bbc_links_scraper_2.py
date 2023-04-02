from googleapiclient.discovery import build
import csv
import datetime
import calendar
import time
import random


def build_article_dates(start_date=None, end_date=None):
    # Set the start and end dates
    if not start_date:
      print("1. Use last date in articles.csv 2. Enter a start date in the format YYYY-MM-DD")
      option = input("> ")
      if option == "1":
        with open("articles.csv", "r") as csvfile:
          reader = csv.reader(csvfile)
          rows = list(reader)
          last_row = rows[-1]
          start_date = datetime.datetime.strptime(
              last_row[0], "%d %B %Y").date() + datetime.timedelta(days=1)
      elif option == "2":
        start_date = input("> ")
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
      else:
        print("Invalid option")
        return

    if not end_date:
      print("Enter the end date in the format YYYY-MM-DD")
      end_date = input("> ")
      end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

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


def search_for_articles(api_key, cx_id, query, article_dates, new_file=True):
    # Set up the Google Custom Search API client
    service = build("customsearch", "v1", developerKey=api_key)

    # Define the search query parameters
    search_params = {"q": query, "cx": cx_id, "num": 1}

    mode = 'w' if new_file else 'a'  # determine whether to start a new file or append to an existing one

    # Open the CSV file for writing
    with open("articles2.csv", mode, newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write headers to the file if it's a new file
        if new_file:
            writer.writerow(["Date", "Link"])

        # Perform the search for each date and write the results to the CSV file
        for i, date in enumerate(article_dates):
            search_params["q"] = f"{query} \"{date}\""
            print(f"Searching for articles published on {date}...")

            # Send the search request to the API
            response = service.cse().list(**search_params).execute()

            # Extract the first search result and write it to the CSV file
            if "items" in response:
                item = response["items"][0]
                writer.writerow([date, item["link"]])
                print(f"Article found: {item['link']}")
            else:
                print(f"No articles found for {date}.")

            if i % 8 == 7 or i % 9 == 8 or i % 10 == 9:
                # Sleep for random time between 180 and 360 seconds
                sleep_time = random.randint(180, 360)
                print(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                # Randomize the sleep time between 5 and 60 seconds
                sleep_time = random.randint(5, 60)
                print(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)

    print("CSV file generation complete.")


def main():
  # Set up the search parameters
  api_key = "<api_key>"
  cx_id = "<cx_id>"
  query = "transfer rumours"
  article_dates = build_article_dates(datetime.date(2022, 3, 1), datetime.date(2022, 6, 1))

  # Search for articles and write results to CSV file
  search_for_articles(api_key, cx_id, query, article_dates, new_file=False)

if __name__ == "__main__":
  main()
