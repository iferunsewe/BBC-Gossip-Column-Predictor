from googlesearch import search
import csv
import datetime
import calendar
import time
import random
import os

def build_article_dates(start_date=None, end_date=None):
    # Set the start and end dates
    if not start_date:
      print("1. Use last date in transfer_rumours_articles.csv 2. Enter a start date in the format YYYY-MM-DD")
      option = input("> ")
      if option == "1":
        with open("/../data/transfer_rumours_articles.csv", "r") as csvfile:
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

def build_article_csv(article_dates, new_file=True):
  query = "site:bbc.com/sport/ \"transfer rumours:\""
  print("Starting to build CSV file...")

  mode = 'w' if new_file else 'a'  # determine whether to start a new file or append to an existing one

  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_dir = os.path.join(script_dir, '..', '..', 'data')
  output_file = os.path.join(data_dir, 'transfer_rumours_articles.csv')
  # Open the CSV file for writing or appending
  with open(output_file, mode, newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write headers to the file if it's a new file
    if new_file:
      writer.writerow(["Date", "Link"])

    search_count = 1
    for date in article_dates:
      if search_count % 8 == 0 and search_count % 9 == 0 and search_count % 10 == 0:
        print("Skipping search due to cooldown period...")
        cooldown_time = random.randint(180, 360)
        print(f"Sleeping for {cooldown_time} seconds...")
        time.sleep(cooldown_time)
        continue

      search_query = f"{query} \"{date}\""
      print(f"Searching for articles published on {date}...")
      for url in search(search_query, num_results=1):
        writer.writerow([date, url])
        print(f"Article found: {url}")
        break

      # Randomize the sleep time between 5 and 30 seconds
      sleep_time = random.randint(5, 30)
      print(f"Sleeping for {sleep_time} seconds...")
      time.sleep(sleep_time)

      search_count += 1

  print("CSV file generation complete.")

if __name__ == "__main__":
  article_dates = build_article_dates(datetime.date(2021, 6, 1), datetime.date(2022, 1, 31))
  build_article_csv(article_dates, new_file=True)  # append to an existing file
