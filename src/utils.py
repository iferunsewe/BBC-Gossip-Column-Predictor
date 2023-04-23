import os
import datetime
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_data_file_path(filename):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    return os.path.join(data_dir, filename)

def read_last_date_from_csv(filename):
    print(f"Reading last date from {filename}...")
    with open(get_data_file_path(filename), "r") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    if not rows:
        return None

    last_row = rows[-1]
    if "date" not in last_row:
        raise ValueError("The 'date' column is missing in the CSV file.")

    if not last_row["date"]:
        return None

    last_date = datetime.datetime.strptime(last_row["date"], "%d %B %Y").date()
    return last_date


def create_csv(filename, headers):
    print(f"Creating {filename}...")
    if not os.path.isfile(get_data_file_path(filename)):
        with open(get_data_file_path(filename), "w") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        return True
    else:
        print(f"{filename} already exists")
        return False
    
def scrape_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
    }
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    return soup

def load_api_key(env_var_name):
    api_key = os.getenv(env_var_name)
    print(f"API Key: {api_key}")
    print(f"Checking {env_var_name}...")
    if not api_key:
        print(f"Please set the {env_var_name} environment variable")
        exit(1)
    return api_key

def pandas_load_csv(file_path):
    return pd.read_csv(get_data_file_path(file_path))
