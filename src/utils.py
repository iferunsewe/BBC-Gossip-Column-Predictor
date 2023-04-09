import os
import datetime
import csv
from csv import reader

def get_data_file_path(filename):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    return os.path.join(data_dir, filename)

def read_last_date_from_csv(filename):
    print(f"Reading last date from {filename}...")
    with open(get_data_file_path(filename), "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        last_row = rows[-1]
        if last_row[0] == "Date":
            last_date = None
        else:
            last_date = datetime.datetime.strptime(last_row[0], "%d %B %Y").date()

    return last_date

def create_csv(filename, headers):
    print(f"Creating {filename}...")
    if not os.path.isfile(get_data_file_path(f"{filename}")):
        with open(get_data_file_path(f"{filename}"), "w") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        return True
    else:
        print(f"{filename} already exists")
        return False
