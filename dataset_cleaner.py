import csv

def read_csv_file(file_path):
    with open(file_path, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return [row for row in reader]

def write_csv_file(file_path, fieldnames, rows):
    with open(file_path, mode="w", encoding="utf-8", newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def is_actual_player(player_name):
    # Implement this function to check if the player_name is an actual football player
    pass

def clean_dataset(input_rows):
    cleaned_rows = []
    seen_rows = set()

    for row in input_rows:
        player_name = row["player_name"]
        clubs_mentioned = row["clubs_mentioned"].split(", ")

        # Remove duplicates
        row_data = tuple(row.items())
        if row_data in seen_rows:
            continue
        seen_rows.add(row_data)

        # Only keep rows with actual football players in the "player_name" column
        # if not is_actual_player(player_name):
        #     continue

        # Only keep rows with 2 or more teams in the "clubs_mentioned" column
        if len(clubs_mentioned) < 2:
            continue

        cleaned_rows.append(row)

    return cleaned_rows

if __name__ == "__main__":
    input_rows = read_csv_file("primary_dataset_with_news_id_2.csv")
    cleaned_rows = clean_dataset(input_rows)

    fieldnames = input_rows[0].keys()
    write_csv_file("cleaned_primary_dataset.csv", fieldnames, cleaned_rows)