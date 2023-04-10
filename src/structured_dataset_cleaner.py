import os
from fuzzywuzzy import process
import sys
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_csv_file, create_csv, write_csv_row

# Load player names and corresponding initials from CSV files
def load_players_from_csv(season_2021_players, season_2022_players):
    players = []
    for player in season_2021_players + season_2022_players:
        players.append((player['full_name'], player['name']))
    return players

# Check if the player name exists in the given data
def is_actual_player(player_name, players, transfermarkt_data):
    print(f"Checking if {player_name} is an actual player...")
    player_initials = convert_to_initials(player_name)
    all_names = [name for full_name, name in players] + [full_name for full_name, name in players] + [row['player_name'] for row in transfermarkt_data]

    match, score = process.extractOne(player_name, all_names)
    if score >= 90:
        print(f"Found match: {match} (score: {score})")
        return True

    match_initials, score_initials = process.extractOne(player_initials, all_names)
    if score_initials >= 90:
        print(f"Found match with initials: {match_initials} (score: {score_initials})")
        return True

    return False

# Convert a full name into initials format
def convert_to_initials(full_name):
    name_parts = full_name.split()
    first_name_initial = name_parts[0][0]
    last_name = name_parts[-1]
    return f"{first_name_initial}. {last_name}"

def find_player_info(full_name, season_2021_players, season_2022_players):
    for player in season_2021_players + season_2022_players:
        if player['full_name'] == full_name:
            return player
    return None


def find_transfermarkt_info(player_name, transfermarkt_data):
    match, score = process.extractOne(player_name, [row['player_name'] for row in transfermarkt_data])
    if score >= 90:
        return next(row for row in transfermarkt_data if row['player_name'] == match)
    return None

# Get the player's age, nationality, position, and market value from the given data
def get_player_info(players, player_name, season_2021_players, season_2022_players, transfermarkt_data):
    player_initials = convert_to_initials(player_name)

    for player in players:
        full_name, name = player

        _, score = process.extractOne(player_name, [full_name])
        _, score_initials = process.extractOne(player_initials, [name])

        if score >= 90 or score_initials >= 90:
            player_info = find_player_info(full_name, season_2021_players, season_2022_players)
            if player_info:
                age, nationality, position = player_info['age'], player_info['nationality'], ""
            else:
                age, nationality, position = None, None, None

            transfermarkt_info = find_transfermarkt_info(player_name, transfermarkt_data)
            if transfermarkt_info:
                if age is None:
                    age = transfermarkt_info['player_age']
                if nationality is None:
                    nationality = transfermarkt_info['player_nationality']
                position = transfermarkt_info['player_position']
                market_value = transfermarkt_info['market_value']

                return age, nationality, position, market_value

            return age, nationality, position, None

    return None, None, None, None

# Calculate the number of days to the next transfer window from the given date
def days_to_next_transfer_window(date_str):
    date = datetime.datetime.strptime(date_str, "%d %B %Y")
    year = date.year
    days_to_next = None

    transfer_windows = [
        datetime.datetime(year, 1, 1),
        datetime.datetime(year, 1, 2),
        datetime.datetime(year, 6, 9),
        datetime.datetime(year, 8, 31),
    ]

    for i in range(len(transfer_windows) - 1):
        if transfer_windows[i] <= date <= transfer_windows[i + 1]:
            days_to_next = (transfer_windows[i + 1] - date).days
            break

    return days_to_next

# Get the source of a row using its ID
def get_source_from_id(id, input_rows):
    for row in input_rows:
        if row["id"] == id:
            return row["source"]
    return ""

# Clean the dataset by removing duplicates, non-existent players, and rows with less than two clubs mentioned
def clean_dataset(input_rows, transfer_news_data):
    seen_rows = set()
    season_2021_players = read_csv_file("../data/39_2021_players.csv")
    season_2022_players = read_csv_file("../data/39_2022_players.csv")

    transfermarkt_data = read_csv_file("transfermarkt_data.csv")
    players = load_players_from_csv(season_2021_players, season_2022_players)

    output_path_filename = "cleaned_structured_data.csv"
    csv_headers = [
        "id", "date", "raw_text", "player_name", "clubs_mentioned",
        "player_age", "player_nationality", "player_position", "time_to_transfer_window", "source", "market_value"
    ]
    create_csv(output_path_filename, csv_headers)

    for row in input_rows:
        print("========================================")
        player_name = row["player_name"]
        clubs_mentioned = row["clubs_mentioned"].split(", ")
        print(f"Cleaning row for {player_name}...")

        # Remove duplicates
        row_data = tuple(row.items())
        if row_data in seen_rows:
            print("Duplicate row found, skipping...")
            continue
        seen_rows.add(row_data)

        # Only keep rows with actual football players in the "player_name" column
        if not is_actual_player(player_name, players, transfermarkt_data):
            print("Player not found, skipping...")
            continue

        # Only keep rows with 2 or more teams in the "clubs_mentioned" column
        if len(clubs_mentioned) < 2:
            print("Less than 2 clubs mentioned, skipping...")
            continue

        player_age, player_nationality, player_position, market_value = get_player_info(players, player_name, season_2021_players, season_2022_players, transfermarkt_data)
        time_to_transfer_window = days_to_next_transfer_window(row["date"])
        source = get_source_from_id(row["id"], transfer_news_data)

        new_row = {
            "id": row["id"],
            "date": row["date"],
            "raw_text": row["raw_text"],
            "player_name": player_name,
            "clubs_mentioned": row["clubs_mentioned"],
            "player_age": player_age,
            "player_nationality": player_nationality,
            "player_position": player_position,
            "time_to_transfer_window": time_to_transfer_window,
            "source": source,
            "market_value": market_value
        }

        print(f"Row is valid. Appending {player_name} to cleaned_structured_data.csv...")
        write_csv_row(output_path_filename, csv_headers, new_row)

if __name__ == "__main__":
    structured_data_rows = read_csv_file("structured_data.csv")
    transfer_news_data = read_csv_file("transfer_news_data.csv")
    clean_dataset(structured_data_rows, transfer_news_data)
