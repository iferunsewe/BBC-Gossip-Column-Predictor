import os
import re
import sys
import datetime
import pandas as pd
from fuzzywuzzy import process
import locationtagger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_file_path

def is_actual_player(player_name, football_api_players, transfermarkt_data):
    print(f"Checking if {player_name} is an actual player...")
    all_names = football_api_players['full_name'].tolist() + football_api_players['name'].tolist() + transfermarkt_data['player_name'].tolist()

    if get_best_fuzzy_match(player_name, all_names) is None:
        return False
    else:
        return True
    
def get_best_fuzzy_match(player_name, names_list):
    match, score = process.extractOne(player_name, names_list)
    if score >= 90:
        print(f"Found match: {match} (score: {score})")
        return match
    else:
        return None

def get_api_row(player_name, football_api_players):
    full_names = football_api_players['full_name'].tolist()
    short_names = football_api_players['name'].tolist()

    best_full_name_match = get_best_fuzzy_match(player_name, full_names)
    best_short_name_match = get_best_fuzzy_match(player_name, short_names)

    if best_full_name_match:
        api_row = football_api_players.loc[football_api_players['full_name'] == best_full_name_match]
    elif best_short_name_match:
        api_row = football_api_players.loc[football_api_players['name'] == best_short_name_match]
    else:
        api_row = pd.DataFrame()

    return api_row

def map_to_football_api_position(position):
    position_mapping = {
        "Goalkeeper": "Goalkeeper",
        "Centre-Back": "Defender",
        "Left-Back": "Defender",
        "Right-Back": "Defender",
        "Defensive Midfield": "Midfielder",
        "Central Midfield": "Midfielder",
        "Attacking Midfield": "Midfielder",
        "Left Midfield": "Midfielder",
        "Right Midfield": "Midfielder",
        "Centre-Forward": "Attacker",
        "Left Winger": "Attacker",
        "Right Winger": "Attacker",
        "Second Striker": "Attacker",
    }

    return position_mapping.get(position, "Unknown")

def get_player_data(player_name, transfermarkt_data, football_api_players):
    player_data = {}
    
    transfermarkt_row = transfermarkt_data.loc[transfermarkt_data['player_name'] == player_name]
    api_row = get_api_row(player_name, football_api_players)

    if transfermarkt_row.empty and api_row.empty:
        print(f"Could not find {player_name} in any dataset...")
        return None

    if not transfermarkt_row.empty:
        print(f"Found {player_name} in Transfermarkt dataset...")
        player_data['age'] = transfermarkt_row['player_age'].values[0]
        player_data['position'] = map_to_football_api_position(transfermarkt_row['player_position'].values[0])
        player_data['market_value'] = transfermarkt_row['market_value'].values[0]
        print(f"Player data: {player_data}")

    if not api_row.empty:
        print(f"Found {player_name} in Football API dataset...")
        player_data = update_player_data(player_data, api_row)
        print(f"Player data: {player_data}")


    return player_data


def update_player_data(player_data, api_row):
    if player_data.get('age') is None:
        player_data['age'] = api_row['age'].values[0]

    if player_data.get('position') is None:
        player_data['position'] = api_row['position'].values[0]

    if player_data.get('market_value') is None:
        player_data['market_value'] = None

    player_data['nationality'] = api_row['nationality'].values[0]

    return player_data

def find_age_in_text(text):
    print(f"Finding age in text: {text}")
    # Match patterns like "27-year-old" or "27 year old"
    pattern1 = r"(\d{1,2})[-\s]?year[-\s]?old"
    matches1 = re.findall(pattern1, text)
    
    if matches1:
        return int(matches1[0])

    # Match patterns like ", 27," or ", 27."
    pattern2 = r"[A-Za-z\s]+, (\d{1,2})[.,]"
    matches2 = re.findall(pattern2, text)
    
    if matches2:
        return int(matches2[0])

    return None

def find_nationality_in_text(raw_text):
    print(f"Finding nationality in text: {raw_text}")
    entities = locationtagger.find_locations(text=raw_text)
    countries = entities.countries

    if len(countries) == 1:
        return countries[0]
    else:
        return None


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


def clean_dataset(input_rows, transfer_news_data, football_api_players, transfermarkt_data):
    print("Cleaning dataset...")

    input_rows['is_actual_player'] = input_rows['player_name'].apply(lambda x: is_actual_player(x, football_api_players, transfermarkt_data))
    cleaned_data = input_rows[input_rows['is_actual_player'] & (input_rows['clubs_mentioned'].str.count(',') >= 1)].drop(columns=['is_actual_player'])

    player_data_dicts = [x for x in cleaned_data['player_name'].apply(lambda x: get_player_data(x, transfermarkt_data, football_api_players)).tolist() if x is not None]
    player_data_df = pd.DataFrame(player_data_dicts)
    cleaned_data.reset_index(drop=True, inplace=True)

    cleaned_data = pd.concat([cleaned_data, player_data_df], axis=1)

    cleaned_data['age'] = cleaned_data.apply(lambda row: row['age'] if not pd.isnull(row['age']) else find_age_in_text(row['raw_text']), axis=1)
    cleaned_data['nationality'] = cleaned_data.apply(lambda row: row['nationality'] if not pd.isnull(row['nationality']) else find_nationality_in_text(row['raw_text']), axis=1)
    cleaned_data['time_to_transfer_window'] = cleaned_data['date'].apply(days_to_next_transfer_window)
    cleaned_data = cleaned_data.merge(transfer_news_data[['id', 'source']], on='id', how='left')

    output_path_filename = get_data_file_path("cleaned_structured_data.csv")
    cleaned_data.to_csv(output_path_filename, index=False)
    print(f"Cleaned dataset saved to {output_path_filename}")


def main():
    print("Loading data...")
    structured_data_rows = pd.read_csv(get_data_file_path("structured_data.csv"))
    transfer_news_data = pd.read_csv(get_data_file_path("transfer_news_data.csv"))
    football_api_players = pd.read_csv(get_data_file_path("football_api_players.csv"))
    transfermarkt_data = pd.read_csv(get_data_file_path("transfermarkt_data.csv"))

    clean_dataset(structured_data_rows, transfer_news_data, football_api_players, transfermarkt_data)


if __name__ == "__main__":
    main()

