import re
import datetime
import pandas as pd
from thefuzz import process
import locationtagger
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import utils

POSITION_MAPPING = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender",
    "Left-Back": "Defender",
    "Right-Back": "Defender",
    "Defender": "Defender",
    "Defensive Midfield": "Midfielder",
    "Central Midfield": "Midfielder",
    "Attacking Midfield": "Midfielder",
    "Left Midfield": "Midfielder",
    "Right Midfield": "Midfielder",
    "Midfielder": "Midfielder",
    "Playmaker": "Midfielder",
    "Centre-Forward": "Attacker",
    "Left Winger": "Attacker",
    "Right Winger": "Attacker",
    "Second Striker": "Attacker",
    "Winger": "Attacker",
    "Striker": "Attacker",
    "Forward": "Attacker",
}

CLUB_MAPPING = {
    # All the clubs above 50 and below my threshold of 75 in the fuzzy matching score so they were pointed out automatically
    'West Brom': 'West Bromwich Albion',
    'Newcastle': 'Newcastle United',
    'West Ham': 'West Ham United',
    'Tottenham': 'Tottenham Hotspur',
    'Spurs': 'Tottenham Hotspur',
    'Blackburn': 'Blackburn Rovers',
    'Nice': 'OGC Nice',
    'Roma': 'AS Roma',
    'Lille': 'LOSC Lille',
    'Leeds': 'Leeds United',
    'Brighton': 'Brighton & Hove Albion',
    'Norwich': 'Norwich City',
    'Schalke': 'FC Schalke 04',
    'Sevilla': 'Sevilla FC',
    'Inter': 'Inter Milan',
    'Naples': 'SSC Napoli',
    'Barca': 'Barcelona',
    'United': 'Manchester United',
    'Stoke': 'Stoke City',
    'Atletico': 'Atlético Madrid',
    'City': 'Manchester City',
    'Hertha': 'Hertha Berlin',
    'Metz': 'FC Metz',
    'The Hammers': 'West Ham United',
    'Genk': 'KRC Genk',
    'Mainz': '1.FSV Mainz 05',
    'Eupen': 'KAS Eupen',
    'New England Revolution': 'New England',
    # All the clubs that would not be found would be below 50 in the fuzzy matching score so I manually mapped them
    'Gunners': 'Arsenal FC',
    'Cherries': 'AFC Bournemouth',
    'Reds': 'Liverpool FC',
    'Blues': 'Chelsea FC',
    'Saints': 'Southampton FC',
    'Seagulls': 'Brighton & Hove Albion',
    'Canaries': 'Norwich City',
    'Magpies': 'Newcastle United',
    'Villans': 'Aston Villa',
    'Eagles': 'Crystal Palace',
    'Cottagers': 'Fulham FC',
    'Hornets': 'Watford FC',
}

# Checks if a player is an actual player by comparing their name against a list of all known player names.
def is_actual_player(player_name, known_player_names):
    print(f"Checking if {player_name} is an actual player...")
    
    if get_best_fuzzy_match(player_name, known_player_names) is None:
        return False
    else:
        return True

# Finds the best fuzzy match for a given player name in a list of names.    
def get_best_fuzzy_match(player_name, names_list):
    match, score = process.extractOne(player_name, names_list)
    if score >= 90:
        print(f"Found match: {match} (score: {score})")
        return match
    else:
        return None

# Retrieves the row from the Transfermarkt dataset corresponding to a given player.
def get_transfermarkt_row(player_name, transfermarkt_data):
    player_names = transfermarkt_data['player_name'].tolist()
    best_match = get_best_fuzzy_match(player_name, player_names)

    if best_match:
        transfermarkt_row = transfermarkt_data.loc[transfermarkt_data['player_name'] == best_match]
    else:
        transfermarkt_row = pd.DataFrame()

    return transfermarkt_row

# Retrieves the row from the Football API dataset corresponding to a given player.
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

# Maps a player's position to one of the positions recognized by the Football API.
def map_to_football_api_position(position):
    return POSITION_MAPPING.get(position, "Unknown")

def get_transfermarkt_data(player_name, transfermarkt_data):
    transfermarkt_row = get_transfermarkt_row(player_name, transfermarkt_data)
    
    if transfermarkt_row.empty:
        print(f"{player_name} not found in Transfermarkt dataset")
        return None

    player_data = {
        'age': transfermarkt_row['player_age'].values[0],
        'position': map_to_football_api_position(transfermarkt_row['player_position'].values[0]),
        'market_value': transfermarkt_row['market_value'].values[0]
    }
    print(f"Found {player_name} in Transfermarkt dataset: {player_data}")

    return player_data

def get_api_data(player_name, football_api_players):
    api_row = get_api_row(player_name, football_api_players)

    if api_row.empty:
        print(f"{player_name} not found in Football API dataset")
        return None

    player_data = {
        'age': api_row['age'].values[0],
        'position': api_row['position'].values[0],
        'nationality': api_row['nationality'].values[0]
    }
    print(f"Found {player_name} in Football API dataset: {player_data}")

    return player_data

# Merges the player data from Transfermarkt and Football API datasets.
def merge_player_data(transfermarkt_data, api_data):
    merged_data = {}

    for key in ['age', 'position', 'market_value', 'nationality']:
        merged_data[key] = transfermarkt_data.get(key) or api_data.get(key)

    return merged_data

def get_player_data(player_name, transfermarkt_data, football_api_players):
    transfermarkt_player_data = get_transfermarkt_data(player_name, transfermarkt_data)
    api_player_data = get_api_data(player_name, football_api_players)

    if transfermarkt_player_data is None and api_player_data is None:
        print(f"Could not find {player_name} in any dataset...")
        return None

    player_data = merge_player_data(transfermarkt_player_data or {}, api_player_data or {})
    print(f"Final player data: {player_data}")

    return player_data

# Finds the player's age in the given text using regular expressions.
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

# Finds the player's nationality in the given text using the locationtagger library.
def find_nationality_in_text(raw_text):
    print(f"Finding nationality in text: {raw_text}")
    entities = locationtagger.find_locations(text=raw_text)
    countries = entities.countries

    if len(countries) == 1:
        return countries[0]
    else:
        return None
    
# Finds the player's position in the given text using the POSITION_MAPPING dictionary.
def find_position_in_text(raw_text):
    print(f"Finding position in text: {raw_text}")
    positions = list(POSITION_MAPPING.keys())
    raw_text_lower = raw_text.lower()

    for position in positions:
        if position.lower() in raw_text_lower:
            return map_to_football_api_position(position)
    return None

# Calculates the number of days until the next transfer window or the end of current window based on a given date string.
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

# Clean market_value column by converting to numerical values
def clean_market_value(value):
    if pd.isna(value) or value == '-':
        return np.nan

    value = value.replace('€', '')
    if 'm' in value:
        value = float(value.replace('m', '')) * 1_000_000
    elif 'k' in value:
        value = float(value.replace('k', '')) * 1_000
    return value

# Convert clubs_mentioned column to a list of clubs and map to transfermarkt names
def convert_clubs_mentioned(data):
    data = data.copy()
    data.loc[:, 'clubs_mentioned'] = data['clubs_mentioned'].apply(
        lambda x: eval(x) if x.startswith('[') else x.split(', '))
    data.loc[:, 'clubs_mentioned'] = data['clubs_mentioned'].apply(
        lambda clubs: [CLUB_MAPPING.get(club, club) for club in clubs])
    return data

# One-hot encode categorical columns
def encode_columns(data, columns_to_encode):
    encoded_data = []
    for col in columns_to_encode:
        if col == 'clubs_mentioned':
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(data[col])
            encoded_df = pd.DataFrame(encoded, columns=mlb.classes_)
        else:
            ohe = OneHotEncoder(sparse_output=False)
            encoded = ohe.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]))
        
        encoded_data.append(encoded_df)
    return encoded_data

# Concatenate original data with encoded columns
def concat_encoded_data(data, encoded_data):
    data.reset_index(drop=True, inplace=True)
    data_encoded = pd.concat([data] + encoded_data, axis=1)
    return data_encoded

# Filters out the rows of the input dataset that do not correspond to actual players.
def filter_actual_players(input_rows, football_api_players, transfermarkt_data):
    known_player_names = set(football_api_players['full_name'].tolist() + football_api_players['name'].tolist() + transfermarkt_data['player_name'].tolist())
    input_rows['is_actual_player'] = input_rows['player_name'].apply(lambda x: is_actual_player(x, known_player_names))
    return input_rows[input_rows['is_actual_player'] & (input_rows['clubs_mentioned'].str.count(',') >= 1)].drop(columns=['is_actual_player'])

# Adds player data to the preprocessed dataset.
def add_player_data(preprocessed_data, transfermarkt_data, football_api_players):
    player_data_dicts = [x for x in preprocessed_data['player_name'].apply(lambda x: get_player_data(x, transfermarkt_data, football_api_players)).tolist() if x is not None]
    player_data_df = pd.DataFrame(player_data_dicts)
    preprocessed_data.reset_index(drop=True, inplace=True)
    return pd.concat([preprocessed_data, player_data_df], axis=1)

# Updates missing data in the preprocessed dataset.
def update_missing_data(preprocessed_data):
    preprocessed_data['age'] = preprocessed_data.apply(lambda row: row['age'] if not pd.isnull(row['age']) else find_age_in_text(row['raw_text']), axis=1)
    preprocessed_data['nationality'] = preprocessed_data.apply(lambda row: row['nationality'] if not pd.isnull(row['nationality']) else find_nationality_in_text(row['raw_text']), axis=1)
    preprocessed_data['position'] = preprocessed_data.apply(lambda row: row['position'] if not pd.isnull(row['position']) else find_position_in_text(row['raw_text']), axis=1)
    return preprocessed_data

def calculate_days_to_transfer_window(preprocessed_data):
    preprocessed_data['time_to_transfer_window'] = preprocessed_data['date'].apply(days_to_next_transfer_window)
    return preprocessed_data

def clean_market_values(preprocessed_data):
    preprocessed_data['market_value'] = preprocessed_data['market_value'].apply(clean_market_value)
    return preprocessed_data

def merge_with_transfer_news_data(preprocessed_data, transfer_news_data):
    return preprocessed_data.merge(transfer_news_data[['id', 'source']], on='id', how='left')

# Encodes and concatenates the specified columns.
def encode_and_concat_columns(preprocessed_data, columns_to_encode):
    encoded_data = encode_columns(preprocessed_data, columns_to_encode)
    return concat_encoded_data(preprocessed_data, encoded_data)

def preprocess_data(input_rows, transfer_news_data, football_api_players, transfermarkt_data):
    print("Preprocessing dataset...")

    preprocessed_data = filter_actual_players(input_rows, football_api_players, transfermarkt_data)
    preprocessed_data = add_player_data(preprocessed_data, transfermarkt_data, football_api_players)
    preprocessed_data = update_missing_data(preprocessed_data)
    preprocessed_data = calculate_days_to_transfer_window(preprocessed_data)
    preprocessed_data = clean_market_values(preprocessed_data)
    preprocessed_data = merge_with_transfer_news_data(preprocessed_data, transfer_news_data)
    preprocessed_data = convert_clubs_mentioned(preprocessed_data)
    columns_to_encode = ['clubs_mentioned', 'nationality', 'position', 'source']
    preprocessed_data = encode_and_concat_columns(preprocessed_data, columns_to_encode)

    output_path_filename = utils.get_data_file_path("preprocessed_data.csv")
    preprocessed_data.to_csv(output_path_filename, index=False)
    print(f"Preprocessed dataset saved to {output_path_filename}")

def main():
    print("Loading data...")
    structured_data_rows = pd.read_csv(utils.get_data_file_path("structured_data.csv"))
    transfer_news_data = pd.read_csv(utils.get_data_file_path("transfer_news_data.csv"))
    football_api_players = pd.read_csv(utils.get_data_file_path("football_api_players.csv"))
    transfermarkt_data = pd.read_csv(utils.get_data_file_path("transfermarkt_data.csv"))

    preprocess_data(structured_data_rows, transfer_news_data, football_api_players, transfermarkt_data)


if __name__ == "__main__":    
    main()
