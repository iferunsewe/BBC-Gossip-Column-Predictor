import pandas as pd
from thefuzz import fuzz, process
import utils

def extract_player_name(row):
    return row["player_name"]

def extract_clubs_mentioned(row):
    return string_to_list(row["clubs_mentioned"])

# Converts a string to a list
def string_to_list(my_string):
    try:
        my_list = eval(my_string)
    except:
        my_list = my_string.split(", ")
    return my_list

# Finds the best match between player names and a list of names using fuzzy string matching
def get_best_match(player_name, names_list):
    return process.extractOne(player_name, names_list, scorer=fuzz.token_sort_ratio)

def is_player_found(best_match, threshold=50):
    return best_match[1] >= threshold

# Gets the player row from transfermarkt_data based on the best match.
def get_player_row(player_name, transfermarkt_data):
    best_match = get_best_match(player_name, transfermarkt_data["player_name"])
    if is_player_found(best_match):
        return transfermarkt_data[transfermarkt_data["player_name"] == best_match[0]]
    else:
        return None

def merge_clubs_left_joined(player_row):
    return pd.concat([player_row["club_left"], player_row["club_joined"]])

# Checks if all clubs mentioned in the transfer news exist in the clubs left and joined by the player
def check_clubs_exist(clubs_mentioned, clubs_left_joined, threshold=75):
    clubs_mentioned_set = set(clubs_mentioned)
    clubs_found = set()

    for mentioned_club in clubs_mentioned_set:
        for club in clubs_left_joined.values:
            similarity = fuzz.ratio(mentioned_club.lower(), club.lower())
            if similarity >= threshold:
                clubs_found.add(mentioned_club)
                break
    
    return len(clubs_found) == len(clubs_mentioned_set)

# Determines the veracity of a transfer rumor based on the player found and the clubs mentioned
def get_veracity(player_found, player_row, clubs_mentioned):
    if player_found:
        clubs_left_joined = merge_clubs_left_joined(player_row)
        clubs_exist = check_clubs_exist(clubs_mentioned, clubs_left_joined)
        return clubs_exist
    else:
        return None

def create_output_csv(preprocessed_data):
    preprocessed_data.to_csv(utils.get_data_file_path("output_data.csv"), index=False)

# Removes unnecessary columns and rows with missing values
def clean_data(data):
    columns_to_exclude = ["raw_text", "player_name", "date", "id", "clubs_mentioned"]
    data = data.drop(columns=columns_to_exclude)
    data = data.dropna(subset=['veracity'])
    data['veracity'] = data['veracity'].astype(int)
    return data

# Wrangles the data by adding a veracity column to the preprocessed data
def wrangle_data(preprocessed_data, verified_data):
    veracity_list = []

    for i in range(len(preprocessed_data)):
        row = preprocessed_data.iloc[i]
        player_name = extract_player_name(row)
        print(f"Processing player {player_name}...")
        clubs_mentioned = extract_clubs_mentioned(row)

        player_row = get_player_row(player_name, verified_data)
        player_found = player_row is not None

        veracity = get_veracity(player_found, player_row, clubs_mentioned)
        veracity_list.append(veracity)

    preprocessed_data["veracity"] = veracity_list
    preprocessed_data = clean_data(preprocessed_data)
    create_output_csv(preprocessed_data)

def main():
    preprocessed_data = utils.pandas_load_csv(("preprocessed_data.csv"))
    verified_data = utils.pandas_load_csv(("transfermarkt_data.csv"))

    wrangle_data(preprocessed_data, verified_data)

if __name__ == "__main__":
    main()
