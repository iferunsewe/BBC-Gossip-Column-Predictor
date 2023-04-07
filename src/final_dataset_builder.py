import pandas as pd
from fuzzywuzzy import fuzz, process

def load_cleaned_dataset(file_path):
    return pd.read_csv(file_path)

def load_transfermarkt_data(file_path):
    return pd.read_csv(file_path)

def extract_player_name(row):
    return row["player_name"].split()[-1]

def extract_clubs_mentioned(row):
    return string_to_list(row["clubs_mentioned"])

def string_to_list(my_string):
    try:
        # Try to evaluate the string as a Python expression
        my_list = eval(my_string)
    except:
        # If that fails, assume it's a comma-separated string
        my_list = my_string.split(", ")
    return my_list

def check_player_found(player_name, transfermarkt_data):
    best_match = process.extractOne(player_name, transfermarkt_data["player_name"], scorer=fuzz.token_sort_ratio)
    return best_match[1] >= 50

def get_player_row(player_name, transfermarkt_data):
    best_match = process.extractOne(player_name, transfermarkt_data["player_name"], scorer=fuzz.token_sort_ratio)
    if best_match[1] >= 50:
        return transfermarkt_data[transfermarkt_data["player_name"] == best_match[0]]
    else:
        return None

def merge_clubs_left_joined(player_row):
    return pd.concat([player_row["club_left"], player_row["club_joined"]])

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

def set_veracity(clubs_exist):
    if clubs_exist:
        return True
    else:
        return False

def add_veracity_to_list(veracity_list, veracity):
    veracity_list.append(veracity)

def add_veracity_column(cleaned_dataset, veracity_list):
    cleaned_dataset["veracity"] = veracity_list
    return cleaned_dataset

def create_final_csv(cleaned_dataset):
    final_df = cleaned_dataset[["news_id", "player_name", "clubs_mentioned", "veracity"]]
    final_df.to_csv("../data/final.csv", index=False)

def main():
    # Load the cleaned structured dataset
    cleaned_dataset = load_cleaned_dataset("../data/cleaned_structured_data.csv")

    # Load the transfermarkt data
    transfermarkt_data = load_transfermarkt_data("../data/transfermarkt_data.csv")

    # Create a list to store the veracity values
    veracity_list = []

    # Loop through each row in the cleaned structured dataset
    for i in range(len(cleaned_dataset)):
        row = cleaned_dataset.iloc[i]

        # Get the player name and clubs mentioned from the row
        player_name = extract_player_name(row)
        clubs_mentioned = extract_clubs_mentioned(row)

        # Check if the player name exists in the transfermarkt data
        player_found = check_player_found(player_name, transfermarkt_data)

        if player_found:
            print("Player found: " + player_name)
            # Get the row for the player in the transfermarkt data
            player_row = get_player_row(player_name, transfermarkt_data)

            if player_row is not None:
                print("Player row found: " + player_name)
                # Merge the club_left and club_joined columns into a single array
                clubs_left_joined = merge_clubs_left_joined(player_row)

                # Check if all of the mentioned clubs exist in the clubs_left_joined array
                clubs_exist = check_clubs_exist(clubs_mentioned, clubs_left_joined)

                print("Clubs exist: " + str(clubs_exist))
                # Set the veracity value based on whether all clubs exist or not
                veracity = set_veracity(clubs_exist)
            else:
                veracity = ""
        else:
            print("Player not found: " + player_name)
            veracity = ""

        # Add the veracity value to the list
        add_veracity_to_list(veracity_list, veracity)

    # Add the veracity list as a new column in the cleaned dataset
    cleaned_dataset = add_veracity_column(cleaned_dataset, veracity_list)

    # Create a new final CSV with news_id, clubs_mentioned, and veracity columns
    create_final_csv(cleaned_dataset)

if __name__ == "__main__":
    main()

