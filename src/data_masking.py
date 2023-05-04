import utils
import random
import string
import secrets

def remove_player_names_from_raw_text(data, player_names, raw_text_column='raw_text'):
    print(f"Removing player names from {raw_text_column} column")
    
    for name in player_names:
        print(f"Removing {name}")
        data[raw_text_column] = data[raw_text_column].str.replace(name, str(random_string(5)), case=False)

def extract_unique_player_names(*datasets, player_name_column='player_name'):
    print(f"Extracting unique player names from {player_name_column} column")
    unique_names = set()
    
    for data in datasets:
        names = data[player_name_column].unique()
        unique_names.update(names)
    
    print(f"Found {len(unique_names)} unique player names")
    return list(unique_names)

def random_number(length):
    return ''.join(random.choice(string.digits) for _ in range(length))

def random_string(length):
    return ''.join(secrets.choice(string.ascii_letters) for _ in range(length))

def mask_player_names(data, column_name='player_name'):
    print(f"Masking player names in {column_name} column")
    data[column_name] = data[column_name].apply(lambda x: str(random_string(5)))

def mask_nationality(data):
    print("Masking nationality column")
    if 'nationality' in data.columns:
        data['nationality'] = data['nationality'].apply(lambda x: str(random_string(5)))
    else:
        print("Nationality column not found. Skipping masking.")

def mask_age(data, column_name='age'):
    print(f"Masking age column")
    if column_name in data.columns:
        data[column_name] = data[column_name].apply(lambda x: str(random_number(2)))
    else:
        print(f"{column_name} column not found. Skipping masking.")

def mask(data, player_name_column='player_name', player_age_column='age', raw_text_column=None):
    if raw_text_column:
        unique_player_names = data[player_name_column].unique()
        remove_player_names_from_raw_text(data, unique_player_names, raw_text_column)

    mask_player_names(data, player_name_column)
    mask_nationality(data)
    mask_age(data, player_age_column)


def main():
    transfermarkt_data = utils.pandas_load_csv("transfermarkt_data.csv")
    structured_data = utils.pandas_load_csv("structured_data.csv")
    transfer_news_data = utils.pandas_load_csv("transfer_news_data.csv")
    football_api_players = utils.pandas_load_csv("football_api_players.csv")
    preprocessed_data = utils.pandas_load_csv("preprocessed_data.csv")

    unique_player_names = extract_unique_player_names(transfermarkt_data, structured_data, player_name_column='player_name')
    remove_player_names_from_raw_text(transfer_news_data, unique_player_names, raw_text_column='raw_text')

    mask(transfermarkt_data, player_name_column='player_name', player_age_column='player_age')
    mask(structured_data, raw_text_column='raw_text')
    mask(football_api_players, player_name_column='name')
    mask_player_names(football_api_players, column_name='full_name')
    mask(preprocessed_data, raw_text_column='raw_text')

    transfermarkt_data.to_csv(utils.get_data_file_path("transfermarkt_data.csv"), index=False)
    structured_data.to_csv(utils.get_data_file_path("structured_data.csv"), index=False)
    transfer_news_data.to_csv(utils.get_data_file_path("transfer_news_data.csv"), index=False)
    football_api_players.to_csv(utils.get_data_file_path("football_api_players.csv"), index=False)
    preprocessed_data.to_csv(utils.get_data_file_path("preprocessed_data.csv"), index=False)

if __name__ == '__main__':
    main()
