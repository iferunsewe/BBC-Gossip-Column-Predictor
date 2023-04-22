from data_structuring import structure_data
from data_preprocessing import preprocess_data
from data_wrangling import wrangle_data
import utils
import pandas as pd

def main():
    structure_data("transfer_news_data.csv")
    
    structured_data_rows = utils.pandas_load_csv("structured_data.csv")
    transfer_news_data = utils.pandas_load_csv("transfer_news_data.csv")
    football_api_players = utils.pandas_load_csv("football_api_players.csv")
    transfermarkt_data = utils.pandas_load_csv("transfermarkt_data.csv")

    preprocess_data(structured_data_rows, transfer_news_data, football_api_players, transfermarkt_data)

    preprocessed_data = utils.pandas_load_csv("preprocessed_data.csv")

    wrangle_data(preprocessed_data, transfermarkt_data)

if __name__ == "__main__":
    main()