from data_structuring import structure_data
from data_preprocessing import preprocess_data
from data_wrangling import wrangle_data
from model_training import train_and_evaluate_models
from visualization_and_analysis import visualize_and_analyze
import utils
import time


def main():
    structure_data("transfer_news_data.csv")
    
    structured_data_rows = utils.pandas_load_csv("structured_data.csv")
    transfer_news_data = utils.pandas_load_csv("transfer_news_data.csv")
    football_api_players = utils.pandas_load_csv("football_api_players.csv")
    transfermarkt_data = utils.pandas_load_csv("transfermarkt_data.csv")

    preprocess_data(structured_data_rows, transfer_news_data, football_api_players, transfermarkt_data)

    preprocessed_data = utils.pandas_load_csv("preprocessed_data.csv")

    wrangle_data(preprocessed_data, transfermarkt_data)

    output_data = utils.pandas_load_csv("output_data.csv")
    
    train_and_evaluate_models(output_data)

    data = utils.pandas_load_csv("output_data.csv")

    continuous_features_to_analyze = ['age', 'time_to_transfer_window', 'market_value']
    categorical_features_to_analyze = ['nationality', 'position', 'source']
    visualize_and_analyze(data, continuous_features_to_analyze, categorical_features_to_analyze, 'veracity')

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time 
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")  