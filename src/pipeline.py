from data_structuring import structure_data
from data_preprocessing import preprocess_data
from data_wrangling import wrangle_data
from model_training import train_and_evaluate_models
from visualization_and_analysis import visualize_and_analyze
from data_collecting import collect_data
import utils
import time

def main():
    while True:
        print("Choose an option:")
        print("1. Run all steps.")
        print("2. Run steps interactively.")
        print("3. Run only one step.")
        print("4. Exit.")
        choice = int(input("Enter the option number: "))

        if choice == 1:
            run_all_steps()
        elif choice == 2:
            run_steps_interactively()
        elif choice == 3:
            run_single_step()
        elif choice == 4:
            break
        else:
            print("Invalid input. Please try again.")

def run_all_steps():
    steps = [collect_data, structure_data_step, preprocess_data_step, wrangle_data_step, train_and_evaluate_models_step, visualize_and_analyze_step]

    for step in steps:
        step()

def run_steps_interactively():
    steps = [collect_data, structure_data_step, preprocess_data_step, wrangle_data_step, train_and_evaluate_models_step, visualize_and_analyze_step]

    for step in steps:
        answer = input(f"Do you want to run the step '{step.__name__}'? (y/n): ")
        if answer.lower() == 'y':
            step()

def run_single_step():
    step_mapping = {
        1: collect_data,
        2: structure_data_step,
        3: preprocess_data_step,
        4: wrangle_data_step,
        5: train_and_evaluate_models_step,
        6: visualize_and_analyze_step
    }

    print("Choose a step to run:")
    for idx, step in step_mapping.items():
        print(f"{idx}. {step.__name__}")

    choice = int(input("Enter the step number: "))
    step_mapping[choice]()

def structure_data_step():
    structure_data("transfer_news_data.csv")

def preprocess_data_step():
    structured_data_rows = utils.pandas_load_csv("structured_data.csv")
    transfer_news_data = utils.pandas_load_csv("transfer_news_data.csv")
    football_api_players = utils.pandas_load_csv("football_api_players.csv")
    transfermarkt_data = utils.pandas_load_csv("transfermarkt_data.csv")

    preprocess_data(structured_data_rows, transfer_news_data, football_api_players, transfermarkt_data)

def wrangle_data_step():
    preprocessed_data = utils.pandas_load_csv("preprocessed_data.csv")
    transfermarkt_data = utils.pandas_load_csv("transfermarkt_data.csv")

    wrangle_data(preprocessed_data, transfermarkt_data)

def train_and_evaluate_models_step():
    output_data = utils.pandas_load_csv("output_data.csv")
    train_and_evaluate_models(output_data)

def visualize_and_analyze_step():
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