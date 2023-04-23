## README

### Introduction

This project aims to predict misinformation in football transfer news using various machine learning algorithms, particularly ensemble learning models. The main techniques used in this project include:

- Collecting data from the BBC gossip column and verifying it using transfermarkt data and API-Football data
- Structuring the data using GPT-3
- Extracting additional features from the data using NLP & fuzzy matching
- Encoding the data using one-hot encoding and multi-label binarization techniques
- Correcting the imbalanced dataset using random over sampling
- Filling in missing data by either dropping rows or filling in some columns with the mean
- Visualizing the data using bar charts, tables, and box plots
- Training and evaluating ensemble learning models such as Random Forest, XGBoost, and AdaBoost to find the best features

The target audience for this project includes researchers, developers, football clubs, and anyone with a general interest in football or methods to detect misinformation. The project results show that the Random Forest model achieved the highest accuracy of 0.8639 after applying random over sampling. The top 5 features identified by the Random Forest model are the market value of a player, the time to the start/end of the transfer window, and the age of a player. The detailed results can be found in the results directory.

### Project File Structure

The main folders and files for this project are organized as follows:

- **data/**: Contains all the project data in CSV format.
- **results/**: Stores images of model evaluations and dataset analyses.
- **src/**: Contains Python scripts for the project:
    - **data_collection/**: Scripts for collecting data.
        - bbc_transfer_rumours_scraper.py: Scrapes BBC gossip column links.
        - football_api.py: Collects player data from API-Football.
        - transfer_news_scraper.py: Extracts transfer news from gossip column links.
        - transfermarkt_scraper.py: Scrapes transfermarkt data for verifying transfer news.
    - data_collecting.py: Runs all data collection scripts.
    - data_preprocessing.py: Preprocesses the dataset (updates missing data, removes irrelevant data, encodes data).
    - data_structuring.py: Structures raw data using GPT-3.
    - data_wrangling.py: Identifies true/false transfer news by cross-referencing datasets.
    - model_training.py: Evaluates and trains ensemble learning models, identifies important features.
    - pipeline.py: Sets up the data pipeline, requiring possible user input.
    - utils.py: Shared functions used across the project.
    - visualization_and_analysis.py: Visualizes the dataset and analyzes results.

### Installation

To set up the project environment, please follow these steps:

1. Ensure you have Python 3.6+ installed. You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).

2. Install the required libraries and dependencies. You can find a list of the main libraries used in this project below:
    - numpy
    - pandas
    - scikit-learn
    - xgboost
    - matplotlib
    - seaborn
    - thefuzz
    - locationtagger
    - google-api-python-client
    - imbalanced-learn
    - beautifulsoup4
    - requests
    - anaconda3
    - juypter notebook

3. It is recommended to use Anaconda to manage the project dependencies:
    a. Create an Anaconda environment: `conda create -n <name of env> python=<python version>`
    b. Activate the environment: `conda activate <name of env>`

4. Run Jupyter Notebook: `jupyter notebook` which will open up a new browser window containing python scripts in the project

5. Install the required libraries within the Anaconda environment by running the following command: `pip install -r requirements.txt` (Make sure you have a `requirements.txt` file in your project directory with the necessary libraries and their versions listed.)

### Usage

To use the project, you have two main options:

1. Run individual scripts from the command line or Jupyter Notebook: You can execute each script in the src directory (except utils.py) either by running python <filepath> in the command line or by pressing the play button in the corresponding .ipynb file in the Jupyter Notebook UI.

2. Run the entire data pipeline by executing the src/pipeline.py file or by pressing the play button in the pipeline.ipynb file in the Jupyter Notebook UI.

    The data pipeline offers various options for running the different steps of the project. You can choose to run all steps at once, run steps interactively, or run only a single step. The available steps are:

    1. Collect data
    2. Structure data
    3. Preprocess data
    4. Wrangle data
    5. Train and evaluate models
    6. Visualize and analyze data

    The pipeline script allows you to select an option using a simple menu. For example, you can choose to run all steps at once, run steps interactively (you will be prompted whether you want to run each step or not), or run only a single step by entering the corresponding number. The pipeline will execute the selected steps and display the results.



