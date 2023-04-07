import pandas as pd
import numpy as np

# Load the transfer news data
transfer_news = pd.read_csv("transfer_news_data.csv")

# Add a new ID column
start_id = np.random.randint(100000, 999999)
transfer_news.insert(0, "id", range(start_id, start_id + len(transfer_news)))

# Save the updated transfer news data to a new file
transfer_news.to_csv("transfer_news_data_with_id.csv", index=False)

# Load the structured dataset
structured_dataset = pd.read_csv("structured_dataset.csv")

# Add a news_id column to the structured dataset
structured_dataset["news_id"] = np.nan

# Iterate over the rows in the structured dataset
for index, row in structured_dataset.iterrows():
    # Find the corresponding row in the transfer news data
    transfer_news_row = transfer_news.loc[transfer_news["raw_text"] == row["raw_text"]]
    if len(transfer_news_row) > 0:
        # Get the ID from the transfer news data and add it to the news_id column in the structured dataset
        structured_dataset.at[index, "news_id"] = transfer_news_row.iloc[0]["id"]

# Save the updated structured dataset to a new file
structured_dataset.to_csv("structured_dataset_with_news_id.csv", index=False)
