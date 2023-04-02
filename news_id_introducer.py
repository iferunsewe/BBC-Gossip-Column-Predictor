import pandas as pd
import numpy as np

# Load the transfer news data
transfer_news = pd.read_csv("transfer_news_data.csv")

# Add a new ID column
start_id = np.random.randint(100000, 999999)
transfer_news.insert(0, "id", range(start_id, start_id + len(transfer_news)))

# Save the updated transfer news data to a new file
transfer_news.to_csv("transfer_news_data_with_id.csv", index=False)

# Load the primary dataset
primary_dataset = pd.read_csv("primary_dataset.csv")

# Add a news_id column to the primary dataset
primary_dataset["news_id"] = np.nan

# Iterate over the rows in the primary dataset
for index, row in primary_dataset.iterrows():
    # Find the corresponding row in the transfer news data
    transfer_news_row = transfer_news.loc[transfer_news["raw_text"] == row["raw_text"]]
    if len(transfer_news_row) > 0:
        # Get the ID from the transfer news data and add it to the news_id column in the primary dataset
        primary_dataset.at[index, "news_id"] = transfer_news_row.iloc[0]["id"]

# Save the updated primary dataset to a new file
primary_dataset.to_csv("primary_dataset_with_news_id.csv", index=False)
