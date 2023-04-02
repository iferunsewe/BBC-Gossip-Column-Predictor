import pandas as pd

# Load the cleaned primary dataset
cleaned_dataset = pd.read_csv("cleaned_primary_dataset.csv")

# Load the transfermarkt data
transfermarkt_data = pd.read_csv("transfermarkt_data.csv")

# Create a list to store the veracity values
veracity_list = []

# Loop through each row in the cleaned primary dataset
for i in range(len(cleaned_dataset)):
    row = cleaned_dataset.iloc[i]

    # Get the player name and clubs mentioned from the row
    player_name = row["player_name"].split()[-1]  # Use the last name in case only 2nd name is provided
    clubs_mentioned = row["clubs_mentioned"].split(", ")

    # Check if the player name exists in the transfermarkt data
    player_found = transfermarkt_data["player_name"].str.contains(player_name, case=False).any()

    if player_found:
        # Get the row for the player in the transfermarkt data
        player_row = transfermarkt_data[transfermarkt_data["player_name"].str.contains(player_name, case=False)]

        # Merge the club_left and club_joined columns into a single array
        clubs_left_joined = pd.concat([player_row["club_left"], player_row["club_joined"]])

        # Check if all of the mentioned clubs exist in the clubs_left_joined array
        clubs_exist = all(club in clubs_left_joined for club in clubs_mentioned)

        # Set the veracity value based on whether all clubs exist or not
        if clubs_exist:
            veracity = True
        else:
            veracity = False
    else:
        veracity = ""

    # Add the veracity value to the list
    veracity_list.append(veracity)

# Add the veracity list as a new column in the cleaned dataset
cleaned_dataset["veracity"] = veracity_list

# Create a new final CSV with news_id, clubs_mentioned, and veracity columns
final_df = cleaned_dataset[["news_id", "player_name", "clubs_mentioned", "veracity"]]
final_df.to_csv("final.csv", index=False)
