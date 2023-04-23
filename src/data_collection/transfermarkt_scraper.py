import csv
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import scrape_content, get_data_file_path

# Process player data for a given row and h2_club
def process_player_data(row, h2_club, table):

    # Extract player details from the row
    player_name = row.find_all("td")[0].find(class_="hide-for-small").text.strip()
    player_age = row.find(class_="alter-transfer-cell").text.strip()
    player_position = row.find(class_="pos-transfer-cell").text.strip()

    # Extract club information for the player
    club_left = process_club_data(row, h2_club, "Left", table)
    club_joined = process_club_data(row, h2_club, "Joined", table)

    # Extract market value for the player
    market_value = row.find(class_="rechts").text.strip()

    return {
        "player_name": player_name,
        "player_age": player_age,
        "player_position": player_position,
        "club_left": club_left,
        "club_joined": club_joined,
        "market_value": market_value
    }

# Process club data for a given row
def process_club_data(row, h2_club, cell_text, table):
    club_cell = table.find("th", class_="verein-transfer-cell", text=cell_text)
    club_link = row.find(class_="verein-flagge-transfer-cell").find("a")

    if club_cell and club_link:
        club = club_link.text.strip()
        if club == "Without Club":
            return None
    elif club_cell and not club_link:
        return None
    else:
        club = h2_club

    return club

def process_transfer_data(soup, season):
    # Find all transfer tables in the page
    tables = soup.select('.box > .responsive-table > table')
    transfer_data = []

    # Iterate through each table and extract transfer data
    for table in tables:
        table_rows = table.find("tbody").find_all("tr")
        h2_sibling = table.find_parent("div", class_="responsive-table").find_previous_sibling("h2")
        clubs = h2_sibling.find_all("a")
        h2_club = clubs[1].text.strip()

        # Iterate through each row and process player data
        for row in table_rows:
            transfer = process_player_data(row, h2_club, table)
            transfer["season"] = season
            if transfer["club_left"] and transfer["club_joined"]:
                transfer_data.append(transfer)

    return transfer_data

def save_transfer_data_to_csv(transfer_data, filename):
    with open(get_data_file_path(filename), "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "player_name", "player_age", "player_position", "club_left", "club_joined", "market_value", "season"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, transfer in enumerate(transfer_data, start=1):
            transfer["id"] = idx
            writer.writerow(transfer)
            print(f"Saved transfer {idx}: {transfer['player_name']} from {transfer['club_left']} to {transfer['club_joined']} in {transfer['season']} season")

def run():
    seasons = {
        "2021": "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2021",
        "2022": "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2022",
    }

    all_transfer_data = []

    # Iterate through each season URL and fetch the transfer data
    for season, url in seasons.items():
        print(f"Fetching data for {season} season...")
        soup = scrape_content(url)
        transfer_data = process_transfer_data(soup, season)
        all_transfer_data.extend(transfer_data)

    # Save all transfer data to a CSV file
    save_transfer_data_to_csv(all_transfer_data, "transfermarkt_data.csv")
    print("Transfer data saved to transfermarkt_data.csv.")

def main():
    run()

if __name__ == "__main__":
    main()
