import requests
from bs4 import BeautifulSoup
import csv

url = "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2021"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
}

# Fetch the webpage content
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Find tables matching the specified selector
tables = soup.select('.box > .responsive-table > table')

# Initialize transfer data and ID counter
transfer_data = []
id_counter = 385490

# Create the CSV file
with open("../../data/transfermarkt_data.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "player_name", "player_age", "player_position", "club_left", "club_joined", "transfer_fee"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    print(f"Found {len(tables)} tables")
    # Process each table
    for table in tables: 
        table_rows = table.find("tbody").find_all("tr")

        print(f"Found {len(table_rows)} rows in the table")
        # Find club_left and club_joined from h2 element of the parent element "responsive-table"
        h2_sibling = table.find_parent("div", class_="responsive-table").find_previous_sibling("h2")
        clubs = h2_sibling.find_all("a")
        h2_club = clubs[1].text.strip()
        print(f"Found h2 club: {h2_club}")

        # Process each row in the table
        for row in table_rows:
            player_name = row.find_all("td")[0].find(class_="hide-for-small").text.strip()
            player_age = row.find(class_="alter-transfer-cell").text.strip()
            player_position = row.find(class_="pos-transfer-cell").text.strip()

            club_left_cell = table.find("th", class_="verein-transfer-cell", text="Left")
            club_left_link = row.find(class_="verein-flagge-transfer-cell").find("a")

            if club_left_cell and club_left_link:
                club_left = club_left_link.text.strip()
                if club_left == "Without Club":
                    continue
            elif club_left_cell and not club_left_link:
                continue
            else:
                club_left = h2_club

            club_joined_cell = table.find("th", class_="verein-transfer-cell", text="Joined")
            club_joined_link = row.find(class_="verein-flagge-transfer-cell").find("a")
            if club_joined_cell and club_joined_link:
                club_joined = club_joined_link.text.strip()
                if club_joined == "Without Club":
                    continue
            elif club_joined_cell and not club_joined_link:
                continue
            else:
                club_joined = h2_club

            transfer_fee = row.find(class_="rechts").text.strip()

            transfer = {
                "id": id_counter,
                "player_name": player_name,
                "player_age": player_age,
                "player_position": player_position,
                "club_left": club_left,
                "club_joined": club_joined,
                "transfer_fee": transfer_fee
            }

            # Save the data to the CSV file
            writer.writerow(transfer)
            print(f"Processed transfer {id_counter}: {player_name} from {club_left} to {club_joined}")

            transfer_data.append(transfer)
            id_counter += 1
