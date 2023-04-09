import os
import requests
import csv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_file_path, create_csv

def get_player_data(response_data):
    id = response_data["player"]["id"]
    name = response_data["player"]["name"]
    full_name = f"{response_data['player']['firstname']} {response_data['player']['lastname']}"
    age = response_data["player"]["age"]
    nationality = response_data["player"]["nationality"]
    team_name = response_data["statistics"][0]["team"]["name"]
    return id, name, full_name, age, nationality, team_name

def fetch_players(url, headers):
    response = requests.get(url, headers=headers)
    return response

def write_player_to_csv(file_name, season, player_data):
    player_id = player_data[0]
    if not player_exists_in_csv(file_name, player_id):
        with open(get_data_file_path(f"{file_name}"), mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([*player_data, season])
    else:
        print(f"Player {player_id} already exists in {file_name}")


def get_players_by_league_season(league_id, season, file_name):
    print("Running get_players_by_league_season()")
    url = f"https://v3.football.api-sports.io/players?league={league_id}&season={season}"
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": os.environ.get("FOOTBALL_API_KEY")
    }

    response = fetch_players(url, headers)
    players = []

    while response.status_code == 200 and response.json()["errors"] == []:
        data = response.json()
        print(f"Found {len(data['response'])} players on page {data['paging']['current']}")
        for player in data["response"]:
            player_data = get_player_data(player)
            write_player_to_csv(file_name, season, player_data)
            players.append(player)

        if data["paging"]["current"] == data["paging"]["total"]:
            break

        next_page = data["paging"]["current"] + 1
        url = f"{url}&page={next_page}"
        print(f"Getting next page of results: {url}")
        response = fetch_players(url, headers)

    return players

def player_exists_in_csv(file_name, player_id):
    with open(get_data_file_path(f"{file_name}"), mode="r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0] == str(player_id):
                return True
    return False

def fetch_players_for_season(league_id, season):
    file_name = f"{league_id}_{season}_players.csv"
    csv_headers = ["id", "name", "full_name", "age", "nationality", "team_name", "season"]
    create_csv(file_name, csv_headers)
    players = get_players_by_league_season(league_id, season, file_name)
    print(f"Found {len(players)} players in {season} season.")
    return players

if __name__ == "__main__":
    print("Running football_api_scraper.py")
    api_key = os.environ.get("FOOTBALL_API_KEY")
    if not api_key:
        print("Please set the API_KEY environment variable")
        exit(1)

    league_id = 39
    seasons = [2021, 2022]

    for season in seasons:
        fetch_players_for_season(league_id, season)

