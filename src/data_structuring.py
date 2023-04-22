import csv
import openai
import os
import json
import utils

def read_csv_file(filename):
  with open(utils.get_data_file_path(filename), mode="r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    return list(reader)

def set_api_key():
    api_key = utils.load_api_key("OPENAI_API_KEY")
    openai.api_key = api_key

def determine_file_mode(filename):
    return "a" if os.path.exists(filename) else "w"

def write_csv_row(filename, fieldnames, row):
    mode = determine_file_mode(filename)

    with open(utils.get_data_file_path(filename), mode=mode, encoding="utf-8", newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)

def extract_football_info(raw_text):
    prompt = f"Structure the following raw_text into json that includes fields 'player_name' and 'clubs_mentioned': {raw_text}. The result should be an array of objects where each object contains the player_name and clubs_mentioned fields. Only include football clubs in the clubs_mentioned field and not international teams. Only include actual football players in the player_name field and not football managers."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Get the response text
    response_text = response.choices[0].text.replace("'", '"')
    print(f"Response text: {response_text}")

    # Convert the response text into a dictionary using json.loads
    structured_data = json.loads(response_text)

    print(f"Structured data: {structured_data}")
    return structured_data

def process_rows(input_rows, fieldnames):
    errors = []

    for row in input_rows:
        raw_text = row["raw_text"]
        try:
            structured_data = extract_football_info(raw_text)

            for data in structured_data:
                if not data["player_name"]:
                    continue

                new_row = {
                    "id": row["id"],
                    "date": row["date"],
                    "raw_text": raw_text,
                    "player_name": data["player_name"],
                    "clubs_mentioned": data["clubs_mentioned"]
                }

                write_csv_row("structured_data.csv", fieldnames, new_row)

        except Exception as e:
            error_info = {"error": str(e), "raw_text": raw_text}
            errors.append(error_info)
            print(f"Error: {error_info}")

    return errors

def structure_data(input_filename):
    csv_headers = ["id", "date", "raw_text", "player_name", "clubs_mentioned"]
    utils.create_csv("structured_data.csv", csv_headers)
    set_api_key()

    # Read the input file
    print(f"Reading '{input_filename}'...")
    input_rows = read_csv_file(input_filename)

    # Process each row in the input file
    print("Processing rows...")
    errors = process_rows(input_rows, csv_headers)

    # Print errors
    print(f"Errors ({len(errors)}): {errors}")

def main():
    structure_data("transfer_news_data.csv")

if __name__ == "__main__":
    main()