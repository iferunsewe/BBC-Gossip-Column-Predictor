import csv
import openai
import os
import json


def load_api_key():
  api_key = os.getenv("OPENAI_API_KEY")
  openai.api_key = api_key


def read_csv_file(file_path):
  with open(file_path, mode="r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    return list(reader)


def write_csv_file(file_path, fieldnames, data_rows):
  with open(file_path, mode="w", encoding="utf-8", newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in data_rows:
      writer.writerow(row)

def structure_data(raw_text):
  prompt = f"Structure the following raw_text into json that includes fields 'player_name' and 'clubs_mentioned' and whichever ones you suggest: {raw_text}. The result should be an array of objects where each object contains the player_name and clubs_mentioned fields. Only include football clubs in the clubs_mentioned field and not international teams. Only include actual football players in the player_name field and not football managers."
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=256,
    n=1,
    stop=None,
    temperature=0.5,
  )

  print(f"Response: {response}")

  # Get the response text
  response_text = response.choices[0].text.replace("'", '"')
  print(f"Response text: {response_text}")

  # Convert the response text into a dictionary using json.loads
  structured_data = json.loads(response_text)

  print(f"Structured data: {structured_data}")
  return structured_data

def process_rows(rows):
    processed_rows = []
    errors = []
    
    for row in rows:
        try:
            raw_text = row["raw_text"]
            structured_data = structure_data(raw_text)
            print(f"Structured fields: {structured_data}")

            for data in structured_data:
                # If there is no player name, skip this row
                if not data["player_name"]:
                    continue
                print(f"Data: {data}")
                new_row = {
                    "date": row["date"],
                    "player_name": data["player_name"],
                    "clubs_mentioned": data["clubs_mentioned"],
                }
                print(f"New row: {new_row}")
                processed_rows.append(new_row)
        except Exception as e:
            error_info = {"error": str(e), "raw_text": raw_text}
            errors.append(error_info)
            print(f"Error: {error_info}")

    return processed_rows, errors


if __name__ == "__main__":
    load_api_key()

    # Read the 'transfer_news_data.csv' file
    print("Reading 'transfer_news_data.csv'...")
    input_rows = read_csv_file("transfer_news_data.csv")

    # Process each row in 'transfer_news_data.csv'
    print("Processing rows...")
    output_rows, errors = process_rows(input_rows)

    # Print errors
    print(f"Errors ({len(errors)}): {errors}")

    # Create the 'primary_dataset.csv' file
    print("Creating 'primary_dataset.csv'...")
    output_fieldnames = ["date", "player_name", "clubs_mentioned"]
    write_csv_file("primary_dataset.csv", output_fieldnames, output_rows)


  # # Add any new suggested columns by GPT-3
  # for field in structured_data[2:]:
  #     print(f"Field: {field}")
  #     field_name, field_value = field.split(": ")
  #     if field_name not in fieldnames:
  #         fieldnames.append(field_name)
  #         writer = csv.DictWriter(outfile, fieldnames=fieldnames)
  #         writer.writeheader()
  #     new_row[field_name] = field_value

  # writer.writerow(new_row)
