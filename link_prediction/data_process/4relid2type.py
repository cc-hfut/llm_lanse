import csv
import json


def generate_relid_type_mapping(csv_file_path, json_output_path):
    'Read relid2type.csv and write a JSON mapping from relation IDs to type IDs.'
    relid2type = {}

    # Read CSV input.
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        # Skip the header.
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 4:  # Check that the row contains all required fields.
                rel_name = row[0]  # Relation name
                rel_id = int(row[1])  # Relation ID
                rel_type = row[2]  # Relation type
                type_id = int(row[3])  # Type ID
                relid2type[rel_id] = type_id

    # Save the JSON output.
    with open(json_output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(relid2type, jsonfile, ensure_ascii=False, indent=2)

    return relid2type


if __name__ == "__main__":
    csv_path = "relid2type.csv"
    json_path = "relid2typeid.json"

    # Generate the mapping file.
    mapping = generate_relid_type_mapping(csv_path, json_path)
