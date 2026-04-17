import json

INPUT_FILE = "data/clean/firstaid_docs.json"
OUTPUT_FILE = "data/analysis/categories.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = sorted(set(item["category"] for item in data))

print(f"Found {len(categories)} unique categories")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(categories, f, indent=2)

print("Saved categories to", OUTPUT_FILE)
