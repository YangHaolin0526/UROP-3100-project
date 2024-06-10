import os
import json

input_file = os.path.expanduser("./skg_raw_data/skginstruct.json")
output_file = "/home/nlpintern1/haolin/StructLM/transfered.json"

with open(input_file, "r") as f:
    json_data = json.load(f)

output_data = {
    "type": "text_only",
    "instances": []
}

# Convert data format
for item in json_data:
    instance = {
        "text": item["sys_prompt"] + "\n" + item["input"] + "\n" + item["label"]
    }
    output_data["instances"].append(instance)

# Write output file
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)
