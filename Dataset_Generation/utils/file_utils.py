import json

def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
