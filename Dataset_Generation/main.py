import json
from pathlib import Path
from pipeline.annotate_resume import annotate_resume
from utils.file_utils import append_jsonl

def load_processed_files(jsonl_path):
    processed = set()
    if Path(jsonl_path).exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["file_name"])
                except Exception:
                    pass
    return processed



RESUME_DIR = Path("resumes")
OUTPUT_FILE = "data/processed/resume_dataset.jsonl"

def final_processing():
    processed_files = load_processed_files(OUTPUT_FILE)

    for pdf_path in RESUME_DIR.glob("*.pdf"):
        if pdf_path.name in processed_files:
            print(f"Skipping {pdf_path.name}")
            continue

        record = annotate_resume(str(pdf_path))
        record["file_name"] = pdf_path.name

        append_jsonl(OUTPUT_FILE, record)
        print(f"Processed {pdf_path.name}")


if __name__ == "__main__":
    final_processing()

