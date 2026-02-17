from llm.entity_llm import extract_entities
from utils.spans import entities_to_spans, spans_to_token_bio
from utils.clean import clean_text
from utils.parser import parse

from typing import List, Tuple, Dict

def bio_to_json(
    bio_data: List[Tuple[str, str]]
) -> Dict[str, List[str]]:
   
    tokens = []
    labels = []

    for token, label in bio_data:
        tokens.append(token)
        labels.append(label)

    return {
        "tokens": tokens,
        "labels": labels
    }


def annotate_resume(resume_path: str):
   
    resume_text = parse(resume_path)

    resume_text = clean_text(resume_text)

    entities = extract_entities(resume_text[:6000])

    spans = entities_to_spans(entities)

    bio = spans_to_token_bio(resume_text, spans)

    bio_json = bio_to_json(bio)

    return bio_json



