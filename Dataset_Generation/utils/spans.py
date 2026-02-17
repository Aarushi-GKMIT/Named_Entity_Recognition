from schemas.entities_schema import ResumeEntities
import re
from typing import List, Dict, Tuple

def entities_to_spans(entities: ResumeEntities):
    spans = []

    def add(text, label):
        if text:
            spans.append({"text": text.strip(), "label": label})

    add(entities.name, "NAME")

    for skill in entities.skills:
        add(skill, "SKILL")

    for company in entities.companies:
        add(company, "ORG")

    for desig in entities.designations:
        add(desig, "DESIGNATION")

    for award in entities.awards:
        add(award, "AWARD")

    for cert in entities.certificates:
        add(cert, "CERTIFICATE")

    for project in entities.projects:
        add(project, "PROJECT")

    for edu_ins in entities.education:
        add(edu_ins.institution, "EDUCATION_INSTITUTION")

    for edu_deg in entities.education:
        add(edu_deg.degree, "EDUCATION_DEGREE")

    for edu_year in entities.education:
        add(edu_year.year, "EDUCATION")
        
    for exp in str(entities.experience_years):
        add(exp, "EXPERIENCE")

    return spans


def tokenize(text: str) -> List[str]:
    """
    Token-level tokenizer suitable for BIO tagging.
    Splits words, numbers, and punctuation separately.
    """
    return re.findall(r"\w+|[^\w\s]", text)


def normalize(token: str) -> str:
    return token.lower()


def spans_to_token_bio(
    text: str,
    spans: List[Dict[str, str]]
) -> List[Tuple[str, str]]:

    tokens = tokenize(text)

    # labels = []
    labels = ["O"] * len(tokens)

    norm_tokens = [normalize(t) for t in tokens]

    spans = sorted(spans, key=lambda s: len(s["text"]), reverse=True)

    for span in spans:
        span_tokens = tokenize(span["text"])
        span_norm = [normalize(t) for t in span_tokens]
        span_len = len(span_norm)
        tag = span["label"]

        for i in range(len(norm_tokens) - span_len + 1):

            if any(labels[i + j] != "O" for j in range(span_len)):
                continue

            if norm_tokens[i:i + span_len] == span_norm:
                labels[i] = f"B-{tag}"
                for j in range(1, span_len):
                    labels[i + j] = f"I-{tag}"

    return list(zip(tokens, labels))

