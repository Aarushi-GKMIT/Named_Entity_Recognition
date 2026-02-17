import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()

    # remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # remove phone numbers
    text = re.sub(
        r"\b(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b",
        " ",
        text
    )

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # keep only alphabets
    text = re.sub(r"[^a-z\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords + single-letter tokens
    tokens = [
        w for w in text.split()
        if w not in STOPWORDS and len(w) > 1
    ]

    return " ".join(tokens)