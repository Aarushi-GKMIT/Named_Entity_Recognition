## Dataset Preparation

### Dataset Format

The dataset must be in **JSON Lines (`.jsonl`)** format.  
Each line represents one sequence with tokens and their BIO labels.

### Example (`dataset.jsonl`)
```json
{"tokens": ["John", "Doe", "Python", "developer"], "labels": ["B-NAME", "I-NAME", "B-SKILL", "O"]}
{"tokens": ["Worked", "at", "Google"], "labels": ["O", "O", "B-ORG"]}


