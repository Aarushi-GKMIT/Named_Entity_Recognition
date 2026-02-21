# BERT-based Named Entity Recognition (NER)

This repository contains a **BERT-based Named Entity Recognition (NER)** model implemented using **PyTorch** and **Hugging Face Transformers**.  
The project focuses on handling **class imbalance** in NER using **weighted cross-entropy loss** and provides detailed **evaluation and visualization**.

---

## Features

- Token-level NER using **BERT**
- Handles **imbalanced datasets** via class-weighted loss
- Supports **BIO tagging scheme**
- Evaluation using **Precision, Recall, F1-score**
- Confusion matrix visualization
- Training using **Hugging Face Trainer API**

---

## Model Architecture

- **Base Model**: `bert-base-cased`
- **Task**: Token Classification (NER)
- **Loss Function**:
  - Weighted Cross Entropy Loss
  - `ignore_index = -100` for padded tokens

---

## Labeling Scheme

The model follows the **BIO format**:

- `B-ENTITY` → Beginning of an entity
- `I-ENTITY` → Inside an entity
- `O` → Outside any entity