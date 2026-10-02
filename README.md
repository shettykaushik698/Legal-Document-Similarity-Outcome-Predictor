# Legal Document Similarity & Outcome Predictor

A Streamlit application that compares a user-provided legal dispute description against a local database of precedent-style case summaries using sentence embeddings and cosine similarity. It surfaces the most relevant historical cases and presents the corresponding historic outcome.

## Overview

This project demonstrates a lightweight legal AI workflow:

- encodes legal case text with `sentence-transformers/all-MiniLM-L6-v2`
- stores the case records in a local Chroma vector database
- retrieves the closest matches for a new legal fact pattern
- displays precedent titles, similarity scores, and historical outcomes

## Tech stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal.

## Example query

Use a description like:

> A former software developer stole confidential algorithms and joined a competing startup.

The app will return the nearest legal cases and summarize their outcomes.

## GitHub repository

https://github.com/shettykaushik698/Legal-Document-Similarity-Outcome-Predictor