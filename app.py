import os

# Disable the interactive progress bar to avoid the tqdm/FloatProgress crash
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

st.set_page_config(page_title="Legal Precedent Finder", layout="wide")


@st.cache_resource
def get_vector_db():
  client = chromadb.Client()
  # Use full model name to avoid ambiguous hub resolution
  embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      normalize_embeddings=True,
  )

  collection = client.get_or_create_collection(
      name="legal_docs",
      embedding_function=embed_fn,
      metadata={"hnsw:space": "cosine"},
  )

  if collection.count() == 0:
    docs = [
        (
            "NDA violation: Ex-engineer exfiltrated proprietary training"
            " datasets and source code to rival firm."
        ),
        (
            "Commercial lease breach: Tenant defaulted on payments citing"
            " unforeseen market drop."
        ),
        (
            "Patent infringement: Competitor replicated patented hardware"
            " sensor telemetry pipeline."
        ),
        (
            "Severance and non-compete clause dispute: Restrictive covenant"
            " ruled overly broad and unenforceable."
        ),
    ]
    metas = [
        {
            "title": "Alpha v. Beta Corp",
            "outcome": "Full Damages Awarded to Plaintiff",
        },
        {
            "title": "Realty Partners v. Urban Store",
            "outcome": "Partial Repayment Order",
        },
        {
            "title": "SensorTech v. MicroDevices",
            "outcome": "Preliminary Injunction Granted",
        },
        {
            "title": "Executive v. CloudServices",
            "outcome": "Case Dismissed, Covenant Voided",
        },
    ]
    ids = [f"id_{i}" for i in range(len(docs))]
    collection.add(ids=ids, documents=docs, metadatas=metas)

  return collection


collection = get_vector_db()

st.title("⚖️ Legal Document Similarity & Outcome Predictor")
st.markdown(
    "Query legal dispute records using semantic embeddings"
    " (`all-MiniLM-L6-v2`) and cosine similarity."
)

col1, col2 = st.columns([3, 1])
with col1:
  query = st.text_area(
      "Enter Case Facts / Claim Description:",
      value=(
          "A former software developer stole confidential algorithms and"
          " joined a competing startup."
      ),
  )
with col2:
  top_k = st.slider("Top K Matches", min_value=1, max_value=4, value=2)

if st.button("Analyze Similar Cases"):
  results = collection.query(
      query_texts=[query],
      n_results=top_k,
      include=["documents", "metadatas", "distances"],
  )

  st.subheader("Matching Precedents & Predictions")
  for i in range(len(results["documents"][0])):
    dist = results["distances"][0][i]
    sim_pct = round(max(0.0, 1.0 - dist) * 100, 2)
    meta = results["metadatas"][0][i]
    doc = results["documents"][0][i]

    with st.expander(
        f"#{i+1} — {meta['title']} | Similarity: {sim_pct}%", expanded=True
    ):
      st.write(f"**Case Facts:** {doc}")
      st.write(f"**Historical Outcome:** `{meta['outcome']}`")
