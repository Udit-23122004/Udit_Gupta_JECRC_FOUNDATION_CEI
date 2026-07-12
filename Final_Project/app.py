from pathlib import Path
import streamlit as st
import time
import os
import pandas as pd

from datetime import datetime
from google import genai
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Page Configuration
st.set_page_config(
    page_title="DriveWise",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "drivewise_faiss"
# Models

BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "drivewise_faiss"


@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return vector_store, reranker


vector_store, reranker = load_models()


# Gemini Client
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# Query Section Detection
def detect_query_section(question):

    question = question.lower()

    if "cng" in question:
        return "cng_safety"

    elif any(word in question for word in [
        "adas", "lane", "collision", "emergency braking"
    ]):
        return "adas"

    elif any(word in question for word in [
        "safety", "airbag", "isofix", "stability"
    ]):
        return "safety"

    elif any(word in question for word in [
        "engine", "power", "torque", "transmission"
    ]):
        return "performance"

    elif any(word in question for word in [
        "infotainment", "speaker",
        "android auto", "apple carplay"
    ]):
        return "infotainment"

    elif any(word in question for word in [
        "seat", "comfort", "interior", "legroom"
    ]):
        return "interior"

    return "general"


# Logging
LOG_FILE = "drivewise_logs.csv"


def save_log(
    brand,
    model,
    question,
    section,
    response_time,
    status
):

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brand": brand,
        "model": model,
        "question": question,
        "section": section,
        "response_time": round(response_time, 2),
        "status": status
    }

    log_df = pd.DataFrame([log_data])

    log_df.to_csv(
        LOG_FILE,
        mode="a",
        header=not os.path.exists(LOG_FILE),
        index=False
    )


# RAG Pipeline
def ask_drivewise(brand, model, question):

    start_time = time.time()

    section = detect_query_section(question)

    try:

        retrieved_docs = vector_store.similarity_search(
            question,
            k=10,
            fetch_k=200,
            filter={
                "brand": brand,
                "model": model,
                "section": section
            }
        )

        if not retrieved_docs:

            retrieved_docs = vector_store.similarity_search(
                question,
                k=10,
                fetch_k=200,
                filter={
                    "brand": brand,
                    "model": model
                }
            )

        if not retrieved_docs:

            response_time = time.time() - start_time

            save_log(
                brand,
                model,
                question,
                section,
                response_time,
                "Failed"
            )

            return {
                "answer": "Information not available in the selected brochure.",
                "sources": [],
                "section": section
            }


        # Re-ranking
        pairs = [
            [question, doc.page_content]
            for doc in retrieved_docs
        ]

        scores = reranker.predict(pairs)

        reranked_docs = sorted(
            zip(retrieved_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )


        # Page Diversity
        top_docs = []
        seen_pages = set()

        for doc, score in reranked_docs:

            page = doc.metadata["page"]

            if page not in seen_pages:

                top_docs.append((doc, score))
                seen_pages.add(page)

            if len(top_docs) == 3:
                break


        context = "\n\n".join([
            doc.page_content
            for doc, score in top_docs
        ])


        prompt = f"""
You are DriveWise, an automotive assistant.

Answer the user's question only using the provided brochure context.

Do not use outside knowledge.

If the information is unavailable, say:
"Information not available in the selected brochure."

User Question:
{question}

Brochure Context:
{context}

Give a clear and concise answer.
"""


        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )


        sources = []

        for doc, score in top_docs:

            source = {
                "brochure": doc.metadata["source"],
                "page": doc.metadata["page"]
            }

            if source not in sources:
                sources.append(source)


        response_time = time.time() - start_time

        save_log(
            brand,
            model,
            question,
            section,
            response_time,
            "Success"
        )


        return {
            "answer": response.text,
            "sources": sources,
            "section": section
        }


    except Exception as e:

        response_time = time.time() - start_time

        save_log(
            brand,
            model,
            question,
            section,
            response_time,
            "Failed"
        )

        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "section": section
        }


# UI
st.title("🚗 DriveWise")

st.subheader(
    "Metadata-Aware Automotive RAG Assistant"
)

st.write(
    "Ask questions based on official car brochures."
)


car_data = {
    "Hyundai": ["Venue", "Creta"],
    "Tata": ["Nexon", "Punch"]
}


brand = st.selectbox(
    "Select Brand",
    list(car_data.keys())
)


model = st.selectbox(
    "Select Car Model",
    car_data[brand]
)


question = st.text_input(
    "Ask your question",
    placeholder="Example: What are the safety features?"
)


if st.button("Ask DriveWise"):

    if question.strip():

        with st.spinner(
            "Searching brochure..."
        ):

            result = ask_drivewise(
                brand,
                model,
                question
            )


        st.subheader("Answer")

        st.write(result["answer"])


        st.subheader("Sources")

        for source in result["sources"]:

            st.write(
                f"📄 {source['brochure']} | Page {source['page']}"
            )


        st.caption(
            f"Detected Section: {result['section']}"
        )

    else:

        st.warning(
            "Please enter a question."
        )
