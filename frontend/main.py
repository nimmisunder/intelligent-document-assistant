
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Intelligent Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Assistant")

st.write("Manage your documents and build your AI knowledge base.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_documents():
    try:
        response = requests.get(
            f"{API_URL}/documents/",
            timeout=10
        )
        response.raise_for_status()
        return response.json()["documents"]

    except requests.RequestException:
        st.error("Unable to connect to FastAPI.")
        return []


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Document Management",
        "AI Chat",
        "Evaluation"
    ]
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Dashboard")

    documents = get_documents()

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Documents",
        len(documents)
    )

    col2.metric(
        "Total Pages",
        sum(doc["total_pages"] for doc in documents)
    )


# --------------------------------------------------
# DOCUMENT MANAGEMENT
# --------------------------------------------------

elif page == "Document Management":

    st.header("Document Management")

    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("Upload Document"):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files=files,
                    timeout=300
                )

                if response.ok:
                    result = response.json()

                    st.success(
                        "Document uploaded and indexed successfully!"
                    )

                    st.write(
                        f"Total pages: {result['total_pages']}"
                    )

                    st.write(
                        f"Total chunks: {result['total_chunks']}"
                    )

                    st.rerun()

                else:
                    st.error(response.text)

            except requests.RequestException as error:
                st.error(f"Upload failed: {error}")

    st.subheader("Uploaded Documents")

    documents = get_documents()

    if not documents:
        st.info("No documents uploaded yet.")

    for doc in documents:

        col1, col2, col3 = st.columns([4, 1, 1])

        col1.write(doc["original_filename"])
        col2.write(f'{doc["total_pages"]} pages')

        if col3.button("Delete", key=doc["id"]):

            try:
                delete_response = requests.delete(
                    f'{API_URL}/documents/{doc["id"]}',
                    timeout=120
                )

                if delete_response.ok:
                    st.success("Document deleted.")
                    st.rerun()

                else:
                    st.error(delete_response.text)

            except requests.RequestException as error:
                st.error(f"Delete failed: {error}")


# --------------------------------------------------
# AI CHAT
# --------------------------------------------------

elif page == "AI Chat":

    st.header("💬 AI Document Assistant")

    st.write(
        "Ask questions about your uploaded PDF documents."
    )

    documents = get_documents()

    # Document selection
    document_options = {
        "All Documents": None
    }

    for doc in documents:
        document_options[
            doc["original_filename"] + " (" + doc["id"][:8] + ")"
        ] = doc["id"]

    selected_document = st.selectbox(
        "Select document",
        options=list(document_options.keys())
    )

    selected_document_id = document_options[selected_document]

    # Clear conversation
    col1, col2 = st.columns([5, 1])

    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander("📚 View Sources"):

                        for source in sources:

                            st.markdown(
                                f"**{source['filename']}**"
                            )

                            st.write(
                                f"Page: {source['page_number']}"
                            )

                            st.caption(
                                f"Chunk: {source['chunk_id']}"
                            )

                            st.divider()

    # User question
    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        payload = {
            "question": question,
            "document_id": selected_document_id
        }

        try:

            with st.spinner("Searching documents and generating answer..."):

                response = requests.post(
                    f"{API_URL}/chat/ask",
                    json=payload,
                    timeout=240
                )

                response.raise_for_status()

                result = response.json()

                answer = result.get(
                    "answer",
                    result.get("message", "No answer returned.")
                )

                sources = result.get("sources", [])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except requests.RequestException as error:

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Unable to get an answer from the API: {error}",
                "sources": []
            })

        st.rerun()


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

elif page == "Evaluation":

    st.header("📊 RAG Evaluation Dashboard")

    st.write(
        "Monitor the retrieval quality of your Intelligent Document Assistant."
    )

    import json
    from pathlib import Path
    import pandas as pd

    BASE_DIR = Path(__file__).resolve().parent.parent

    RESULTS_PATH = (
        BASE_DIR / "evaluation" / "evaluation_results.json"
    )

    if not RESULTS_PATH.exists():

        st.warning(
            "No evaluation results found. "
            "Please run evaluate_retrieval.py first."
        )

        if st.button("Run Evaluation Now"):

            import subprocess
            import sys

            with st.spinner("Running RAG evaluation..."):

                process = subprocess.run(
                    [
                        sys.executable,
                        str(BASE_DIR / "evaluate_retrieval.py")
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(BASE_DIR)
                )

            if process.returncode == 0:
                st.success("Evaluation completed successfully!")
                st.code(process.stdout)
                st.rerun()

            else:
                st.error("Evaluation failed.")
                st.code(process.stderr)

    else:

        with open(RESULTS_PATH, "r", encoding="utf-8") as file:
            evaluation_data = json.load(file)

        metrics = evaluation_data["average_metrics"]

        st.subheader("Evaluation Overview")

        st.info(
            f"Questions evaluated: {evaluation_data['total_questions']}"
        )

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision@5",
            f"{metrics['precision_at_k']:.2%}"
        )

        col2.metric(
            "Recall@5",
            f"{metrics['recall_at_k']:.2%}"
        )

        col3.metric(
            "MRR",
            f"{metrics['mrr']:.2%}"
        )

        col4.metric(
            "Hit Rate@5",
            f"{metrics['hit_rate_at_k']:.2%}"
        )

        st.divider()

        # Retrieval performance chart
        st.subheader("Retrieval Performance")

        chart_data = pd.DataFrame({
            "Metric": [
                "Precision@5",
                "Recall@5",
                "MRR",
                "Hit Rate@5"
            ],
            "Score": [
                metrics["precision_at_k"],
                metrics["recall_at_k"],
                metrics["mrr"],
                metrics["hit_rate_at_k"]
            ]
        })

        st.bar_chart(
            chart_data.set_index("Metric")
        )

        st.divider()

        # Individual results
        st.subheader("Individual Evaluation Results")

        individual_results = evaluation_data.get(
            "question_results", []
        )

        for index, result in enumerate(individual_results):

            with st.expander(f"Question {index + 1}: {result['question']}"):

                st.write("Retrieved chunks:")

                st.write(result["retrieved_chunks"])

                question_metrics = result["metrics"]

                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

                col1.metric(
                    "Precision@5",
                    f"{question_metrics['precision_at_k']:.2%}"
                )

                col2.metric(
                    "Recall@5",
                    f"{question_metrics['recall_at_k']:.2%}"
                )

                col3.metric(
                    "MRR",
                    f"{question_metrics['mrr']:.2%}"
                )

                col4.metric(
                    "Hit Rate@5",
                    f"{question_metrics['hit_rate_at_k']:.2%}"
                )

        st.caption(
            "Metrics are calculated from your labelled evaluation dataset."
        )

        if st.button("Refresh Evaluation"):
            st.rerun()
