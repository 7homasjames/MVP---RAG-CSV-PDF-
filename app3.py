import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
import asyncio
import requests
from tqdm import tqdm
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from prep import load_and_process_csv
from csvplugin import CSVPlugin
import json
import PyPDF2

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please check your environment variables.")

kernel = Kernel()
kernel.add_plugin(CSVPlugin(), plugin_name="CSV")

# Helper Functions
def get_url(endpoint):
    return f"http://127.0.0.1:8000/{endpoint}/"

def get_context(user_input):
    try:
        payload = {"query": user_input}
        #print("payload:",payload)
        response = requests.post(get_url("context"), json=payload)
        #print("Response:",response)
        #response.raise_for_status()
        data = response.json()
        print("Docs:",data)
        docs = [str(doc) for doc in data["docs"]]  # Convert each dictionary to a string
        return "\n".join(docs), data["filenames"], data["page_number"]
        #return "\n".join(data["docs"]), data["filenames"], data["page_number"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching context: {e}")
        return None, None, None

def get_response(user_input):
    context, files, page_no = get_context(user_input)
    if not context:
        return "Unable to fetch context.", "", ""
    try:
        payload = {"query": user_input, "context": context}
        response = requests.post(get_url("response"), json=payload)
        #print(response.json())
        #response.raise_for_status()

        return response.json()["output"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching response: {e}")
        return "Unable to fetch response.", "", ""


# Sidebar
with st.sidebar:
    st.title("PDF & CSV Question Answering System 💬")
    uploaded_files = st.file_uploader(
        "Choose CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        if "files" not in st.session_state:
            st.session_state["files"] = []

        with st.spinner("Indexing documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state["files"]:
                    try:
                        # Process CSV files
                        if uploaded_file.name.endswith(".csv"):
                            csv_data = asyncio.run(load_and_process_csv(uploaded_file))
                            file_type = "csv"
                        # Process PDF files
                        elif uploaded_file.name.endswith(".pdf"):
                            pdf_data = []
                            reader = PyPDF2.PdfReader(uploaded_file)
                            for page_number, page in enumerate(reader.pages, start=1):
                                text = page.extract_text()
                                pdf_data.append(
                                    {
                                        "index": page_number,
                                        "data": {"text": text},
                                        "filename": uploaded_file.name,
                                        "page_number": page_number,
                                    }
                                )
                            csv_data = pdf_data  # Reuse the same logic for indexing
                            file_type = "pdf"

                        # Index documents
                        for i in tqdm(range(0, len(csv_data), 10)):
                            payload = {
                                "items": [
                                    {
                                        "id": str(item["index"]),
                                        "line": json.dumps(item["data"]),
                                        "filename": uploaded_file.name,
                                        "page_number": str(item["page_number"]),
                                    }
                                    for item in csv_data
                                ]
                            }
                            print("Payload being sent to /push_docs/:")
                            #print(json.dumps(payload, indent=2))
                            response = requests.post(get_url("push_docs"), json=payload)
                            response.raise_for_status()

                        st.session_state["files"].append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Failed to index {uploaded_file.name}: {e}")

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Your Message:")
if user_input:
    for message, is_bot_response in st.session_state.chat_history:
        if is_bot_response:
            st.chat_message("assistant").write(message)
        else:
            st.chat_message("user").write(message)

    st.session_state.chat_history.append((user_input, False))
    st.chat_message("user").write(user_input)

    #resp, files, page_number = get_response(user_input)
    resp = get_response(user_input)
    bot_response = f"{resp}\n\nReferences:\n{uploaded_file.name}\n"
    st.session_state.chat_history.append((bot_response, True))
    st.chat_message("assistant").write(bot_response)
