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
import sqlite3

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
        response = requests.post(get_url("context"), json=payload)
        response.raise_for_status()

        data = response.json()

        docs = data.get("docs", [])
        filenames = data.get("filenames", [])
        page_numbers = data.get("page_numbers", [])

        context = "\n".join([doc.get("text", "") for doc in docs if isinstance(doc, dict)])
        return context, filenames, page_numbers
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching context: {e}")
        return None, None, None

def get_response(user_input):
    context, filenames, page_numbers = get_context(user_input)
    if not context:
        return "Unable to fetch context.", filenames, page_numbers
    try:
        payload = {"query": user_input, "context": context}
        response = requests.post(get_url("response"), json=payload)
        response.raise_for_status()

        return response.json()["output"], filenames, page_numbers
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching response: {e}")
        return "Unable to fetch response.", filenames, page_numbers

# SQLite Setup
DB_PATH = "data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                content TEXT,
                page_number INTEGER
            )
        """)
        conn.commit()

def store_in_db(filename, content, page_number):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (filename, content, page_number)
            VALUES (?, ?, ?)
        """, (filename, content, page_number))
        conn.commit()

def retrieve_from_db(query):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        print(cursor)
        cursor.execute("""
            SELECT filename, content, page_number
            FROM documents
            WHERE content LIKE ?
        """, (f"%{query}%",))
        print(cursor.fetchall())
        return cursor.fetchall()

# Initialize Database
init_db()

# Sidebar
with st.sidebar:
    st.title("PDF & CSV Question Answering System 💬")
    use_database = st.checkbox("Use SQLite Database for Retrieval", value=False)
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
                        if uploaded_file.name.endswith(".csv"):
                            csv_data = asyncio.run(load_and_process_csv(uploaded_file))
                            for row in csv_data:
                                store_in_db(uploaded_file.name, json.dumps(row["data"]), None)
                        elif uploaded_file.name.endswith(".pdf"):
                            reader = PyPDF2.PdfReader(uploaded_file)
                            for page_number, page in enumerate(reader.pages, start=1):
                                text = page.extract_text()
                                store_in_db(uploaded_file.name, text, page_number)

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

    if use_database:
        results = retrieve_from_db(user_input)
        print(results)
        if not results:
            bot_response = "No relevant data found."
        else:
            context = "\n".join([f"Page {row[2]}: {row[1]}" for row in results])
            filenames = list(set(row[0] for row in results))
            bot_response = f"{context}\n\nReferences:\n{', '.join(filenames)}"
    else:
        response, filenames, page_numbers = get_response(user_input)
        bot_response = f"{response}\n\nReferences:\n{uploaded_file.name}"

    st.session_state.chat_history.append((bot_response, True))
    st.chat_message("assistant").write(bot_response)
