from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv,dotenv_values
import json
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from uuid import uuid4
import pandas as pd
from csvplugin import CSVPlugin
import streamlit as st


def get_pdf_data(file_, num_pages = 1):
    reader = PdfReader(file_)
    text_data = []
    pages = reader.pages
    num_pages = len(pages) 
    
    try:
        for page in range(num_pages):
            current_page = reader.pages[page]
            text = current_page.extract_text()
            if text:  # Ensure text is not None
                text_data.append((text, page + 1))
    except:
        print("Error reading file")
    finally:
        return text_data


async def load_and_process_csv(uploaded_file):
    csv_plugin = CSVPlugin()
    success = await csv_plugin.load_csv(uploaded_file)
    if success:
        data = await csv_plugin.get_data()
        print("Processed CSV Data:")
        print(data)
        return data
    else:
        st.error(f"Failed to load CSV file {uploaded_file.name}")
        return None

    
def get_chunks(text_data, chunk_length=500):
    chunks = []
    print("text" ,text_data)
    exit()
    for text, page_num in text_data:
        
        
        while len(text) > chunk_length:
            last_period_index = text[:chunk_length].rfind('.')
            if last_period_index == -1:
                last_period_index = chunk_length
            chunks.append((text[:last_period_index].strip(), page_num))
            text = text[last_period_index+1:].strip()
        if text:  # Ensure no empty strings are added
            chunks.append((text.strip(), page_num))
    return chunks



def embed(text):
    model = SentenceTransformer(os.getenv("MODEL_NAME"))
    query_vector = model.encode(text).reshape(-1).astype(float).tolist()
    return query_vector




def get_embeddings(text, fn, page_num):
    # query_vector = embed(text)
    resp = {"id" : f"{uuid4()}",
            "line" : text,
            "filename" : fn,
            "page_number":f"{page_num}"}
            # "embedding" : query_vector}

    return resp