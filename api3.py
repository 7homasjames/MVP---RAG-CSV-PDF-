from fastapi import FastAPI
from dotenv import load_dotenv
import os
import hashlib
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from semantic_kernel.functions import KernelArguments
import chromadb
import json

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings, OpenAITextEmbedding
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin


# Initialize Semantic Kernel
kernel = Kernel()

# Prompt Template for Chat Completion with Grounding
prompt_template = """
    You are a chatbot that can have a conversation about any topic related to the provided context.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    Provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

#if os.getenv("GLOBAL_LLM_SERVICE") == "OpenAI":

    # Add OpenAI Chat Completion Service
openai_service = OpenAIChatCompletion(
    api_key=os.getenv("OPENAI_API_KEY"),
    ai_model_id="gpt-3.5-turbo"
)
kernel.add_service(openai_service)

chat_execution_settings = OpenAIChatPromptExecutionSettings(
    ai_model_id="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.0,
    top_p=0.5
)


chat_prompt_template_config = PromptTemplateConfig(
    template=prompt_template,
    name="grounded_response",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="db_record", description="The database record", is_required=True),
        InputVariable(name="query_term", description="The user input", is_required=True),
    ],
    execution_settings=chat_execution_settings,
)

chat_function = kernel.add_function(
        function_name="ChatGPTFunc",
        plugin_name="chatGPTPlugin",
        prompt_template_config=chat_prompt_template_config
        )


# Configure environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get collection
collection = chroma_client.get_or_create_collection(name="my_collection")
#print("Collection:", collection)

# Function to generate a unique hash for a given text
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Function to upsert documents with unique hash IDs
def upsert_documents(documents):
    # Prepare documents and metadata for ChromaDB
    ids = [generate_hash(doc['line']) for doc in documents]
    texts = [doc['line'] for doc in documents]  # Use 'line' as the document content
    embeddings = [doc['embedding'] for doc in documents] 

    #print("Upserting IDs:", ids)
    #print("Upserting Texts:", texts)
    #print("Upserting Embeddings:", embeddings)
    # Upsert into ChromaDB
    #print("2")
    collection.upsert(documents=texts, ids=ids, embeddings = embeddings)
    #print(collection)
    return texts


# Function to query the collection
def query_collection(query_texts, n_results):
    #print("Done")
    #print(f"Query Texts: {query_texts}")
    #print(f"Number of Results: {n_results}")
    all_docs = collection.get()
    #print("Stored Documents:", all_docs)
    try:
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=["documents", "embeddings"]
        )
        #print("Query Results:", results)
        return results
    except Exception as e:
        #print(f"Error during query: {e}")
        return None


# Load the model
model = SentenceTransformer(os.getenv("MODEL_NAME"))

app = FastAPI()

# Data models
class Item(BaseModel):
    id: str
    line: str
    filename: str
    page_number: str = "1"  # Default to "1" if missing

class Docs(BaseModel):
    items: List[Item]


class Query(BaseModel):
    query: str

class QA(BaseModel):
    query: str
    context: str

@app.post("/push_docs/")
async def push_docs(item: Docs):
    try:
        docs = item.dict()["items"]
        #print("Received documents:", docs)

        # Add embeddings
        for doc in docs:
            doc['embedding'] = model.encode(doc['line']).tolist()

        # Debugging: Check before upsert
        #print("Prepared for upsertion:", docs)

        ids = upsert_documents(docs)
        return {"status": "success", "inserted_ids": ids}
    except Exception as e:
        print(f"Error in push_docs: {e}")
        return {"error": str(e)}

@app.post("/context/")
async def context(item: Query):
    try:
        query = item.query
        query_embedding = model.encode(query).tolist()

        # Query ChromaDB directly
        results = query_collection(query_texts=[query], n_results=5)

        # Prepare the response
        if results and "documents" in results:
            response_docs = [
                json.loads(result) for group in results["documents"] for result in group
            ]
            return {
                "docs": response_docs,
                "filenames": [doc.get("filename", "Unknown") for doc in response_docs],
                "page_numbers": [doc.get("page_number", "Unknown") for doc in response_docs],
            }
        return {"error": "No results found"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/response/")
async def response(item: QA):
    try:
        query = item.query
        context = item.context
        arguments = KernelArguments(db_record=context, query_term=query)

        result = await kernel.invoke(
            chat_function,arguments
        )


        # Example response using context
        #print("Result_OG",result)

        return {"output" : f"{result}" }
    except Exception as e:
        return {"error": str(e)}
