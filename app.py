import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import os

# Load API keys from environment variables (replace with actual keys if testing locally)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "vmrs"

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Function to generate embeddings
def generate_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

# Function to search service tasks in Pinecone
def search_service_tasks(query, top_k=10):
    query_embedding = generate_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    return [
        f"{match['id']} - {match['metadata'].get('description', 'No description')}"
        for match in results["matches"]
    ]

# Streamlit UI
st.title("🔍 Service Task Search")

# User input field
search_query = st.text_input("Enter a service task:")

# When the user enters a query, perform search
if search_query:
    results = search_service_tasks(search_query)
    if results:
        selected_task = st.selectbox("Select a service task:", results)
        st.success(f"🔹 You selected: **{selected_task}**")
    else:
        st.warning("No results found. Try a different query.")
