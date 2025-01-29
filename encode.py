import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="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")
import pinecone
from pinecone import Pinecone

# Set up Pinecone
# Initialize Pinecone
pc = Pinecone(
    api_key="pcsk_2RwDCV_9iPmpUCuQNbfxUh3X46JPwyr5xMEEyAt6LXubq8dHtA6HW6iVzretqXAi5oMRD1"
)

index = pc.Index("vmrs")  # Ensure the index name matches exactly

# Load your data
df = pd.read_csv("/Users/vedantkhattar/Downloads/vmrscodeswithdescription.csv")
texts = df["text"].tolist()

# Function to generate embeddings
def generate_embedding(text):
    response = client.embeddings.create(input=text,
    model="text-embedding-ada-002")
    return response.data[0].embedding

# Insert data into Pinecone
for i, text in enumerate(texts):
    # Generate embedding
    embedding = generate_embedding(text)

   # Insert into Pinecone with metadata
    index.upsert([
        {
            "id": str(i), 
            "values": embedding, 
            "metadata": {"description": text}  # Store the text as metadata
        }
    ])
      # Store the text as metadata

    # Print progress
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} rows...")

print("Data insertion complete!")