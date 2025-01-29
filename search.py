from openai import OpenAI
from pinecone import Pinecone

# Set up OpenAI API key
client = OpenAI(api_key="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")

# Initialize Pinecone
pc = Pinecone(
    api_key="pcsk_2RwDCV_9iPmpUCuQNbfxUh3X46JPwyr5xMEEyAt6LXubq8dHtA6HW6iVzretqXAi5oMRD1",
    environment="us-west1-gcp"  # Replace with your Pinecone environment
)

# Connect to the Pinecone index
index = pc.Index("vmrs")  # Ensure the index name matches your Pinecone dashboard

# Function to generate embeddings
def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )

    return response.data[0].embedding

# Query the Pinecone index
# query = "change tires 001-"
query = "tire replacement 001-"
# query = "ABS control module replacement 013-"


query_embedding = generate_embedding(query)

# Search the index using keyword arguments
results = index.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True  # Ensure metadata is included in the response
)

# Print results
for match in results['matches']:
    # The metadata contains the description of the VMRS code
    id = match['id']
    metadata = match.get('metadata', {})  # Get metadata (description of VMRS code)
    description = metadata.get("description", "No description found")
    
    print(f"ID: {id}, Score: {match['score']}, Description: {description}")