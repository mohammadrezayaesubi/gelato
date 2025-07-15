from pinecone import Pinecone, ServerlessSpec
import os

import time
from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path
from uuid import uuid4
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY,proxy_url="http://fastweb.int.bell.ca:8083", ssl_verify=False)

embedding_model_name="multilingual-e5-large"
index_name = "tutorial"
dimension=1024
metrics="cosine"
cloud="aws"
region="us-east-1"

try:
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metrics,
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    # while True:
    #     status = pc.describe_index(index_name).status()
    #     if status.get("ready"):
    #         print(f"Index {index_name} is ready.")
    #         break
    #     else:
    #         print(f"Index {index_name} is not ready yet. Current status: {status}")
    #         time.sleep(5)
except Exception as e:
    print(f"An error occurred while creating the index: {e}")
    
index = pc.Index(index_name) 

def extract_chunk_from_pdf(pdf_path, chunk_size=500, overlap=50):
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    words = full_text.split()
    
    print(words)
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        #print(chunk)
        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())
            
    return chunks

pdf_file = Path("/mnt/c/Users/mohammadreza.yaesubi/Documents/Books/Dictionary-of-English-Idioms.pdf")
chunks = extract_chunk_from_pdf(pdf_file)
#print(chunks)

try:
    embeddings = pc.inference.embed(
        model=embedding_model_name,
        inputs=chunks,
        parameters={"input_type": "passage", "truncate": "END"}
        
    )
except Exception as e:
    print(f"An error occurred while embedding: {e}")
    exit()
    
vectors = []
for chunk, embedding in zip(chunks, embeddings):
    vectors.append({
        "id": str(uuid4()),
        "values": embedding['values'],
        "metadata": {
            "text": chunk,
            "source": os.path.basename(pdf_file),
            "category": "document",
            "type": "pdf"
        }
    })
    
print(index.describe_index_stats())

try:
    index.upsert(
        namespace="ns2",
        vectors=vectors,
    )
    print(f"Upserted {len(vectors)} records to the index.")
except Exception as e:
    print(f"An error occurred while upserting records: {e}")
    
    
time.sleep(3)  # Sleep to avoid rate limiting
print(index.describe_index_stats())
    
