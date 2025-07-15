from pinecone import Pinecone, ServerlessSpec
import os

import time
from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path
from uuid import uuid4
import spacy

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY,proxy_url="http://fastweb.int.bell.ca:8083", ssl_verify=False)

embedding_model_name="multilingual-e5-large"
index_name = "tutorial"
dimension=1024
metrics="cosine"
cloud="aws"
region="us-east-1"
namespace="ns3"

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

nlp =  spacy.load("en_core_web_sm")

def extract_chunks_from_pdf_spacy(pdf_path, max_tokens=500):
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    
    doc = nlp(full_text)
    sentences = [sent.text.strip().replace('\n', ' ') for sent in doc.sents if sent.text.strip()]
    
    chunks = []
    current_chunk = []
    current_length =0
    for sentence in sentences:
        token_len=len(nlp(sentence))
        
        if current_length + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_len  # reset length for new chunk            
        else:
            current_chunk.append(sentence)
            current_length += token_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

pdf_file = Path("/mnt/c/Users/mohammadreza.yaesubi/Documents/Books/Dictionary-of-English-Idioms.pdf")
chunks = extract_chunks_from_pdf_spacy(pdf_file)
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
log_data = []
for chunk, embedding in zip(chunks, embeddings):
    vector_id = str(uuid4())
    vector ={
        "id": vector_id,
        "values": embedding['values'],
        "metadata": {
            "text": chunk,
            "source": os.path.basename(pdf_file),
            "category": "document",
            "type": "pdf"
        }
    }
    vectors.append(vector)
    log_data.append({
        "id": vector_id,
        "source": vector["metadata"]["source"],
        "category": vector["metadata"]["category"]  ,
        "type": vector["metadata"]["type"]
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
    
log_file = f"vector_log_{os.path.splitext(os.path.basename(pdf_file))[0]}.json"
log_file = Path(Path.cwd(), log_file)
with open(log_file, "w") as f:
    import json
    json.dump(log_data, f, indent=2)
    
print(f"Vector log saved to {log_file}")

print(index.describe_index_stats())
    
