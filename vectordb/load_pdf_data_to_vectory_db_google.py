from pinecone import Pinecone, ServerlessSpec
import os

import time
from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



embedding_model_name="multilingual-e5-large"
PINECONE_INDEX_NAME = "langchain-pdf-index"
dimension=1024
metrics="cosine"
cloud="aws"
region="us-east-1"
PINECONE_NAMESPACE = "ns3"

GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"
PDF_DIRECTORY = Path(Path.home(),"pdf_files")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

try:
    pc = Pinecone(api_key=PINECONE_API_KEY,proxy_url="http://fastweb.int.bell.ca:8083", ssl_verify=False)
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
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

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    print(f"Using Google Generative AI model: {GOOGLE_EMBEDDING_MODEL} initialized successfully.")
except Exception as e:
    print(f"An error occurred while initializing the Google Generative AI model: {e}")
    exit()
    
index = pc.Index(PINECONE_INDEX_NAME) 

print("index created successfully.stats:", index.describe_index_stats())

print(f"\nLoading documents from directory: {PDF_DIRECTORY}")

if not PDF_DIRECTORY.exists() or not PDF_DIRECTORY.is_dir():
    print(f"Directory {PDF_DIRECTORY} does not exist or is not a directory.")
    exit()
    
try:
    loader = DirectoryLoader(PDF_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True,use_multithreading=True)
    documents = loader.load()
    #print(documents)
    
    if not documents:
        print(f"No PDF documents found in {PDF_DIRECTORY}.")
        exit()
    print(f"Loaded {len(documents)} pages  from PDF files in {PDF_DIRECTORY}.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(docs_chunks)} chunks.")
    
except Exception as e:
    print(f"Failed during document loading or splitting: {e}")
    
    

print("Setting up Pinecone Vector Store...")

try:
    vector_store = PineconeVectorStore.from_documents(docs_chunks,embedding=embeddings,namespace=PINECONE_NAMESPACE,index=index,
                                                      proxy_url="http://fastweb.int.bell.ca:8083", ssl_verify=False)
    
    print(f"Pinecone Vector Store initialized successfully {PINECONE_NAMESPACE}")
    time.sleep(5)
    print(f"Vector Store contains {vector_store.count()} vectors.")
    print("Index stats:", index.describe_index_stats())
except Exception as e:
    print(f"An error occurred while initializing the Pinecone Vector Store: {e}")
    exit()
    
    
# try:
#     embeddings = pc.inference.embed(
#         model=embedding_model_name,
#         inputs=chunks,
#         parameters={"input_type": "passage", "truncate": "END"}
        
#     )
# except Exception as e:
#     print(f"An error occurred while embedding: {e}")
#     exit()
    
# vectors = []
# for chunk, embedding in zip(chunks, embeddings):
#     vectors.append({
#         "id": str(uuid4()),
#         "values": embedding['values'],
#         "metadata": {
#             "text": chunk,
#             "source": os.path.basename(pdf_file),
#             "category": "document",
#             "type": "pdf"
#         }
#     })
    
# print(index.describe_index_stats())

# try:
#     index.upsert(
#         namespace="ns2",
#         vectors=vectors,
#     )
#     print(f"Upserted {len(vectors)} records to the index.")
# except Exception as e:
#     print(f"An error occurred while upserting records: {e}")
    
    
# time.sleep(3)  # Sleep to avoid rate limiting
# print(index.describe_index_stats())
    
