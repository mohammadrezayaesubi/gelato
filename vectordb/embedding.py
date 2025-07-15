from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY,proxy_url="http://fastweb.int.bell.ca:8083", ssl_verify=False)


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
    
records = [
    { "_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history" },
    { "_id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science" },
    { "_id": "rec3", "chunk_text": "Albert Einstein developed the theory of relativity.", "category": "science" },
    { "_id": "rec4", "chunk_text": "The mitochondrion is often called the powerhouse of the cell.", "category": "biology" },
    { "_id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature" },
    { "_id": "rec6", "chunk_text": "Water boils at 100°C under standard atmospheric pressure.", "category": "physics" },
    { "_id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history" },
    { "_id": "rec8", "chunk_text": "Honey never spoils due to its low moisture content and acidity.", "category": "food science" },
    { "_id": "rec9", "chunk_text": "The speed of light in a vacuum is approximately 299,792 km/s.", "category": "physics" },
    { "_id": "rec10", "chunk_text": "Newton's laws describe the motion of objects.", "category": "physics" }
]
#index = pc.Index(index_name)  
#index.upsert_records("ns1", records)    


embedding_model_name="multilingual-e5-large"
embedding_parameters = {"input_type": "passage", "truncate": "END"}
embedding_inputs = [record["chunk_text"] for record in records]


try:
    embeddings = pc.inference.embed(model=embedding_model_name,inputs=embedding_inputs,parameters=embedding_parameters)
    for embedding in embeddings:
        print(embedding)
except Exception as e:
    print(f"An error occurred while generating embeddings: {e}")

vectors = []
for d, e in zip(records, embeddings):
    vectors.append({
        "id": d["_id"],
        "values": e['values'],
        "metadata": {
            "chunk_text": d["chunk_text"],
            "category": d["category"]
        }
    })

try:
    index= pc.Index(index_name)
    index.upsert(
        namespace="ns1",
        vectors=vectors
    )
except Exception as e:
    print(f"An error occurred while upserting vectors: {e}")