from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY,proxy_url="http://fastweb.int.bell.ca:8083", ssl_verify=False)

index_name = "tutorial"
# pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec( 
#         scale_to_zero=True,
#         min_replica_count=1,
#         max_replica_count=1,
#         autoscale_enabled=True,
#         autoscale_target_qps=1000
#         )
    
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
index = pc.Index(index_name) 

#query
query="What temperature does water boil at?"
embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={"input_type": "query"}
)

print(embedding)

results = index.query(namespace="ns1",
                      vector=embedding[0].values,
                      top_k=1,
                      include_values=False,
                      include_metadata=True)

print(results)

response = index.fetch(
    ids=['rec1', 'rec5'],
    namespace="ns1"
)
print(response)
for vector_id, vector_data in response.vectors.items():
    print(f"ID: {vector_id}, Metadata: {vector_data.metadata}")


print(index.describe_index_stats())

# response = index.delete(ids=['rec1', 'rec5'], namespace="ns1")
# print(index.describe_index_stats())

