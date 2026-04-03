from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
TextCrossEncoder.list_supported_models()
encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
dense_embedding_model = TextEmbedding(model_name=encoder_name)
reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')
descriptions_embeddings = list(
    dense_embedding_model.embed(descriptions)
)
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")  # Qdrant is running from RAM.
client.create_collection(
    collection_name="movies",
    vectors_config={
        "embedding": models.VectorParams(
            size=client.get_embedding_size("sentence-transformers/all-MiniLM-L6-v2"), 
            distance=models.Distance.COSINE
        )
    }
)
client.upload_points(
    collection_name="movies",
    points=[
        models.PointStruct(
            id=idx, 
            payload={"description": description}, 
            vector={"embedding": vector}
        )
        for idx, (description, vector) in enumerate(
            zip(descriptions, descriptions_embeddings)
        )
    ],
)

# First-stage retrieval
query = "A story about a strong historically significant female figure."
query_embedded = list(dense_embedding_model.query_embed(query))[0]

initial_retrieval = client.query_points(
    collection_name="movies",
    using="embedding",
    query=query_embedded,
    with_payload=True,
    limit=10
)

description_hits = []
for i, hit in enumerate(initial_retrieval.points):
    print(f'Result number {i+1} is \"{hit.payload["description"]}\"')
    description_hits.append(hit.payload["description"])
	
	
new_scores = list(
    reranker.rerank(query, description_hits)
)  # returns scores between query and each document

ranking = [
    (i, score) for i, score in enumerate(new_scores)
]  # saving document indices
ranking.sort(
    key=lambda x: x[1], reverse=True
)  # sorting them in order of relevance defined by reranker

for i, rank in enumerate(ranking):
    print(f'''Reranked result number {i+1} is \"{description_hits[rank[0]]}\"''')