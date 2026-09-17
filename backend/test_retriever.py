from app.services.retriever import Retriever

retriever = Retriever()

print("Stored chunks:", retriever.vector_store.count)

results = retriever.retrieve(
    "What projects has Barath worked on?"
)

print("\nRetrieved results:")

for result in results:
    print("\n" + "=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Chunk: {result['chunk_id']}")
    print(result["text"])
