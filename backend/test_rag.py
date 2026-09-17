from app.services.rag import answer_question


result = answer_question(
    "What projects has Barath worked on?"
)


print("\nAnswer:")
print(result["answer"])


print("\nSources:")

for source in result["sources"]:
    print(
        f"- {source['document']} "
        f"(page {source['page']}, "
        f"score={source['score']})"
    )