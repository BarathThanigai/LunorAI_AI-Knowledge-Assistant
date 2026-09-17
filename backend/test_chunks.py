from app.services.document_processor import process_pdf

chunks = process_pdf(r"D:\Projects\LunorAI\test.pdf")

print("Chunks:", len(chunks))

for chunk in chunks[:10]:
    print(f"\n--- {chunk['chunk_id']} | page {chunk['page']} ---")
    print(chunk["text"])
