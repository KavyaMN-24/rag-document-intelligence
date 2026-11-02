import pymupdf

document = pymupdf.open()

page1 = document.new_page()
page1.insert_text(
    (72, 72),
    """
    RAG Document Intelligence

    Retrieval-Augmented Generation (RAG) combines
    information retrieval with large language models.

    RAG retrieves relevant information from external
    documents before generating an answer.

    This helps language models answer questions using
    specific document content.
    """
)

page2 = document.new_page()
page2.insert_text(
    (72, 72),
    """
    Vector Databases

    Vector databases store numerical representations
    of text called embeddings.

    ChromaDB is a vector database that supports semantic
    similarity search.

    Sentence Transformers can generate embeddings
    locally without requiring a paid API.
    """
)

document.save("sample_rag_document.pdf")
document.close()

print("Sample PDF created successfully.")