from pathlib import Path

from app.rag_pipeline import RAGPipeline


def main():
    pdf_path = Path("sample_rag_document.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {pdf_path}"
        )

    print("\nInitializing RAG pipeline...")

    pipeline = RAGPipeline()

    print("\nIndexing document...")

    result = pipeline.index_document(pdf_path)

    print("\nIndexing complete!")
    print(f"Document: {result['filename']}")
    print(f"Pages: {result['page_count']}")
    print(f"Chunks: {result['chunk_count']}")

    print("\nAsk a question about your document.")
    question = input("Question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nGenerating answer...")

    response = pipeline.ask(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response["answer"])

    print("\nSOURCES")
    print("=" * 60)

    for source in response["sources"]:
        print(
            f"Page: {source['page_number']} | "
            f"Similarity: {source['similarity']:.3f}"
        )
        print(source["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()