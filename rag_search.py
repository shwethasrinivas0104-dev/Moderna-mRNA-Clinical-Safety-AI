from pathlib import Path
import csv
import time

import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "outputs" / "text_chunks.txt"
DB_DIR = PROJECT_DIR / "outputs" / "chroma_store"
EVALUATION_FILE = PROJECT_DIR / "outputs" / "rag_evaluation.csv"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global objects initialized in main()
model = None
collection = None


def main():
    global model, collection

    print("Starting RAG pipeline...")

    # Check whether the chunks file exists
    if not CHUNKS_FILE.exists():
        print(f"ERROR: Chunks file not found: {CHUNKS_FILE}")
        print("Run text_chunking.py first.")
        return

    # Read and split saved chunks
    saved_text = CHUNKS_FILE.read_text(encoding="utf-8")

    chunks = [
        chunk.strip()
        for chunk in saved_text.split("--- CHUNK ---")
        if chunk.strip()
    ]

    if not chunks:
        print("ERROR: No text chunks found.")
        return

    print(f"Total chunks loaded: {len(chunks)}")

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create persistent ChromaDB
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="mrna_safety_guidelines",
        metadata={"hnsw:space": "cosine"}
    )

    # Index chunks only if the collection is empty
    if collection.count() == 0:
        print("Creating document embeddings...")

        embeddings = model.encode(chunks).tolist()

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

        print("Documents indexed successfully!")

    else:
        print(
            f"Existing ChromaDB collection found "
            f"with {collection.count()} documents."
        )

    print("RAG database is ready.")


def retrieve_documents(question, top_k=3):
    if model is None or collection is None:
        raise RuntimeError("Run main() before retrieving documents.")

    query_embedding = model.encode([question]).tolist()

    start_time = time.perf_counter()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "distances"]
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"Retrieval time: {elapsed_time:.4f} seconds")
    print("\nRetrieved safety information:")

    documents = results.get("documents", [[]])[0]

    for i, document in enumerate(documents, start=1):
        print(f"\nResult {i}:")
        print(document)

    return results


if __name__ == "__main__":

    # Initialize the RAG database
    main()

    if model is not None and collection is not None:

        test_questions = [
            "What are the storage requirements for mRNA products?",
            "What are the recommended monitoring requirements after administration?",
            "What are the contraindications for mRNA products?",
            "How should adverse events be documented?",
            "What precautions apply to patients with prior allergic reactions?",
            "What is the population of Mars?"
        ]

        evaluation_results = []

        # Test retrieval for every question
        for question in test_questions:

            print("\n" + "=" * 60)
            print("QUESTION:", question)

            start_time = time.perf_counter()

            results = retrieve_documents(question)

            latency = time.perf_counter() - start_time

            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            results_found = len(documents)

            # Flag potentially irrelevant retrievals
            if not distances:
                status = "No results"
            elif min(distances) > 1.0:
                status = "Potential retrieval failure"
            else:
                status = "Results retrieved - review relevance"

            top_distance = min(distances) if distances else None

            evaluation_results.append({
                "question": question,
                "retrieval_latency_seconds": round(latency, 4),
                "results_found": results_found,
                "top_distance": (
                    round(top_distance, 4)
                    if top_distance is not None else ""
                ),
                "status": status
            })

        # Save evaluation results
        with open(
            EVALUATION_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "question",
                    "retrieval_latency_seconds",
                    "results_found",
                    "top_distance",
                    "status"
                ]
            )

            writer.writeheader()
            writer.writerows(evaluation_results)

        print("\nRAG evaluation saved successfully!")
        print(f"File location: {EVALUATION_FILE}")
        print("RAG evaluation completed successfully!")

