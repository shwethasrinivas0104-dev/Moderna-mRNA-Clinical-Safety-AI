from pathlib import Path

# Find the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input file containing extracted safety guidelines
INPUT_FILE = PROJECT_DIR / "outputs" / "safety_guidelines_extracted.txt"

# Output file for text chunks
OUTPUT_FILE = PROJECT_DIR / "outputs" / "text_chunks.txt"

# Chunk settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    text = INPUT_FILE.read_text(encoding="utf-8")
    chunks = split_text(text)

    OUTPUT_FILE.write_text(
        "\n\n--- CHUNK ---\n\n".join(chunks),
        encoding="utf-8"
    )

    print("Text chunking completed successfully!")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Overlap: {CHUNK_OVERLAP} words")
    print(f"Chunks saved to: {OUTPUT_FILE}")
    print("\nFirst chunk preview:")
    print(chunks[0][:500] if chunks else "No text found.")


if __name__ == "__main__":
    main()

