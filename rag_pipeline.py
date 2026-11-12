from pathlib import Path
from docx import Document

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
GUIDE_PATH = PROJECT_DIR / "data" / "mrna_safety_guidelines.docx"


def main():
    # Check whether the safety guide exists
    if not GUIDE_PATH.exists():
        print("ERROR: Safety guidelines document was not found.")
        print(f"Expected location: {GUIDE_PATH}")
        return

    # Read the Word document
    document = Document(GUIDE_PATH)

    # Extract non-empty paragraphs
    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    # Combine the paragraphs into one text
    guide_text = "\n\n".join(paragraphs)

    # Create the outputs folder
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the extracted text
    output_file = output_dir / "safety_guidelines_extracted.txt"
    output_file.write_text(guide_text, encoding="utf-8")

    print("Safety guidelines document read successfully!")
    print(f"Paragraphs extracted: {len(paragraphs)}")
    print(f"Text saved to: {output_file}")

    print("\nFirst part of the document:")
    print(guide_text[:1000])

    print("\nEducational project only—not for clinical decisions.")


if __name__ == "__main__":
    main()

