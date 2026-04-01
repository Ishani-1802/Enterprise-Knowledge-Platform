import pdfplumber
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs_from_directory(directory_path: str) -> str:
    """
    Loads all PDFs using pdfplumber for better layout preservation.
    """
    all_text = ""
    pdf_dir = Path(directory_path)

    for pdf_file in pdf_dir.glob("*.pdf"):
        # SWITCH: Using pdfplumber instead of pypdf
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Adding extra newlines helps the semantic splitter find boundaries
                    all_text += text + "\n\n" 

    return all_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    SWITCH: Using RecursiveCharacterTextSplitter.
    It splits by Paragraph -> Sentence -> Word to keep meaning intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""] # Order of priority
    )
    
    # split_text returns a list of strings
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # Ensure you have the 'data/raw_docs' folder
    text = load_pdfs_from_directory("data/raw_docs")
    chunks = chunk_text(text)

    print(f"Total semantic chunks created: {len(chunks)}")
    if chunks:
        print("\nFirst chunk preview (Note how it ends at a clean boundary):\n")
        print(chunks[0])