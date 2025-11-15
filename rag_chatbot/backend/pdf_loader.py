"""PDF Loader Module"""

from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from utils import clean_text

def extract_text_from_pdfs(folder_path: str) -> dict:
    """Extract text from PDFs"""
    folder = Path(folder_path)
    processed_folder = folder.parent / "processed_docs"
    processed_folder.mkdir(exist_ok=True)
    
    processed_count = 0
    total_chars = 0
    
    if not folder.exists():
        return {
            "status": "error",
            "message": "Documents folder not found",
            "processed_count": 0,
            "total_characters_extracted": 0
        }
    
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        return {
            "status": "success",
            "message": "No PDFs found",
            "processed_count": 0,
            "total_characters_extracted": 0
        }
    
    for pdf_file in pdf_files:
        try:
            print(f"  Processing: {pdf_file.name}")
            reader = PdfReader(str(pdf_file))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            cleaned = clean_text(text)
            output_file = processed_folder / f"{pdf_file.stem}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            total_chars += len(cleaned)
            processed_count += 1
            print(f"     OK: {len(cleaned):,} chars")
            
        except Exception as e:
            print(f"     ERROR: {e}")
    
    return {
        "status": "success",
        "message": f"Processed {processed_count} files",
        "processed_count": processed_count,
        "total_characters_extracted": total_chars
    }

def load_processed_documents(folder_path: str) -> dict:
    """Load processed text files"""
    folder = Path(folder_path)
    documents = {}
    
    if not folder.exists():
        return documents
    
    for txt_file in folder.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                documents[txt_file.name] = f.read()
                print(f"  OK Loaded: {txt_file.name}")
        except Exception as e:
            print(f"  ERROR loading {txt_file.name}: {e}")
    
    return documents
