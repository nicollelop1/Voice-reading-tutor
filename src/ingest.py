import re
from pathlib import Path

from src.clean import clean_book_text


def load_raw_text(path: str) -> str:
    """Extrae el texto plano de un archivo PDF, EPUB o TXT, tal cual viene (con portada, licencias…)."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".txt":
        return p.read_text(encoding="utf-8")

    if ext == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(p) as doc:
            return "\n".join(page.get_text() for page in doc)

    if ext == ".epub":
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
        book = epub.read_epub(str(p))
        parts = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            parts.append(soup.get_text("\n"))
        return "\n".join(parts)

    raise ValueError(f"Formato no soportado: {ext}")


def load_book_text(path: str) -> tuple[str, dict]:
    """Texto listo para leer: sin portada, licencias ni encabezados, empezando donde empieza la historia.
    Devuelve (texto, info) con lo que se saltó (info["skipped_words"], info["preview"]…)."""
    return clean_book_text(load_raw_text(path))


def load_text(path: str) -> str:
    """Texto limpio del libro (compatibilidad con el código existente)."""
    return load_book_text(path)[0]


def split_chunks(text: str, max_words: int = 40) -> list[str]:
    """Divide el texto en fragmentos que terminan en fin de oración."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks