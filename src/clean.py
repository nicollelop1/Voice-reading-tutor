import re
from collections import Counter

MIN_WORDS = 45          
MAX_START_SHARE = 0.5   
END_SHARE = 0.7        

# Rastros de ficha editorial. STRONG: una sola aparición descarta el bloque.
STRONG = re.compile(
    r"isbn|creative\s+commons|copyright|©|derechos\s+reservados|todos\s+los\s+derechos|"
    r"all\s+rights\s+reserved|https?://|www\.|recurso\s+electr[oó]nico|dep[oó]sito\s+legal|"
    r"project\s+gutenberg", re.I)
# WEAK: hacen falta dos en el mismo bloque.
WEAK = re.compile(
    r"licencia|editorial|imprenta|edici[oó]n|biblioteca|impreso|publisher|published|"
    r"translated|traducci[oó]n|diagramaci[oó]n|dise[nñ]o\s+de|\bpdf\b|\bkb\b|\bmb\b|"
    r"primera\s+ed|\bed\.|cuentos?\s+(?:franceses|infantiles)|\bdgb\b|fuente:", re.I)

CHAPTER_ONE = re.compile(
    r"^\s*(?:cap[ií]tulo|cap\.|chapter|chapitre)\s*(?:1|i|uno|one|primero|primer|premier|un)\b", re.I)
CHAPTER_ANY = re.compile(
    r"^\s*(?:cap[ií]tulo|cap\.|chapter|chapitre)\s*(?:\d+|[ivxlcdm]+|[a-záéíóúñ]+)\b", re.I)
CHAPTER_BARE = re.compile(
    r"^\s*(?:cap[ií]tulo|cap\.|chapter|chapitre)\s*(?:\d+|[ivxlcdm]+|[a-záéíóúñ]+)\.?\s*$", re.I)
ROMAN_ONLY = re.compile(r"^[IVXLCDM]{1,7}\.?$")
DIGITS_ONLY = re.compile(r"^\d{1,4}$")
PAGE_LABEL = re.compile(r"^(?:p[aá]g(?:ina)?\.?|page)\s*\d+(?:\s*(?:de|of|/)\s*\d+)?$|^[-–—]\s*\d+\s*[-–—]$", re.I)
END_MARK = re.compile(
    r"^(?:fin|the\s+end|fin\s+del\s+(?:cuento|libro)|[íi]ndice|contenido|tabla\s+de\s+contenidos?|"
    r"colof[oó]n|sobre\s+el\s+autor|acerca\s+del\s+autor|about\s+the\s+author|agradecimientos|"
    r"bibliograf[ií]a|otros\s+t[ií]tulos|cat[aá]logo)\.?$", re.I)
SENTENCE_END = re.compile(r"[.!?…]")


def _words(s: str) -> int:
    return len(s.split())


def _prose_like(line: str) -> bool:
    """¿Parece una línea de narración (y no un título, un dato o una cifra)?"""
    if _words(line) < 5 or line.isupper():
        return False
    if sum(c.isalpha() for c in line) / len(line) < 0.75:
        return False
    return True


def _is_heading(line: str) -> bool:
    return bool(CHAPTER_ANY.match(line) or ROMAN_ONLY.match(line) or DIGITS_ONLY.match(line))


def _window_ok(lines: list[str], j: int) -> bool:
    """¿Desde la línea j hay un bloque de prosa narrativa sin rastro editorial?"""
    words = n = prose = sentences = weak = 0
    if not lines[j] or not _prose_like(lines[j]):
        return False
    k = j
    while k < len(lines) and words < MIN_WORDS and n < 16:
        line = lines[k]
        k += 1
        if not line:
            continue
        n += 1
        if STRONG.search(line):
            return False
        weak += len(WEAK.findall(line))
        if _prose_like(line):
            prose += 1
            words += _words(line)
            sentences += len(SENTENCE_END.findall(line))
    return words >= MIN_WORDS and prose / n >= 0.8 and weak < 2 and sentences >= 2


def _prose_after_heading(lines: list[str], i: int, max_skip: int = 3) -> int | None:
    """Tras un encabezado, salta hasta 3 líneas cortas (el título del capítulo) y
    exige prosa. Si antes aparece otro encabezado es un índice, no el inicio."""
    skipped, j = 0, i
    while j < len(lines) and skipped <= max_skip:
        line = lines[j]
        if not line:
            j += 1
            continue
        if _window_ok(lines, j):
            return j
        if _is_heading(line):
            return None
        skipped += 1
        j += 1
    return None


def _find_start(lines: list[str], total_words: int) -> tuple[int, str]:
    """Devuelve (índice de la primera línea de la historia, motivo)."""
    def too_deep(idx: int) -> bool:
        return sum(_words(l) for l in lines[:idx]) > MAX_START_SHARE * total_words

    stripped = [l.strip() for l in lines]
    has_two = any(re.fullmatch(r"II\.?", l) for l in stripped)
    has_2 = any(l == "2" for l in stripped)

    for i, line in enumerate(stripped):
        is_candidate = bool(CHAPTER_ONE.match(line)) or \
            (re.fullmatch(r"I\.?", line) and has_two) or (line == "1" and has_2)
        if not is_candidate:
            continue
        start = _prose_after_heading(lines, i + 1)
        if start is not None and not too_deep(start):
            return start, "capítulo"

    for i in range(len(lines)):
        if _window_ok(lines, i):
            return (i, "prosa") if not too_deep(i) else (0, "ninguno")
    return 0, "ninguno"


def _strip_repeated(lines: list[str]) -> tuple[list[str], int]:
    """Quita encabezados y pies de página: líneas cortas sin punto final que se repiten."""
    def key(l: str) -> str:
        return re.sub(r"[\W_]+", "", l.lower())     

    counts = Counter(key(l) for l in lines
                     if l and len(l) <= 80 and key(l)
                     and not _is_heading(l) and not SENTENCE_END.search(l[-1:])
                     and l[0] not in "—-–«“\"'¿¡")
    repeated = {k for k, c in counts.items() if c >= 3}
    kept = [l for l in lines if not (l and key(l) in repeated and len(l) <= 80
                                     and not _is_heading(l) and l[0] not in "—-–«“\"'¿¡"
                                     and not SENTENCE_END.search(l[-1:]))]
    removed = len(lines) - len(kept)
    if removed > 0.6 * max(1, sum(1 for l in lines if l)):    # demasiado: no son encabezados
        return lines, 0
    return kept, removed


def clean_book_text(text: str) -> tuple[str, dict]:
    """Devuelve (texto limpio, info). `info` explica qué se saltó, para mostrarlo."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "").replace("\xa0", " ")
    text = text.replace("\f", "\n")
    m = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", text, re.I)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text, re.I)
    if m:
        text = text[:m.start()]
    text = re.sub(r"(\w)-\n(?=[a-záéíóúñ])", r"\1", text)      # palabras partidas al final de línea

    original_words = _words(text)
    info = {"start_reason": "ninguno", "skipped_words": 0, "trimmed_end_words": 0,
            "repeated_lines_removed": 0, "preview": text.strip()[:120]}
    if original_words < MIN_WORDS:                 # texto muy corto: no hay nada que recortar
        return text.strip(), info

    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n")]
    lines, info["repeated_lines_removed"] = _strip_repeated(lines)

    def finish(ls: list[str]) -> str:
        ls = [l for l in ls if not (DIGITS_ONLY.match(l) or PAGE_LABEL.match(l) or CHAPTER_BARE.match(l)
                                    or (has_roman_seq and ROMAN_ONLY.match(l))
                                    or (len(l) <= 160 and STRONG.search(l)))]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(ls)).strip()

    stripped = [l.strip() for l in lines]
    has_roman_seq = any(re.fullmatch(r"I\.?", l) for l in stripped) and \
        any(re.fullmatch(r"II\.?", l) for l in stripped)
    safe = finish(lines)                           # versión mínima por si el recorte se pasa

    total_words = sum(_words(l) for l in lines)
    start, reason = _find_start(lines, total_words)
    body = lines[start:]

    body_words = sum(_words(l) for l in body)
    seen = 0
    for idx, l in enumerate(body):
        seen += _words(l)
        if seen >= END_SHARE * body_words and END_MARK.match(l):
            cut = idx                          # se descarta el marcador ("FIN", "Índice"…) y lo que sigue
            info["trimmed_end_words"] = sum(_words(x) for x in body[cut:])
            body = body[:cut]
            break

    cleaned = finish(body)
    if _words(cleaned) < MIN_WORDS or _words(cleaned) < 0.15 * original_words:
        info["start_reason"] = "ninguno (respaldo)"
        info["preview"] = safe[:120]
        return safe, info

    info["start_reason"] = reason
    info["skipped_words"] = sum(_words(l) for l in lines[:start])
    info["preview"] = cleaned[:120]
    return cleaned, info
