import re
import unicodedata
from difflib import SequenceMatcher



# ---------- números: "6" y "seis" deben contar igual ----------
# Chrome transcribe lo que se dice como cifras ("6 largos meses") aunque el libro diga "seis".
_UN = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once",
       "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte",
       "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete",
       "veintiocho", "veintinueve"]
_DEC = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_CEN = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos",
        6: "seiscientos", 7: "setecientos", 8: "ochocientos", 9: "novecientos"}
_ORD = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto", 6: "sexto", 7: "séptimo",
        8: "octavo", 9: "noveno", 10: "décimo"}
# Sustantivos masculinos terminados en -a: "1 día" es "un día", no "una día"
_MASC_A = {"dia", "mapa", "problema", "tema", "poema", "sistema", "idioma", "clima", "programa", "planeta",
           "cometa", "pirata", "atleta", "turista", "artista", "poeta", "dalmata", "pijama", "cambista",
           "colega", "guia", "lema", "drama", "diploma", "esquema", "fantasma", "panorama", "aroma"}
_NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?:\s?(%)|\.?\s?([ºª°]))?(?!\w)")


def _below_1000(n: int, fem: bool) -> str:
    if n == 100:
        return "cien"
    parts = []
    h, r = divmod(n, 100)
    if h:
        w = _CEN[h]
        parts.append(w[:-2] + "as" if fem and h != 1 else w)
    if r:
        if r < 30:
            parts.append(_UN[r])
        else:
            t, u = divmod(r, 10)
            parts.append(_DEC[t] + (" y " + _UN[u] if u else ""))
    return " ".join(parts)


def _apocope(w: str) -> str:
    if w.endswith("veintiuno"):
        return w[:-9] + "veintiún"
    return w[:-3] + "un" if w.endswith("uno") else w


def spanish_number(n: int, fem: bool = False, noun_follows: bool = False) -> str:
    """12 -> 'doce', 1944 -> 'mil novecientos cuarenta y cuatro', 1 + sustantivo -> 'un'."""
    if n == 0:
        return "cero"
    if n >= 10 ** 12:
        return " ".join(_UN[int(d)] for d in str(n))
    millions, rest = divmod(n, 10 ** 6)
    thousands, units = divmod(rest, 1000)
    parts = []
    if millions:
        parts.append("un millón" if millions == 1 else _apocope(_below_1000(millions, False)) + " millones")
    if thousands:
        parts.append("mil" if thousands == 1 else _apocope(_below_1000(thousands, fem)) + " mil")
    if units:
        w = _below_1000(units, fem)
        if w.endswith("uno"):
            if fem:
                w = w[:-3] + "una"
            elif noun_follows:
                w = _apocope(w)
        parts.append(w)
    return " ".join(parts)


def numbers_to_words(text: str) -> str:
    """Cambia las cifras por su lectura en español: '6 largos meses' -> 'seis largos meses'."""
    def repl(m: re.Match) -> str:
        whole, dec, percent, ordinal = m.group(1), m.group(2), m.group(3), m.group(4)
        n = int(whole.replace(".", ""))
        nxt = re.match(r"\s*([^\W\d_]+)", m.string[m.end():])
        noun = unicodedata.normalize("NFD", nxt.group(1).lower()) if nxt else ""
        noun = "".join(c for c in noun if unicodedata.category(c) != "Mn")
        fem = bool(noun and re.search(r"as?$", noun) and noun.rstrip("s") not in _MASC_A)
        if ordinal and n in _ORD:
            w = _ORD[n]
            return " " + (w[:-1] + "a" if ordinal == "ª" else w) + " "
        out = spanish_number(n, fem, bool(nxt))
        if dec:
            zeros = len(dec) - len(dec.lstrip("0"))
            tail = " ".join(["cero"] * zeros + ([spanish_number(int(dec))] if dec.strip("0") else []))
            out = spanish_number(n) + " coma " + tail
        if percent:
            out += " por ciento"
        return " " + out + " "
    return _NUM_RE.sub(repl, text)


def normalize(text: str) -> list[str]:
    """Minúsculas, números en palabras, sin tildes ni puntuación; devuelve lista de palabras."""
    text = unicodedata.normalize("NFD", numbers_to_words(text.lower()))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def evaluate_reading(expected: str, transcript: str, min_accuracy: float = 0.9) -> dict:
    """Compara el texto esperado con lo que el usuario leyó.

    Devuelve: correct (bool), accuracy (0-1), wrong_words (palabra esperada
    que fue reemplazada), omitted_words (palabras que faltaron) y
    extra_words (palabras que sobraron).
    """
    exp = normalize(expected)
    got = normalize(transcript)

    wrong, omitted, extra = [], [], []
    matcher = SequenceMatcher(None, exp, got)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            wrong.extend(exp[i1:i2])
        elif tag == "delete":
            omitted.extend(exp[i1:i2])
        elif tag == "insert":
            extra.extend(got[j1:j2])

    accuracy = matcher.ratio()
    correct = accuracy >= min_accuracy and not wrong and not omitted
    return {
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "wrong_words": wrong,
        "omitted_words": omitted,
        "extra_words": extra,
    }