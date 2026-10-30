"""Diagnóstico de una lectura, pensado para decírselo a un niño.

En vez de "fallaste en: tarde, obra", compara lo que debía leer con lo que se le
entendió y explica cada problema: qué dijo, qué decía el texto, en qué letra o
sonido está la diferencia y cómo se separa la palabra en sílabas. También
reconoce cuando se saltó un tramo, cuando agregó palabras, cuánto acertó y si
mejoró respecto al intento anterior.
"""
import re
from difflib import SequenceMatcher

from src.evaluate import normalize

NEAR = 0.75          # a partir de aquí la palabra "casi" estaba bien
RUN = 4              # palabras seguidas omitidas = "tramo saltado"

# ---------- sílabas en español (reglas estándar) ----------
_VOW = "aeiouáéíóúü"
_STRONG = "aeoáéó"
_INSEP = {"pr", "pl", "br", "bl", "fr", "fl", "gr", "gl", "cr", "cl", "tr", "dr", "ch", "ll", "rr"}


def syllables(word: str) -> list[str]:
    """Separa una palabra en sílabas: 'tarde' -> ['tar', 'de']."""
    w = word.lower()
    if not w or not any(c in _VOW for c in w):
        return [w] if w else []

    is_v = []
    for k, c in enumerate(w):                       # la "u" de que/qui/gue/gui no es vocal
        silent_u = c == "u" and k > 0 and w[k - 1] in "qg" and k + 1 < len(w) and w[k + 1] in "eiéí"
        is_v.append(c in _VOW and not silent_u)

    nuclei, i = [], 0                               # grupos de vocales (diptongos juntos, hiatos separados)
    while i < len(w):
        if not is_v[i]:
            i += 1
            continue
        j = i + 1
        while j < len(w) and is_v[j]:
            a, b = w[j - 1], w[j]
            if (a in _STRONG and b in _STRONG) or (a in "íú" and b in _VOW) or (b in "íú" and a in _VOW):
                break                               # hiato: dos sílabas
            j += 1
        nuclei.append((i, j))
        i = j

    cuts = []                                       # posiciones donde empieza una sílaba nueva
    for (_, e1), (s2, _) in zip(nuclei, nuclei[1:]):
        cons = w[e1:s2]
        if len(cons) <= 1:
            cuts.append(e1)
        elif len(cons) >= 2 and cons[-2:] in _INSEP:
            cuts.append(s2 - 2)
        else:
            cuts.append(s2 - 1)
    # una consonante sola va con la vocal siguiente; sin consonante, se corta entre vocales
    out, prev = [], 0
    for c in cuts:
        out.append(w[prev:c])
        prev = c
    out.append(w[prev:])
    return [s for s in out if s]


def _dash(word: str) -> str:
    return "-".join(syllables(word))


def _comma(word: str) -> str:
    return ", ".join(syllables(word))


# ---------- alineación palabra por palabra ----------
def _char_tip(expected: str, said: str) -> str:
    """Qué letra/sonido cambia entre lo esperado y lo dicho."""
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, expected, said).get_opcodes():
        if tag == "replace":
            return f"en lugar de «{expected[i1:i2]}» dijiste «{said[j1:j2]}»"
        if tag == "delete":
            return f"te faltó el sonido «{expected[i1:i2]}»"
        if tag == "insert":
            return f"agregaste «{said[j1:j2]}»"
    return ""


def align(expected: str, transcript: str) -> dict:
    """Una marca por cada palabra *visible* del texto (la misma partición que la pantalla),
    aunque una palabra (p. ej. una cifra) se lea como varias o un guion no se lea."""
    raw = expected.split()
    flat: list[tuple[int, str]] = []                 # (índice de la palabra visible, palabra hablada)
    for i, t in enumerate(raw):
        flat += [(i, w) for w in normalize(t)]
    exp_norm = [w for _, w in flat]
    got = normalize(transcript)
    status = ["ok"] * len(flat)
    said: list[str | None] = [None] * len(flat)
    extras: list[str] = []

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, exp_norm, got, autojunk=False).get_opcodes():
        if tag == "delete":
            for i in range(i1, i2):
                status[i] = "omitted"
        elif tag == "insert":
            extras += got[j1:j2]
        elif tag == "replace":
            if i2 - i1 == j2 - j1:                              # mismo número de palabras: una por una
                for k in range(i2 - i1):
                    status[i1 + k], said[i1 + k] = "wrong", got[j1 + k]
            else:
                pool = list(range(j1, j2))
                for i in range(i1, i2):
                    best, best_r = None, 0.0
                    for j in pool:
                        r = SequenceMatcher(None, exp_norm[i], got[j]).ratio()
                        if r > best_r:
                            best, best_r = j, r
                    if best is not None and best_r >= 0.5:
                        pool.remove(best)
                        status[i], said[i] = "wrong", got[best]
                    else:
                        status[i] = "omitted"
                extras += [got[j] for j in pool]

    marks = []
    for i, t in enumerate(raw):
        idx = [k for k, (ti, _) in enumerate(flat) if ti == i]
        shown = re.sub(r"^[^\w]+|[^\w]+$", "", t) or t
        if not idx:
            marks.append({"w": shown, "s": "na", "said": None, "norm": ""})   # guion, signo suelto…
            continue
        norm = " ".join(flat[k][1] for k in idx)
        wrong = [k for k in idx if status[k] == "wrong"]
        if wrong:
            heard = " ".join(said[k] for k in wrong)
            marks.append({"w": shown, "s": "wrong", "said": heard, "norm": norm,
                          "ratio": round(SequenceMatcher(None, " ".join(flat[k][1] for k in wrong), heard).ratio(), 2)})
        elif any(status[k] == "omitted" for k in idx):
            marks.append({"w": shown, "s": "omitted", "said": None, "norm": norm})
        else:
            marks.append({"w": shown, "s": "ok", "said": None, "norm": norm})

    runs, i = [], 0                                             # tramos omitidos seguidos
    real = [k for k, m in enumerate(marks) if m["s"] != "na"]
    while i < len(real):
        if marks[real[i]]["s"] == "omitted":
            j = i
            while j < len(real) and marks[real[j]]["s"] == "omitted":
                j += 1
            if j - i >= RUN:
                runs.append((real[i], real[j - 1] + 1))
            i = j
        else:
            i += 1
    return {"marks": marks, "extras": extras, "runs": runs}


# ---------- diagnóstico completo ----------
def _issue_for(mark: dict) -> dict:
    w = mark["w"]
    syl = _comma(w) if len(syllables(w)) >= 2 else ""
    if mark["s"] == "wrong":
        said = mark["said"]
        tip = _char_tip(mark["norm"], said) if mark["ratio"] >= NEAR else ""
        text = f"dijiste «{said}» en vez de «{w}»"
        screen = text + (f" ({tip})" if tip else "")
        return {"kind": "wrong", "word": w, "said": said, "syl": _dash(w), "text": text,
                "screen": screen, "speak_help": f"Se lee: {syl}." if syl else ""}
    return {"kind": "omitted", "word": w, "syl": _dash(w), "text": f"te saltaste «{w}»",
            "screen": f"te saltaste «{w}»", "speak_help": f"Esa palabra es: {syl}." if syl else ""}


def diagnose(expected: str, transcript: str, attempt: int = 1, prev: dict | None = None) -> dict:
    """prev: {"accuracy": float, "bad": [palabras normalizadas que fallaron]} del intento anterior."""
    al = align(expected, transcript)
    marks, extras, runs = al["marks"], al["extras"], al["runs"]
    total = sum(1 for m in marks if m["s"] != "na")
    good = sum(1 for m in marks if m["s"] == "ok")
    in_run = {k for a, b in runs for k in range(a, b)}

    issues = [_issue_for(m) for k, m in enumerate(marks) if m["s"] == "wrong"]
    for a, b in runs:
        tail = all(m["s"] == "na" for m in marks[b:])
        words = " ".join(m["w"] for m in marks[a:min(b, a + 4)])
        txt = (f"no llegaste hasta el final: faltó desde «{words}…»" if tail
               else f"te saltaste un pedazo del texto: «{words}…»")
        issues.append({"kind": "run", "text": txt, "screen": txt, "speak_help": "", "word": words})
    issues += [_issue_for(m) for k, m in enumerate(marks) if m["s"] == "omitted" and k not in in_run]
    if extras:
        shown = ", ".join(f"«{e}»" for e in extras[:3])
        issues.append({"kind": "extra", "text": f"agregaste {shown}, que no está en el texto",
                       "screen": f"agregaste {shown}, que no está en el texto", "speak_help": "", "word": ""})

    accuracy = good / total if total else 0.0
    bad_now = [m["norm"] for m in marks if m["s"] not in ("ok", "na")]
    limit = 2 if attempt == 1 else 4

    # ---- lo que se dice en voz alta ----
    sp = []
    if attempt == 1:
        if accuracy >= 0.7:
            sp.append(f"¡Vas muy bien! Leíste bien {good} de {total} palabras.")
        elif accuracy >= 0.4:
            sp.append(f"Leíste bien {good} de {total} palabras. ¡Sigue así!")
        else:
            sp.append("Esta parte es difícil, pero no pasa nada. Vamos con calma.")
    else:
        if prev and prev.get("accuracy") is not None and accuracy > prev["accuracy"] + 0.05:
            sp.append(f"¡Mejoraste! Ahora leíste bien {good} de {total} palabras.")
        else:
            sp.append(f"Esta vez leíste bien {good} de {total} palabras.")
        if prev and prev.get("bad"):
            fixed = [w for w in prev["bad"] if w not in bad_now and len(w) > 3][:2]
            if fixed:
                sp.append("Ya dices bien " + " y ".join(f"«{w}»" for w in fixed) + ".")

    heard_short = " ".join(transcript.split()[:8])
    if accuracy < 0.3:                               # leyó algo distinto: no tiene sentido listar palabras sueltas
        sp = [f"Creo que leíste algo distinto a lo que está escrito: te escuché «{heard_short}». "
              "No te preocupes. Lee justo lo que dice el texto, despacito y con el dedo debajo de cada palabra."
              if attempt == 1 else
              "Esta parte se te complicó, ¡y está bien! Vamos a verla juntos."]
    elif len(issues) > 5 and attempt == 1:           # demasiados errores: consejo general, no una lista
        sp.append("Hay varias palabras para cuidar. Léelo más despacito, palabra por palabra, "
                  "y puedes seguir el texto con el dedo.")
        hard = [i for i in issues if i["kind"] in ("wrong", "omitted") and len(i["word"]) > 3][:2]
        if hard:
            sp.append("Empieza fijándote en " + " y ".join(f"«{i['word']}»" for i in hard) + ".")
    else:
        for n, it in enumerate(issues[:limit]):
            lead = "Fíjate: " if n == 0 else "Y también "
            sp.append(f"{lead}{it['text']}." + (f" {it['speak_help']}" if it["speak_help"] else ""))
        rest = len(issues) - limit
        if rest > 0 and attempt == 1:
            sp.append(f"Hay {rest} detalle{'s' if rest != 1 else ''} más: míralo{'s' if rest != 1 else ''} en la pantalla.")
    if any(i["kind"] == "run" for i in issues) and attempt == 1 and len(issues) <= 5:
        sp.append("Sigue el texto con el dedo para no perder la línea.")

    return {
        "accuracy": round(accuracy, 2), "good": good, "total": total,
        "heard": transcript.strip(),
        "marks": [{"w": m["w"], "s": m["s"], "said": m["said"]} for m in marks],
        "issues": [{"kind": i["kind"], "text": i["screen"], "syllables": i.get("syl", "")} for i in issues],
        "extras": extras,
        "spoken": " ".join(sp),
        "bad": bad_now,
    }
