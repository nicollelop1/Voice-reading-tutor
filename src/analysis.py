"""Análisis pedagógico de una sesión de lectura.

Convierte el registro de intentos (qué debía leer el niño y qué se le entendió)
en un texto narrativo para el vault de Obsidian: qué pronunció bien, dónde tuvo
dificultad, qué patrones se repiten y qué debe trabajar el tutor la próxima vez.
No usa red ni IA: es determinista, así que siempre funciona y se puede probar.
"""
import re
from difflib import SequenceMatcher

from src.diagnose import syllables, _char_tip, align as diagnose_align
from src.evaluate import normalize

NEAR_MISS = 0.75      
RUN = 4              
LONG_WORD = 7         

CLUSTERS = ("pr", "br", "tr", "dr", "cr", "gr", "fr", "pl", "bl", "cl", "gl", "fl")


# ---------- alineación palabra por palabra (misma que el diagnóstico en vivo) ----------
def _tokens(text: str) -> list[tuple[str, str]]:
    """[(palabra como se ve en el libro, su primera palabra normalizada)] sin signos sueltos."""
    out = []
    for tok in text.split():
        norm = normalize(tok)
        if norm:
            out.append((re.sub(r"^[^\w]+|[^\w]+$", "", tok), norm[0]))
    return out


def align(expected: str, transcript: str) -> dict:
    """Sustituciones, omisiones, tramos saltados y añadidos de una lectura."""
    al = diagnose_align(expected, transcript)
    marks = al["marks"]
    in_run = {k for a, b in al["runs"] for k in range(a, b)}
    return {
        "subs": [{"word": m["w"], "said": m["said"], "ratio": m["ratio"]} for m in marks if m["s"] == "wrong"],
        "omitted": [m["w"] for k, m in enumerate(marks) if m["s"] == "omitted" and k not in in_run],
        "runs": [[marks[k]["w"] for k in range(a, b) if marks[k]["s"] != "na"] for a, b in al["runs"]],
        "extra": al["extras"],
    }


def _context(expected: str, word: str, span: int = 3) -> str:
    """Trozo de la frase alrededor de la palabra, para dar contexto."""
    toks = expected.split()
    target = normalize(word)[0] if normalize(word) else word
    for i, t in enumerate(toks):
        n = normalize(t)
        if n and n[0] == target:
            return " ".join(toks[max(0, i - span): i + span + 1])
    return expected


# ---------- rasgos fonéticos de las palabras que falló ----------
def _traits(word: str) -> list[str]:
    w = (normalize(word) or [""])[0]
    traits = []
    if w.startswith("r") or "rr" in w:
        traits.append("la erre fuerte (r- inicial o rr)")
    if any(c in w for c in CLUSTERS):
        traits.append("los grupos de consonantes (tr, br, pl, gr…)")
    if "ll" in w or "ch" in w:
        traits.append("los sonidos «ll» y «ch»")
    if len(w) >= LONG_WORD:
        traits.append("las palabras largas")
    if 0 < len(w) <= 3:
        traits.append("las palabras cortas (artículos y conectores)")
    return traits


def _q(words: list[str], limit: int = 6) -> str:
    uniq = list(dict.fromkeys(words))[:limit]
    return ", ".join(f"«{w}»" for w in uniq)


def _short(text: str, n: int = 140) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def _pct(part: int, total: int) -> int:
    return round(100 * part / total) if total else 0


# ---------- análisis completo ----------
def analyze_session(name: str, log: list[dict], stats: dict, silences: int = 0) -> dict:
    """Devuelve:
    markdown   -> secciones de la nota de sesión (resumen, aciertos, dificultades, plan)
    summary    -> una línea para el perfil del usuario y el libro
    focus      -> palabras a vigilar la próxima vez
    word_notes -> {palabra: qué pasó} para la nota de cada palabra
    """
    name = name[:1].upper() + name[1:]
    done = [e for e in log if e.get("attempts")]
    total = len(done)
    first = sum(1 for e in done if e["outcome"] == "first_try")
    second = sum(1 for e in done if e["outcome"] == "second_try")
    failed = sum(1 for e in done if e["outcome"] == "failed")

    word_notes: dict[str, str] = {}
    problems: list[dict] = []       # un registro por fragmento con dificultad
    fixed_on_retry: list[str] = []
    well_read_hard: list[str] = []
    kinds = {"sub": 0, "omit": 0, "extra": 0}
    near, far = [], []
    trait_words: dict[str, list[str]] = {}
    problem_words: list[str] = []
    unresolved: list[str] = []      # siguen mal en el último intento (las más importantes)

    for e in done:
        exp = e["expected"]
        per_attempt = [align(exp, a["transcript"]) for a in e["attempts"]]
        first_al = per_attempt[0]
        said_ok_first = ({s["word"] for s in first_al["subs"]} | set(first_al["omitted"])
                         | {w for r in first_al["runs"] for w in r})

        # palabras largas que sí leyó bien aunque el fragmento tuviera fallos
        if said_ok_first:
            for w, n in _tokens(exp):
                if len(n) >= LONG_WORD and w not in said_ok_first:
                    well_read_hard.append(w)
        elif e["outcome"] == "first_try":
            well_read_hard += [w for w, n in _tokens(exp) if len(n) >= LONG_WORD]

        if e["outcome"] == "first_try":
            continue

        if len(per_attempt) > 1:
            last = per_attempt[-1]
            still = ({s["word"] for s in last["subs"]} | set(last["omitted"])
                     | {w for r in last["runs"] for w in r})
            skipped = {w for r in first_al["runs"] for w in r}
            fixed_on_retry += [w for w in said_ok_first if w not in still and w not in skipped]

        final = per_attempt[-1]
        entry_words: list[str] = []
        for s in first_al["subs"]:
            kinds["sub"] += 1
            (near if s["ratio"] >= NEAR_MISS else far).append(s)
            tip = _char_tip(normalize(s["word"])[0], s["said"]) if s["ratio"] >= NEAR_MISS else ""
            syl = "-".join(syllables(s["word"]))
            word_notes.setdefault(
                s["word"], f"dijo «{s['said']}» en vez de «{s['word']}» en «{_context(exp, s['word'])}»"
                + (f" ({tip})" if tip else "") + (f" · sílabas: {syl}" if "-" in syl else ""))
            entry_words.append(s["word"])
        for w in first_al["omitted"]:
            kinds["omit"] += 1
            word_notes.setdefault(w, f"omitió «{w}» en «{_context(exp, w)}»")
            entry_words.append(w)
        kinds["extra"] += len(first_al["extra"])
        kinds["run"] = kinds.get("run", 0) + len(first_al["runs"])
        problem_words += entry_words
        if e["outcome"] == "failed":
            unresolved += [s["word"] for s in final["subs"]] + list(final["omitted"])

        for w in dict.fromkeys(entry_words):
            for t in _traits(w):
                trait_words.setdefault(t, []).append(w)

        problems.append({"entry": e, "first": first_al, "final": final})

    # ---------- redacción ----------
    md = ["## Resumen", ""]
    if total == 0:
        md.append(f"En esta sesión {name} no llegó a completar ninguna lectura, así que todavía no hay datos para analizar.")
    else:
        partes = [f"{first} bien a la primera ({_pct(first, total)} %)"]
        if second:
            partes.append(f"{second} bien en el segundo intento")
        if failed:
            partes.append(f"{failed} que necesitó que el tutor se lo leyera de nuevo" if failed == 1
                          else f"{failed} que necesitaron que el tutor se los leyera de nuevo")
        md.append(f"{name} leyó {total} fragmento{'s' if total != 1 else ''} en voz alta: "
                  + ", ".join(partes[:-1]) + (" y " if len(partes) > 1 else "") + partes[-1] + ".")
        if silences:
            md.append(f"Hubo {silences} intento{'s' if silences != 1 else ''} en que no se escuchó nada (no cuentan como fallo).")
        if first == total:
            md.append(f"Lectura muy fluida: {name} no necesitó repetir ningún fragmento.")
        elif failed == 0:
            md.append(f"Aunque tropezó en algunos fragmentos, {name} los corrigió por sí mismo en el segundo intento.")
        elif failed == total:
            md.append(f"Esta lectura fue exigente para {name}: conviene bajar el ritmo y practicar fragmentos más cortos.")

    md += ["", "## Lo que pronunció bien", ""]
    if first:
        md.append(f"- Leyó sin errores {first} fragmento{'s' if first != 1 else ''} a la primera.")
    if well_read_hard:
        md.append(f"- Pronunció correctamente palabras largas como {_q(well_read_hard)}.")
    if fixed_on_retry:
        md.append(f"- Al reintentar, corrigió por sí mismo {_q(fixed_on_retry)}: sabe pronunciarlas, "
                  "le falta seguridad al leer de corrido.")
    if not (first or well_read_hard or fixed_on_retry):
        md.append("- Se esforzó en cada turno y mantuvo la sesión; eso ya es un avance.")

    md += ["", "## Dónde tuvo dificultad", ""]
    if not problems:
        md.append("- No hubo dificultades en esta sesión.")
        md.append("")
    for n, p in enumerate(problems, 1):
        e, f = p["entry"], p["first"]
        heard1 = e["attempts"][0]["transcript"]
        bits = []
        for s in f["subs"]:
            kind = "casi igual" if s["ratio"] >= NEAR_MISS else "distinta"
            bits.append(f"leyó «{s['said']}» en lugar de «{s['word']}» ({kind})")
        for r in f["runs"]:
            bits.append(f"dejó sin leer un tramo de {len(r)} palabras («{' '.join(r[:3])}…»)")
        if f["omitted"]:
            bits.append(f"se saltó {_q(f['omitted'])}")
        if f["extra"]:
            bits.append(f"añadió {_q(f['extra'], 4)}, que no estaban en el texto")
        outcome = {"second_try": "Lo corrigió en el segundo intento.",
                   "failed": "No lo logró en el segundo intento; el tutor lo leyó despacio y se siguió adelante."}[e["outcome"]]
        md.append(f"**Fragmento {n}:** «{e['expected']}»")
        lejos = e["attempts"][0]["accuracy"] < 0.6
        md.append(f"- Primer intento (precisión {int(e['attempts'][0]['accuracy'] * 100)} %): "
                  + (f"se le entendió «{_short(heard1)}». " if lejos else "")
                  + ("; ".join(bits).capitalize() + "." if bits else "La lectura no coincidió con el texto."))
        if len(e["attempts"]) > 1:
            a2 = e["attempts"][1]
            if e["outcome"] == "second_try":
                detail = "Leyó el fragmento completo y correcto."
            else:
                rest = [s["word"] for s in p["final"]["subs"]] + list(p["final"]["omitted"])
                detail = ((f"Aún falló {_q(rest)}." if len(rest) == 1 else f"Aún fallaron {_q(rest)}.") if rest else "No coincidió con el texto.")
            md.append(f"- Segundo intento (precisión {int(a2['accuracy'] * 100)} %): {detail} {outcome}")
        else:
            md.append(f"- {outcome}")
        md.append("")

    md += ["## Patrones observados", ""]
    if not problems:
        md.append("- Sin patrones de error en esta sesión.")
    else:
        tot = sum(kinds.values()) or 1
        if kinds.get("run"):
            md.append(f"- En {kinds['run']} ocasión(es) **dejó un tramo entero sin leer**: puede haber perdido la línea, "
                      "haberse detenido a mitad del párrafo o fallado el micrófono. Conviene preguntarle si se cansó.")
        if kinds["omit"] >= kinds["sub"] and kinds["omit"]:
            md.append(f"- Tiende a **saltarse palabras** ({kinds['omit']} omisiones): suele pasar al leer rápido o al perder la línea.")
        if kinds["sub"] > kinds["omit"]:
            md.append(f"- Tiende a **cambiar palabras por otras** ({kinds['sub']} sustituciones): confunde sonidos parecidos o adivina la palabra.")
        if kinds["extra"]:
            md.append(f"- **Añade palabras** que no están en el texto ({kinds['extra']} veces): probablemente improvisa.")
        if near and len(near) >= len(far):
            md.append("- La mayoría de sus errores son palabras que **casi** dijo bien (por ejemplo "
                      + _q([f"{s['said']}→{s['word']}" for s in near], 3) + "). Es una pronunciación incompleta, "
                      "no desconocimiento de la palabra.")
        elif far:
            md.append("- Varias palabras las cambió por otras muy distintas (por ejemplo "
                      + _q([f"{s['said']}→{s['word']}" for s in far], 3) + "): quizá no las reconoce al verlas escritas.")
        for trait, words in sorted(trait_words.items(), key=lambda kv: -len(kv[1])):
            if len(set(words)) >= 2:
                md.append(f"- Muestra dificultad con {trait}: {_q(words, 4)}.")
        md.append(f"- Nota: el sistema ignora las tildes y depende del reconocimiento de voz del navegador; "
                  "un error aislado puede ser del micrófono y no del niño.")

    ordered = list(dict.fromkeys(unresolved + problem_words))
    focus = ([w for w in ordered if len(normalize(w)[0]) > 3] or ordered)[:5]
    word_notes = {w: n for w, n in word_notes.items() if w in set(ordered)}
    md += ["", "## Recomendaciones para la próxima sesión", ""]
    md.append(f"_Instrucciones para el tutor (IA) que acompañe a {name}:_")
    if focus:
        md.append(f"- Vigilar estas palabras y practicarlas despacio, por sílabas, antes de leerlas en la frase: {_q(focus)}.")
    if kinds.get("run"):
        md.append("- Si deja tramos sin leer, animarlo a seguir con el dedo y a leer sin detenerse a mitad del párrafo.")
    if kinds["omit"]:
        md.append("- Pedirle que siga el texto con el dedo o lea un poco más despacio para no saltarse palabras.")
    if kinds["sub"]:
        md.append("- Cuando cambie una palabra, repetirle solo esa palabra y luego la frase completa, con tono animado.")
    if trait_words:
        top = max(trait_words.items(), key=lambda kv: len(set(kv[1])))[0]
        md.append(f"- Hacer un pequeño juego de pronunciación con {top}.")
    if first == total and total:
        md.append("- Proponer fragmentos un poco más largos o un libro de nivel mayor.")
    elif failed:
        md.append("- Mantener fragmentos cortos y elogiar el esfuerzo antes de corregir, para evitar frustración.")
    if not (focus or kinds["omit"] or kinds["sub"] or kinds.get("run")):
        md.append("- Seguir con el mismo ritmo y felicitarlo por su lectura.")

    # ---------- resumen de una línea ----------
    if total == 0:
        summary = "sin lecturas completadas"
    else:
        summary = f"{first}/{total} a la primera"
        if focus:
            summary += f" · a reforzar: {', '.join(focus[:3])}"
        else:
            summary += " · sin dificultades"

    return {"markdown": "\n".join(md) + "\n", "summary": summary,
            "focus": focus, "word_notes": word_notes}
