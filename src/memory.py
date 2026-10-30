import re
import unicodedata
from datetime import datetime
from pathlib import Path


def slugify(text: str) -> str:
    """'Nicolás Pérez' -> 'nicolas-perez' (nombre seguro para archivos)."""
    text = unicodedata.normalize("NFD", text.lower().strip())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "sin-nombre"


class Vault:
    """Memoria del agente: notas Markdown enlazadas (grafo de Obsidian)."""

    def __init__(self, root: str = "vault"):
        self.root = Path(root)
        for folder in ("Usuarios", "Libros", "Sesiones", "Palabras"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def _append(self, path: Path, header: str, line: str):
        """Agrega una línea al final de una nota (la crea con encabezado si no existe)."""
        if not path.exists():
            path.write_text(header, encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def login(self, name: str, user_id: str | None = None) -> dict:
        """Identifica al usuario; lo crea si es nuevo y devuelve su perfil.
        `name` es el apodo que se muestra; `user_id` es la clave interna (si no se da, se deriva del nombre)."""
        user = user_id or slugify(name)
        path = self.root / "Usuarios" / f"{user}.md"
        is_new = not path.exists()
        if is_new:
            path.write_text(
                f"---\ntipo: usuario\nnombre: {name}\ncreado: {datetime.now():%Y-%m-%d}\n---\n"
                f"# {name}\n\n## Sesiones\n",
                encoding="utf-8",
            )
        return {
            "user": user,
            "name": name,
            "new": is_new,
            "sessions": self.count_sessions(user),
            "weak_words": self.weak_words(user),
            "coaching": self.coaching(user),
        }

    def coaching(self, user: str, n: int = 2) -> str:
        """Recomendaciones de las últimas sesiones del usuario (lo que la IA debe tener en cuenta)."""
        files = sorted((self.root / "Sesiones").glob(f"*_{user}_*.md"), reverse=True)[:n]
        parts = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            m = re.search(r"## Recomendaciones para la próxima sesión\n(.*?)(?=\n## |\Z)", text, re.S)
            if m and m.group(1).strip():
                parts.append(f"[{f.stem[:10]}]\n" + m.group(1).strip())
        return "\n\n".join(parts)

    def count_sessions(self, user: str) -> int:
        path = self.root / "Usuarios" / f"{user}.md"
        if not path.exists():
            return 0
        lines = path.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line.startswith("- [[Sesiones/"))

    def weak_words(self, user: str, top_n: int = 5) -> list[tuple[str, int]]:
        """Palabras que más ha fallado el usuario: [(palabra, veces), ...]."""
        tag = f"[[Usuarios/{user}]]"
        counts: dict[str, int] = {}
        for note in (self.root / "Palabras").glob("*.md"):
            lines = note.read_text(encoding="utf-8").splitlines()
            n = sum(1 for line in lines if line.startswith("- ") and tag in line)
            if n:
                counts[note.stem] = n
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]

    def save_session(self, user: str, book: str, stats: dict, errors: list[str],
                     analysis: dict | None = None) -> str:
        """Guarda la sesión con su análisis y enlaza usuario, libro y palabras falladas.

        `analysis` viene de src.analysis.analyze_session: markdown, summary y word_notes."""
        analysis = analysis or {}
        word_notes = analysis.get("word_notes", {})
        now = datetime.now()
        book_slug = slugify(Path(book).stem)
        session_id = f"{now:%Y-%m-%d_%H%M%S}_{user}_{book_slug}"
        words = sorted({slugify(w) for w in errors} | {slugify(w) for w in analysis.get("focus", [])})

        body = [
            "---",
            "tipo: sesion",
            f"usuario: {user}",
            f"libro: {book_slug}",
            f"fecha: {now:%Y-%m-%d %H:%M}",
            "---",
            f"# Sesión {now:%Y-%m-%d %H:%M}",
            "",
            f"- Usuario: [[Usuarios/{user}]]",
            f"- Libro: [[Libros/{book_slug}]]",
            f"- Correctas a la primera: {stats.get('first_try', 0)}",
            f"- Correctas a la segunda: {stats.get('second_try', 0)}",
            f"- Corregidas por el agente: {stats.get('failed', 0)}",
            "",
        ]
        if analysis.get("markdown"):
            body += [analysis["markdown"].rstrip(), ""]
        body += ["## Palabras falladas (enlaces del grafo)"]
        body += [f"- [[Palabras/{w}]]" for w in words] or ["- (ninguna)"]
        (self.root / "Sesiones" / f"{session_id}.md").write_text("\n".join(body) + "\n", encoding="utf-8")

        link = f"[[Sesiones/{session_id}]]"
        extra = f" · {analysis['summary']}" if analysis.get("summary") else ""
        self._append(self.root / "Usuarios" / f"{user}.md", "", f"- {link}{extra}")
        self._append(self.root / "Libros" / f"{book_slug}.md",
                     f"# {book_slug}\n\n## Sesiones\n", f"- {link} · [[Usuarios/{user}]]{extra}")
        notes_by_slug = {slugify(k): v for k, v in word_notes.items()}
        for w in words:
            detail = f" · {notes_by_slug[w]}" if w in notes_by_slug else ""
            self._append(self.root / "Palabras" / f"{w}.md",
                         f"# {w}\n\n## Fallos\n",
                         f"- {now:%Y-%m-%d} · [[Usuarios/{user}]] · {link}{detail}")
        return session_id