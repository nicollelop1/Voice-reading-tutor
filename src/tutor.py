from src.ingest import load_book_text, split_chunks
from src.evaluate import evaluate_reading, normalize
from src.diagnose import diagnose

MAX_ATTEMPTS = 2


class TutorSession:
    """Estado de una sesión de lectura. Toda la lógica de turnos vive aquí."""

    def __init__(self):
        self.chunks: list[str] = []
        self.pos = 0
        self.expected: str | None = None
        self.attempts = 0
        self.last_transcript = ""
        self.errors: list[str] = []
        self.finished = False
        self.weak_words: list[str] = []   # palabras difíciles del usuario (vienen de la memoria)
        self.stats = {"first_try": 0, "second_try": 0, "failed": 0}
        self.log: list[dict] = []          # un registro por fragmento que el niño leyó
        self.silences = 0                  # veces que no se escuchó nada
        self._entry: dict | None = None
        self.book_info: dict = {"skipped_words": 0, "start_reason": "ninguno"}

    def forgive_last(self) -> dict:
        """La IA decidió que el fallo fue del reconocedor de voz: cuenta como acierto a la primera."""
        if self.expected is None:
            return {"error": "No hay ninguna lectura pendiente."}
        self.expected = None
        self.stats["first_try"] += 1
        if self._entry is not None:
            self._entry["outcome"] = "first_try"
            self._entry["forgiven"] = True
        return {"verdict": "correct", "attempt": self.attempts, "forgiven": True,
                "instruction": "Felicita en una frase y llama a get_next_passage."}

    def load_book(self, path: str) -> dict:
        """Carga un libro (PDF, EPUB o TXT) y lo divide en fragmentos."""
        text, self.book_info = load_book_text(path)
        self.chunks = split_chunks(text)
        self.pos = 0
        self.expected = None
        self.attempts = 0
        self.errors = []
        self.finished = False
        self.stats = {"first_try": 0, "second_try": 0, "failed": 0}
        self.log = []
        self.silences = 0
        self._entry = None
        return {"ok": True, "fragmentos": len(self.chunks),
                "inicio": self.chunks[0][:100] if self.chunks else "",
                "palabras_omitidas": self.book_info["skipped_words"]}

    def get_next_passage(self) -> dict:
        """Entrega el siguiente turno: narración (agente) o lectura (usuario)."""
        if self.expected is not None:
            return {"error": "El usuario aún debe terminar su lectura. Llama a check_reading."}

        if self.pos >= len(self.chunks):
            self.finished = True
            return {"role": "end", "words_to_practice": sorted(set(self.errors))}

        chunk = self.chunks[self.pos]
        is_narration = self.pos % 2 == 0
        self.pos += 1

        if is_narration:
            return {"role": "narrate", "text": chunk}

        self.expected = chunk
        self.attempts = 0
        self.last_transcript = ""
        self._entry = {"expected": chunk, "attempts": [], "outcome": None}
        self.log.append(self._entry)
        words = set(normalize(chunk))
        watch = [w for w in self.weak_words if w in words]
        return {"role": "user_read", "text": chunk, "watch_words": watch}

    def check_reading(self) -> dict:
        """Evalúa la lectura del usuario y aplica la regla de los 2 intentos."""
        if self.expected is None:
            return {"error": "No hay ninguna lectura pendiente."}

        if not self.last_transcript.strip():
            self.silences += 1
            return {"verdict": "no_audio",
                    "instruction": "No se escuchó nada. Pide que lea de nuevo. No cuenta como intento."}

        result = evaluate_reading(self.expected, self.last_transcript)
        prev = None
        if self._entry is not None and self._entry["attempts"]:
            last = self._entry["attempts"][-1]
            prev = {"accuracy": last.get("word_accuracy"), "bad": last.get("bad")}
        diag = diagnose(self.expected, self.last_transcript, self.attempts + 1, prev)
        if self._entry is not None:
            self._entry["attempts"].append({"transcript": self.last_transcript.strip(),
                                            "accuracy": result["accuracy"],
                                            "word_accuracy": diag["accuracy"], "bad": diag["bad"]})
        self.last_transcript = ""
        self.attempts += 1

        if result["correct"]:
            self.expected = None
            if self._entry is not None:
                self._entry["outcome"] = "first_try" if self.attempts == 1 else "second_try"
            self.stats["first_try" if self.attempts == 1 else "second_try"] += 1
            return {"verdict": "correct", "attempt": self.attempts,
                    "instruction": "Felicita en una frase y llama a get_next_passage."}

        problems = diag["spoken"]

        if self.attempts < MAX_ATTEMPTS:
            return {"verdict": "retry", "problems": problems, "diagnosis": diag,
                    "instruction": "Explica con cariño el diagnóstico (qué dijo, qué decía el texto y cómo se pronuncia) y di exactamente: 'Inténtalo una vez más.' Luego espera en silencio."}

        correct_text = self.expected
        self.errors += result["wrong_words"] + result["omitted_words"]
        self.stats["failed"] += 1
        if self._entry is not None:
            self._entry["outcome"] = "failed"
        self.expected = None
        return {"verdict": "final_correction", "problems": problems, "diagnosis": diag,
                "correct_text": correct_text,
                "instruction": "Corrige con cariño, lee despacio correct_text y llama a get_next_passage."}