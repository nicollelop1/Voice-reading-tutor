import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from web.auth import hash_password, verify_password

MAX_PROFILES = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    consent_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    birth_year INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

_DUMMY_HASH = hash_password("contraseña-de-relleno")   


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path="data/app.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- cuentas ----------
    def create_account(self, full_name: str, email: str, password: str) -> int:
        try:
            with self._conn() as c:
                cur = c.execute(
                    "INSERT INTO accounts (full_name, email, password_hash, consent_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (full_name.strip(), email.strip().lower(), hash_password(password), _now(), _now()),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("email_taken")

    def authenticate(self, email: str, password: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM accounts WHERE email = ?",
                            (email.strip().lower(),)).fetchone()
        if row is None:
            verify_password(password, _DUMMY_HASH)
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "full_name": row["full_name"], "email": row["email"]}

    def get_account(self, account_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT id, full_name, email FROM accounts WHERE id = ?",
                            (account_id,)).fetchone()
        return dict(row) if row else None

    # ---------- perfiles ----------
    def create_profile(self, account_id: int, alias: str, birth_year: int) -> int:
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM profiles WHERE account_id = ?",
                          (account_id,)).fetchone()[0]
            if n >= MAX_PROFILES:
                raise ValueError("too_many_profiles")
            cur = c.execute(
                "INSERT INTO profiles (account_id, alias, birth_year, created_at) VALUES (?, ?, ?, ?)",
                (account_id, alias.strip(), birth_year, _now()),
            )
            return cur.lastrowid

    def list_profiles(self, account_id: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM profiles WHERE account_id = ? ORDER BY id",
                             (account_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_profile(self, profile_id: int, account_id: int) -> dict | None:
        """Solo devuelve el perfil si pertenece a esa cuenta (evita ver perfiles ajenos por la URL)."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM profiles WHERE id = ? AND account_id = ?",
                            (profile_id, account_id)).fetchone()
        return dict(row) if row else None