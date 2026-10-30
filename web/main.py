import os
import secrets
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from src import coach, agent
from src.memory import Vault, slugify
from src.tutor import TutorSession
from src.analysis import analyze_session
from src.ingest import load_text, split_chunks
from web.auth import validate_profile, validate_registration
from web.db import MAX_PROFILES, Database

load_dotenv()

BASE = Path(__file__).parent
secret = os.getenv("SESSION_SECRET")
if not secret:
    print("⚠️  Falta SESSION_SECRET en .env: las sesiones se cerrarán al reiniciar el servidor.")
    secret = secrets.token_hex(32)

app = FastAPI(title="Voice Reading Tutor")
app.add_middleware(SessionMiddleware, secret_key=secret, same_site="lax", max_age=60 * 60 * 8)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

db = Database("data/app.db")
vault = Vault("vault")

LIBROS_DIR = Path("libros")
LIBROS_DIR.mkdir(exist_ok=True)


# ---------- utilidades ----------
def vault_key(profile: dict) -> str:
    """Clave del niño en el vault: apodo + id."""
    return f"{slugify(profile['alias'])}-{profile['id']}"


def current_account(request: Request) -> dict | None:
    account_id = request.session.get("account_id")
    return db.get_account(account_id) if account_id else None


def require_account(request: Request) -> dict:
    account = current_account(request)
    if account is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return account


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_hex(16)
        request.session["csrf"] = token
    return token


def verify_csrf(request: Request, token: str):
    expected = request.session.get("csrf", "")
    if not expected or not secrets.compare_digest(expected.encode(), token.encode()):
        raise HTTPException(status_code=403, detail="Formulario inválido. Recarga la página e inténtalo de nuevo.")


def render(request: Request, name: str, status_code: int = 200, **ctx):
    ctx["csrf"] = csrf_token(request)
    ctx["account"] = current_account(request)
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


# ---------- páginas ----------
@app.get("/")
def home(request: Request):
    return RedirectResponse("/dashboard" if current_account(request) else "/login", status_code=303)


@app.get("/register")
def register_form(request: Request):
    return render(request, "register.html", errors=[], form={})


@app.post("/register")
def register(request: Request, csrf: str = Form(""), full_name: str = Form(""),
             email: str = Form(""), password: str = Form(""),
             password2: str = Form(""), consent: str = Form("")):
    verify_csrf(request, csrf)
    errors = validate_registration(full_name, email, password, password2, bool(consent))
    account_id = None
    if not errors:
        try:
            account_id = db.create_account(full_name, email, password)
        except ValueError:
            errors.append("Ya existe una cuenta con ese correo.")
    if errors:
        return render(request, "register.html", status_code=400, errors=errors,
                      form={"full_name": full_name, "email": email})
    request.session.clear()
    request.session["account_id"] = account_id
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/login")
def login_form(request: Request):
    return render(request, "login.html", errors=[], form={})


@app.post("/login")
def login(request: Request, csrf: str = Form(""), email: str = Form(""), password: str = Form("")):
    verify_csrf(request, csrf)
    account = db.authenticate(email, password)
    if account is None:
        return render(request, "login.html", status_code=400,
                      errors=["Correo o contraseña incorrectos."], form={"email": email})
    request.session.clear()
    request.session["account_id"] = account["id"]
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    verify_csrf(request, csrf)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/dashboard")
def dashboard(request: Request):
    account = require_account(request)
    cards = []
    for p in db.list_profiles(account["id"]):
        key = vault_key(p)
        cards.append({**p, "sessions": vault.count_sessions(key), "weak": vault.weak_words(key)})
    return render(request, "dashboard.html", cards=cards)


@app.get("/profiles/new")
def profile_new_form(request: Request):
    require_account(request)
    return render(request, "profile_new.html", errors=[], form={})


@app.post("/profiles/new")
def profile_create(request: Request, csrf: str = Form(""),
                   alias: str = Form(""), birth_year: str = Form("")):
    account = require_account(request)
    verify_csrf(request, csrf)
    errors, year = validate_profile(alias, birth_year)
    profile_id = None
    if not errors:
        try:
            profile_id = db.create_profile(account["id"], alias, year)
        except ValueError:
            errors.append(f"Máximo {MAX_PROFILES} perfiles por cuenta.")
    if errors:
        return render(request, "profile_new.html", status_code=400, errors=errors,
                      form={"alias": alias, "birth_year": birth_year})
    profile = db.get_profile(profile_id, account["id"])
    vault.login(profile["alias"], user_id=vault_key(profile))
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/profiles/{profile_id}")
def profile_detail(request: Request, profile_id: int):
    account = require_account(request)
    profile = db.get_profile(profile_id, account["id"])
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    key = vault_key(profile)
    return render(request, "profile.html", profile=profile,
                  sessions=vault.count_sessions(key), weak=vault.weak_words(key))


# ---------- LIBROS ----------
@app.get("/profiles/{profile_id}/books")
def profile_books(request: Request, profile_id: int):
    account = require_account(request)
    profile = db.get_profile(profile_id, account["id"])
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    books = [f.name for f in LIBROS_DIR.glob("*.*") if f.suffix.lower() in ['.pdf', '.txt', '.epub']]
    return render(request, "profile_books.html", profile=profile, books=books)


@app.post("/profiles/{profile_id}/books/upload")
async def profile_books_upload(request: Request, profile_id: int, book_file: UploadFile = File(...)):
    account = require_account(request)
    profile = db.get_profile(profile_id, account["id"])
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    if book_file and book_file.filename:
        filename = Path(book_file.filename).name
        filepath = LIBROS_DIR / filename
        contents = await book_file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
    return RedirectResponse(f"/profiles/{profile_id}/books", status_code=303)


# ---------- TUTOR CON AGENTE ----------
tutor_sessions: dict[tuple[int, int], dict] = {}


def _books() -> list[str]:
    return sorted(f.name for f in LIBROS_DIR.glob("*.*") if f.suffix.lower() in (".pdf", ".txt", ".epub"))


def _owned_profile(request: Request, profile_id: int) -> tuple[dict, dict]:
    account = require_account(request)
    profile = db.get_profile(profile_id, account["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    return account, profile


def _active(account: dict, profile_id: int) -> dict:
    state = tutor_sessions.get((account["id"], profile_id))
    if state is None:
        raise HTTPException(status_code=409, detail="No hay sesión activa. Vuelve a empezar.")
    return state


@app.get("/profiles/{profile_id}/tutor")
def profile_tutor(request: Request, profile_id: int, book: str = ""):
    _, profile = _owned_profile(request, profile_id)
    return render(request, "profile_tutor.html", profile=profile, books=_books(), selected_book=book)


@app.post("/profiles/{profile_id}/tutor/start")
async def tutor_start(request: Request, profile_id: int):
    account, profile = _owned_profile(request, profile_id)
    data = await request.json()
    verify_csrf(request, data.get("csrf", ""))

    book = Path(str(data.get("book", ""))).name
    book_path = LIBROS_DIR / book
    if not book or not book_path.is_file():
        raise HTTPException(status_code=400, detail="Libro no encontrado")

    key = vault_key(profile)
    memory = vault.login(profile["alias"], user_id=key)
    tutor = TutorSession()
    tutor.weak_words = [w for w, _ in memory["weak_words"]]
    info = tutor.load_book(str(book_path))
    if info["fragmentos"] < 1:
        raise HTTPException(status_code=400, detail="El libro no tiene texto legible")
    
    # Se carga la memoria y el perfil del agente en vivo
    tutor_sessions[(account["id"], profile_id)] = {
        "tutor": tutor, "book": book, "user": key,
        "alias": profile["alias"], "saved": False, "pending": [],
        "coaching": memory["coaching"] + "\n\n" + agent.read_profile(vault, key)
    }

    alias = profile["alias"]
    greeting = (f"¡Hola, {alias}! Qué bueno verte de nuevo. Vamos a leer juntos."
                if not memory["new"] else f"¡Hola, {alias}! Bienvenido. Vamos a leer juntos.")
    if tutor.weak_words:
        greeting += " Hoy practicaremos con cuidado: " + ", ".join(tutor.weak_words[:3]) + "."
    return {"ok": True, "greeting": greeting, "fragments": info["fragmentos"],
            "book_title": Path(book).stem.replace("_", " ").replace("-", " ").strip().title(),
            "starts_with": info["inicio"], "skipped_words": info["palabras_omitidas"]}


@app.post("/profiles/{profile_id}/tutor/next")
async def tutor_next(request: Request, profile_id: int):
    account, _ = _owned_profile(request, profile_id)
    verify_csrf(request, (await request.json()).get("csrf", ""))
    tutor = _active(account, profile_id)["tutor"]
    turn = tutor.get_next_passage()
    turn["position"], turn["total"] = tutor.pos, len(tutor.chunks)
    return turn


@app.post("/profiles/{profile_id}/tutor/check")
@app.post("/profiles/{profile_id}/tutor/check")
async def tutor_check(request: Request, profile_id: int):
    account, _ = _owned_profile(request, profile_id)
    data = await request.json()
    verify_csrf(request, data.get("csrf", ""))
    state = _active(account, profile_id)
    tutor = state["tutor"]
    expected = tutor.expected or ""
    tutor.last_transcript = str(data.get("transcript", ""))[:2000]
    heard = tutor.last_transcript
    result = tutor.check_reading()
    result["attempt"] = tutor.attempts
    result["max_attempts"] = 2

    verdict = result.get("verdict")
    if verdict in ("correct", "retry", "final_correction"):
        entry = tutor._entry
        last = entry["attempts"][-1] if entry and entry["attempts"] else {}
        diagnosis = result.get("diagnosis") or {"accuracy": last.get("word_accuracy"), "issues": []}
        # 1) Python ya hizo el diagnóstico de hechos; 2) ahora el agente analiza por su cuenta y decide qué decir
        ai = await asyncio.to_thread(
            agent.judge_reading, vault=vault, user=state["user"], alias=state["alias"], entry=entry,
            expected=expected, heard=heard, attempt=tutor.attempts, max_attempts=2, verdict=verdict,
            diagnosis=diagnosis, weak_words=tutor.weak_words, memory=state.get("coaching", ""))
        if ai:
            if ai["forgive"]:
                result = tutor.forgive_last()
                result["max_attempts"] = 2
            result["say"] = ai["say"]            # lo que el navegador dirá en voz alta
            result["analysis"] = ai["analysis"]  # lo que el agente analizó (se muestra en pantalla)
            result["problems"] = ai["say"] or result.get("problems", "")
            result["coach"] = "ai"
    return result


def _save_if_needed(state: dict) -> str | None:
    tutor = state["tutor"]
    if state["saved"] or sum(tutor.stats.values()) == 0:
        return None
    
    analysis = analyze_session(state["alias"], tutor.log, tutor.stats, tutor.silences)
    try:
        # El cerebro crece: analiza toda la bitácora acumulada y genera notas en el vault
        analysis = agent.run_analysis(vault, state["user"], state["alias"], tutor.log, analysis)
    except Exception as e:
        print("⚠️ agente:", e)
        
    state["summary"] = analysis["summary"]
    state["saved"] = True
    return vault.save_session(state["user"], state["book"], tutor.stats, tutor.errors, analysis)


@app.post("/profiles/{profile_id}/tutor/finish")
async def tutor_finish(request: Request, profile_id: int):
    account, _ = _owned_profile(request, profile_id)
    verify_csrf(request, (await request.json()).get("csrf", ""))
    state = tutor_sessions.get((account["id"], profile_id))
    if state is None:
        return {"saved": False, "session_id": None, "words": []}
    
    # Espera a que terminen los análisis pendientes en segundo plano
    pending = state.get("pending") or []
    if pending:
        await asyncio.wait(pending, timeout=20)
        
    session_id = await asyncio.to_thread(_save_if_needed, state)
    words = sorted(set(state["tutor"].errors))
    tutor_sessions.pop((account["id"], profile_id), None)
    return {"saved": session_id is not None, "session_id": session_id, "words": words,
            "summary": state.get("summary", "")}


@app.get("/profiles/{profile_id}/history")
def profile_history(request: Request, profile_id: int):
    account = require_account(request)
    profile = db.get_profile(profile_id, account["id"])
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    
    key = vault_key(profile)
    vault_path = Path("vault/Sesiones")
    session_files = []
    if vault_path.exists():
        session_files = sorted(
            [f.name for f in vault_path.glob("*.md") if f"_{key}_" in f.name], reverse=True)

    sessions = []
    for name in session_files:
        text = (vault_path / name).read_text(encoding="utf-8")
        body = text.split("\n---\n", 1)[-1] if text.startswith("---") else text
        sessions.append({"name": name[:-3], "body": body.strip()})
    return render(request, "profile_history.html", profile=profile, sessions=sessions, weak=vault.weak_words(key))