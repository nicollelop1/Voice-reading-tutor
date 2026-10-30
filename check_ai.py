"""Comprobador del agente: python check_ai.py   (desde la carpeta del proyecto)
Prueba la clave, el modelo, una respuesta simple y una llamada con herramientas, y te dice qué falla."""
import sys
import tempfile

from dotenv import load_dotenv

load_dotenv()
from src import agent, coach
from src.memory import Vault


def step(n, title):
    print(f"\n[{n}] {title}")


step(1, "Configuración")
cfg = coach._config()
var = next((v for v in coach._KEY_VARS if __import__("os").getenv(v)), None)
print("  clave encontrada en:", var or "NINGUNA")
print("  modelo:", cfg["model"])
print("  url:", cfg["url"])
if not var:
    print("\n❌ No hay clave. En tu .env escribe una línea:  GEMINI_API_KEY=tu_clave  (sin comillas ni espacios)"
          "\n   y reinicia uvicorn (el modo --reload NO relee el .env).")
    sys.exit(1)

step(2, "Respuesta simple")
try:
    r = coach.chat("Responde solo con la palabra: listo", "hola")
    print("  respuesta:", r.strip()[:80])
except Exception:
    print("\n❌ Falló. Lee el mensaje [agente] de arriba:"
          "\n   404/400 con el nombre del modelo -> pon COACH_MODEL=<un modelo de tu AI Studio>"
          "\n   401/403 -> la clave es incorrecta o no tiene permiso"
          "\n   429 -> sin cuota por ahora; espera o cambia de modelo")
    sys.exit(1)

step(3, "Llamada con herramientas")
coach._down_until = 0.0
try:
    msg = coach.chat_messages(
        [{"role": "system", "content": "Debes responder llamando a la herramienta decide."},
         {"role": "user", "content": "El niño leyó bien. Decide."}], agent.TURN_TOOLS)
    calls = msg.get("tool_calls") or []
    print("  herramientas pedidas:", [c["function"]["name"] for c in calls] or "NINGUNA")
    if not calls:
        print("  texto:", (msg.get("content") or "")[:120])
        print("\n❌ El modelo no usó herramientas. Prueba otro con COACH_MODEL=...")
        sys.exit(1)
except Exception:
    print("\n❌ Falló con herramientas (mira el mensaje [agente] de arriba).")
    sys.exit(1)

step(4, "Agente completo sobre una lectura de ejemplo")
coach._down_until = 0.0
with tempfile.TemporaryDirectory() as tmp:
    out = agent.judge_reading(
        vault=Vault(tmp), user="prueba-1", alias="Prueba", entry={}, expected="Era muy tarde para volver.",
        heard="era muy tarte para volver", attempt=1, max_attempts=2, verdict="retry",
        diagnosis={"accuracy": 0.8, "marks": [], "issues": [{"text": "dijiste «tarte» en vez de «tarde»"}]})
print("  resultado:", out)
print("\n✅ Todo funciona." if out else "\n❌ El agente no devolvió decisión (mira los mensajes [agente]).")