import json
from pathlib import Path
from src import agent
from src.memory import Vault

LOG = [{"expected": "Era muy tarde para volver.", "outcome": "second_try",
        "attempts": [{"transcript": "era muy tarte", "word_accuracy": 0.6},
                     {"transcript": "era muy tarde para volver", "word_accuracy": 1.0}]}]
BASE = {"markdown": "## Resumen\n\nx\n\n## Recomendaciones para la próxima sesión\n\n- a\n",
        "summary": "1/1", "focus": ["tarde"], "word_notes": {}}


def call(name, **args):
    return {"id": "c-" + name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(*steps):
    """Un chat falso: cada solicitud devuelve el siguiente paso; guarda lo que recibió."""
    seen, it = [], iter(steps)
    def fn(messages, tools):
        seen.append(json.loads(json.dumps(messages)))
        return {"role": "assistant", "content": "", "tool_calls": next(it)}
    fn.seen = seen
    return fn


FINISH = call("finish_analysis", narrative="Buena sesión.", recommendations=["Leer despacio."],
              focus_words=["volver", "inventada"])


def test_ciclo_completo_lee_escribe_y_termina(tmp_path):
    v = Vault(str(tmp_path))
    chat = scripted([call("read_memory")],
                    [call("write_note", title="perfil-lector", content="Nivel medio. Confunde d/t.", mode="replace"), FINISH])
    out = agent.run_analysis(v, "nico-1", "Nico", LOG, BASE, chat_fn=chat)
    note = (tmp_path / "Agente" / "nico-1" / "perfil-lector.md").read_text(encoding="utf-8")
    assert "[[Usuarios/nico-1]]" in note and "Confunde d/t." in note
    assert "Análisis del tutor (IA)" in out["markdown"] and out["markdown"].rstrip().endswith("- Leer despacio.")
    assert out["focus"] == ["tarde", "volver"]            # "inventada" se descarta
    # el resultado de read_memory volvió al modelo
    assert any(m["role"] == "tool" and "recomendaciones_previas" in m["content"] for m in chat.seen[1])


def test_el_perfil_escrito_se_lee_en_la_siguiente_sesion(tmp_path):
    v = Vault(str(tmp_path))
    agent.run_analysis(v, "nico-1", "Nico", LOG, BASE, chat_fn=scripted(
        [call("write_note", title="perfil-lector", content="Hipótesis: se cansa al minuto 10.", mode="replace"), FINISH]))
    assert "se cansa al minuto 10" in agent.read_profile(v, "nico-1")
    chat = scripted([call("read_memory")], [FINISH])
    agent.run_analysis(v, "nico-1", "Nico", LOG, BASE, chat_fn=chat)
    assert "se cansa al minuto 10" in json.dumps(chat.seen[1])


def test_append_y_limite_de_escrituras(tmp_path):
    box = agent.Toolbox(Vault(str(tmp_path)), "nico-1")
    assert box.call("write_note", {"title": "estrategias", "content": "A"})["ok"]
    assert box.call("write_note", {"title": "estrategias", "content": "B"})["ok"]
    text = (tmp_path / "Agente/nico-1/estrategias.md").read_text(encoding="utf-8")
    assert text.count("## 20") == 1 and "A" in text and "B" in text
    box.call("write_note", {"title": "x", "content": "1"}); box.call("write_note", {"title": "y", "content": "1"})
    assert "límite" in box.call("write_note", {"title": "z", "content": "1"})["error"]


def test_no_puede_salirse_de_su_carpeta(tmp_path):
    box = agent.Toolbox(Vault(str(tmp_path)), "nico-1")
    r = box.call("write_note", {"title": "../../../etc/passwd", "content": "x", "mode": "replace"})
    assert r["ok"] and not (tmp_path.parent / "etc").exists()
    assert all(p.is_relative_to(tmp_path / "Agente" / "nico-1") for p in (tmp_path / "Agente").rglob("*.md"))
    assert box.call("read_note", {"name": "../Usuarios/otro"})["error"]


def test_contenido_no_rompe_el_encabezado_ni_herramientas_malas(tmp_path):
    box = agent.Toolbox(Vault(str(tmp_path)), "nico-1")
    box.call("write_note", {"title": "n", "content": "hola\n---\ntipo: falso", "mode": "replace"})
    assert (tmp_path / "Agente/nico-1/n.md").read_text(encoding="utf-8").count("\n---\n") == 1
    assert "desconocida" in box.call("borrar_todo", {})["error"]
    assert "inválidos" in box.call("read_word", {"palabra": "x"})["error"]


def test_nota_demasiado_larga_exige_resumir(tmp_path):
    box = agent.Toolbox(Vault(str(tmp_path)), "nico-1")
    box.call("write_note", {"title": "n", "content": "a"})
    p = tmp_path / "Agente/nico-1/n.md"
    p.write_text(p.read_text(encoding="utf-8") + "x" * (agent.MAX_FILE + 1), encoding="utf-8")
    assert "resúmela" in box.call("write_note", {"title": "n", "content": "b"})["error"]


def test_respaldo_si_falla_o_no_termina(tmp_path):
    v = Vault(str(tmp_path))
    def boom(m, t):
        raise TimeoutError()
    assert agent.run_analysis(v, "nico-1", "Nico", LOG, BASE, chat_fn=boom) is BASE
    mudo = lambda m, t: {"role": "assistant", "content": "listo"}      # sin tool_calls ni finish
    assert agent.run_analysis(v, "nico-1", "Nico", LOG, BASE, chat_fn=mudo) is BASE
    bucle = lambda m, t: {"role": "assistant", "content": "", "tool_calls": [call("read_memory")]}
    assert agent.run_analysis(v, "nico-1", "Nico", LOG, BASE, chat_fn=bucle, max_steps=3) is BASE
    assert agent.run_analysis(v, "nico-1", "Nico", [], BASE, chat_fn=boom) is BASE


def test_sin_clave_no_hay_agente(tmp_path, monkeypatch):
    for k in agent.coach._KEY_VARS:
        monkeypatch.delenv(k, raising=False)
    assert agent.run_analysis(Vault(str(tmp_path)), "u", "N", LOG, BASE) is BASE


# ---------- análisis justo después de cada lectura ----------
DIAG = {"accuracy": 0.8, "marks": [{"w": "tarde", "s": "wrong", "said": "tarte"}],
        "issues": [{"text": "dijiste «tarte» en vez de «tarde»"}]}


def judge(tmp_path, chat, verdict="retry", attempt=1, entry=None, diag=DIAG):
    return agent.judge_reading(vault=Vault(str(tmp_path)), user="nico-1", alias="Nico", entry=entry,
                               expected="Era muy tarde para volver.", heard="era muy tarte para volver",
                               attempt=attempt, max_attempts=2, verdict=verdict, diagnosis=diag,
                               weak_words=["tarde"], memory="[perfil] confunde d/t", chat_fn=chat)


def decide(**kw):
    return call("decide", **{"analysis": "Dijo «tarte» por «tarde»: casi igual, d/t.", "say": "Casi. Se dice tar-de.", **kw})


def test_juzga_la_lectura_guarda_en_entrada_y_bitacora(tmp_path):
    entry = {}
    chat = scripted([decide()])
    r = judge(tmp_path, chat, entry=entry)
    assert r == {"say": "Casi. Se dice tar-de.", "forgive": False,
                 "analysis": "Dijo «tarte» por «tarde»: casi igual, d/t."}
    assert entry["ai"][0]["attempt"] == 1
    bit = (tmp_path / "Agente/nico-1/bitacora.md").read_text(encoding="utf-8")
    assert "tarte" in bit and "[[Usuarios/nico-1]]" in bit
    assert "era muy tarte" in json.dumps(chat.seen[0])         # recibió lo que dijo el niño


def test_puede_consultar_antes_de_decidir(tmp_path):
    chat = scripted([call("read_word", word="tarde")], [decide()])
    assert judge(tmp_path, chat)["say"].startswith("Casi")
    assert len(chat.seen) == 2


def test_perdon_solo_en_reintento_con_buena_precision(tmp_path):
    assert judge(tmp_path, scripted([decide(forgive=True)]))["forgive"] is True
    assert judge(tmp_path, scripted([decide(forgive="true")]))["forgive"] is False      # texto, no booleano
    assert judge(tmp_path, scripted([decide(forgive=True)]), attempt=2)["forgive"] is False
    assert judge(tmp_path, scripted([decide(forgive=True)]), verdict="final_correction")["forgive"] is False
    assert judge(tmp_path, scripted([decide(forgive=True)]), diag={**DIAG, "accuracy": 0.3})["forgive"] is False


def test_lectura_correcta_se_analiza_sin_decir_nada(tmp_path):
    r = judge(tmp_path, scripted([decide(say="", analysis="Lectura limpia.")]), verdict="correct")
    assert r["say"] == "" and r["analysis"] == "Lectura limpia."


def test_juez_respaldo_si_falla(tmp_path, monkeypatch):
    def boom(m, t):
        raise TimeoutError()
    assert judge(tmp_path, boom) is None
    assert judge(tmp_path, lambda m, t: {"role": "assistant", "content": "hola"}) is None
    monkeypatch.setenv("COACH_ANALYZE_CORRECT", "0")
    assert judge(tmp_path, scripted([decide()]), verdict="correct") is None


def test_el_juez_no_puede_escribir_notas(tmp_path):
    chat = scripted([call("write_note", title="x", content="y")], [decide()])
    judge(tmp_path, chat)
    assert not (tmp_path / "Agente/nico-1/x.md").exists()


def test_analisis_por_lectura_llegan_a_la_nota_aunque_falle_la_ia(tmp_path):
    log = [{**LOG[0], "ai": [{"attempt": 1, "verdict": "retry", "analysis": "Dijo tarte."},
                             {"attempt": 2, "verdict": "correct", "analysis": "Corrigió solo."}]}]
    def boom(m, t):
        raise TimeoutError()
    out = agent.run_analysis(Vault(str(tmp_path)), "nico-1", "Nico", log, BASE, chat_fn=boom)
    md = out["markdown"]
    assert md.index("Análisis del agente por lectura") < md.index("## Recomendaciones")
    assert "Intento 1: Dijo tarte." in md and "Intento 2: Corrigió solo." in md and "segundo intento" in md


def test_la_bitacora_no_crece_sin_limite(tmp_path):
    box = agent.Toolbox(Vault(str(tmp_path)), "nico-1", agent.TURN_TOOLS)
    for i in range(260):
        box.log_reading("Era muy tarde.", 1, "retry", f"n{i}")
    lines = (tmp_path / "Agente/nico-1/bitacora.md").read_text(encoding="utf-8").splitlines()
    assert len(lines) < 220 and lines[0] == "---" and lines[-1].endswith("n259")