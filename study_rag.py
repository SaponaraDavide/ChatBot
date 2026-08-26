"""
UniBot — backend RAG locale per interrogare i propri appunti universitari.

Pipeline: caricamento documenti -> chunking -> embedding -> Chroma -> retrieval -> LLM locale.
Tutto gira in locale: nessun dato lascia la macchina e non servono chiavi API.

L'indice è incrementale: a ogni avvio vengono reindicizzati solo i file
aggiunti, modificati o rimossi, confrontando l'hash del contenuto con un
manifest salvato accanto al database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === CONFIGURAZIONE ==========================================================
# I percorsi sono relativi alla posizione di questo file, così il progetto
# funziona da qualsiasi directory di lavoro. Tutto è sovrascrivibile via env.

PROJECT_ROOT = Path(__file__).resolve().parent

BASE_PATH = Path(os.getenv("UNIBOT_DOCS_DIR", PROJECT_ROOT / "Appunti"))
DB_DIR = Path(os.getenv("UNIBOT_DB_DIR", PROJECT_ROOT / "chroma_db_uni"))

EMBEDDING_MODEL = os.getenv(
    "UNIBOT_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LLM_MODEL = os.getenv("UNIBOT_LLM_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("UNIBOT_LLM_TEMPERATURE", "0.01"))

_raw_host = os.getenv("UNIBOT_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "localhost:11434"
OLLAMA_HOST = _raw_host if _raw_host.startswith("http") else f"http://{_raw_host}"

# Avvio automatico di Ollama: disattivato per default. Far partire un demone
# in background senza che l'utente lo chieda è un effetto collaterale che
# non spetta a questa app decidere; l'interfaccia offre un pulsante esplicito.
AUTOSTART_OLLAMA = os.getenv("UNIBOT_AUTOSTART_OLLAMA", "").lower() in {"1", "true", "yes"}

CHUNK_SIZE = int(os.getenv("UNIBOT_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("UNIBOT_CHUNK_OVERLAP", "150"))

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
MANIFEST_NAME = ".index_manifest.json"


# === MANIFEST DELL'INDICE ====================================================

def _manifest_path() -> Path:
    return DB_DIR / MANIFEST_NAME


def _load_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        print("[INDEX] Manifest illeggibile, verrà rigenerato.")
        return {}


def _save_manifest(manifest: dict) -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(_manifest_path(), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _file_digest(path: Path) -> str:
    """SHA-256 del contenuto, letto a blocchi per non caricare file grandi in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# === CARICAMENTO DOCUMENTI ===================================================

def ensure_base_path() -> None:
    BASE_PATH.mkdir(parents=True, exist_ok=True)


def _iter_source_files():
    """Tutti i file indicizzabili sotto BASE_PATH, in ordine deterministico."""
    ensure_base_path()
    for path in sorted(BASE_PATH.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _load_file(path: Path) -> list:
    """Carica un singolo file e arricchisce i metadati di ogni pagina."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    rel = path.relative_to(BASE_PATH).as_posix()
    for doc in docs:
        doc.metadata["file_name"] = path.name
        doc.metadata["file_name_noext"] = path.stem.lower()
        doc.metadata["rel_path"] = rel
    return docs


def _split(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# === COSTRUZIONE E SINCRONIZZAZIONE DELL'INDICE ==============================

def _sync_index(vectordb: Chroma) -> dict:
    """
    Allinea l'indice ai file presenti su disco.

    Reindicizza solo ciò che è cambiato: confronta l'hash di ogni file con
    quello registrato nel manifest, e usa gli id dei chunk salvati per
    rimuovere in modo mirato le vecchie versioni.
    """
    manifest = _load_manifest()
    on_disk = {p.relative_to(BASE_PATH).as_posix(): p for p in _iter_source_files()}

    removed = [rel for rel in manifest if rel not in on_disk]
    added, changed = [], []
    for rel, path in on_disk.items():
        entry = manifest.get(rel)
        if entry is None:
            added.append(rel)
        elif entry.get("sha256") != _file_digest(path):
            changed.append(rel)

    if not (removed or added or changed):
        print(f"[INDEX] Indice già aggiornato ({len(on_disk)} file).")
        return manifest

    print(
        f"[INDEX] Modifiche rilevate — nuovi: {len(added)}, "
        f"modificati: {len(changed)}, rimossi: {len(removed)}."
    )

    # 1) Rimuovo i chunk dei file spariti o cambiati.
    stale_ids = []
    for rel in removed + changed:
        stale_ids.extend(manifest.get(rel, {}).get("ids", []))
    if stale_ids:
        print(f"[INDEX] Rimuovo {len(stale_ids)} chunk obsoleti...")
        vectordb.delete(ids=stale_ids)
    for rel in removed:
        manifest.pop(rel, None)

    # 2) Indicizzo i file nuovi e quelli modificati.
    for rel in added + changed:
        path = on_disk[rel]
        print(f"[INDEX] Indicizzo {rel} ...")
        digest = _file_digest(path)
        chunks = _split(_load_file(path))
        if not chunks:
            print(f"[INDEX] Nessun testo estratto da {rel}, lo salto.")
            manifest[rel] = {"sha256": digest, "ids": []}
            continue

        ids = [f"{rel}::{digest[:12]}::{i}" for i in range(len(chunks))]
        vectordb.add_documents(documents=chunks, ids=ids)
        manifest[rel] = {"sha256": digest, "ids": ids}
        print(f"[INDEX] {rel}: {len(chunks)} chunk.")

    _save_manifest(manifest)
    print("[INDEX] Indice aggiornato.")
    return manifest


def build_vectordb(rebuild: bool = False) -> Chroma:
    """
    Apre il database vettoriale e lo allinea ai documenti presenti su disco.

    rebuild=True cancella l'indice e lo ricostruisce da zero. Serve solo se si
    cambia modello di embedding o parametri di chunking: le modifiche ai file
    vengono già gestite in modo incrementale.
    """
    ensure_base_path()

    if rebuild and DB_DIR.exists():
        print(f"[BACKEND] Cancello l'indice esistente in {DB_DIR} ...")
        shutil.rmtree(DB_DIR)

    # Un indice senza manifest proviene da una versione precedente: non sappiamo
    # quali chunk contenga, quindi lo ricostruiamo una volta sola per evitare duplicati.
    if DB_DIR.exists() and not _manifest_path().exists():
        print("[BACKEND] Indice legacy senza manifest: lo ricostruisco una volta.")
        shutil.rmtree(DB_DIR)

    print(f"[BACKEND] Carico il modello di embedding ({EMBEDDING_MODEL})...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectordb = Chroma(
        embedding_function=embedding_fn,
        persist_directory=str(DB_DIR),
    )

    _sync_index(vectordb)
    return vectordb


class OllamaUnavailable(RuntimeError):
    """Ollama non è raggiungibile, oppure il modello richiesto non è scaricato."""


def check_ollama() -> tuple[bool, str]:
    """
    Verifica che Ollama risponda e che il modello configurato sia disponibile.

    Il client di LangChain non contatta il server alla costruzione, ma solo alla
    prima invocazione: senza questo controllo l'app partirebbe normalmente per
    poi fallire alla prima domanda, con un errore di rete poco leggibile.

    Restituisce (ok, messaggio_per_l_utente).
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read())
    except Exception:
        return False, (
            f"Ollama non risponde su {OLLAMA_HOST}.\n\n"
            "Avvialo (`ollama serve`, oppure apri l'applicazione Ollama) e ricarica la pagina."
        )

    available = [model.get("name", "") for model in payload.get("models", [])]
    wanted = LLM_MODEL.split(":")[0]
    if not any(name == LLM_MODEL or name.split(":")[0] == wanted for name in available):
        listed = ", ".join(sorted(available)) or "nessuno"
        return False, (
            f"Ollama è attivo ma il modello '{LLM_MODEL}' non è scaricato.\n\n"
            f"Esegui: `ollama pull {LLM_MODEL}`\n\n"
            f"Modelli attualmente disponibili: {listed}."
        )

    return True, f"Ollama attivo — modello '{LLM_MODEL}' pronto."


def start_ollama(timeout: float = 30.0) -> tuple[bool, str]:
    """
    Avvia `ollama serve` in background e attende che risponda.

    Pensata per essere chiamata su richiesta esplicita dell'utente, non
    all'avvio: il processo sopravvive a questa applicazione e non le appartiene.
    Non scarica modelli — quello resta una decisione dell'utente, sono gigabyte.
    """
    if check_ollama()[0]:
        return True, "Ollama era già attivo."

    executable = shutil.which("ollama")
    if executable is None:
        return False, (
            "Il comando `ollama` non è nel PATH: Ollama non sembra installato.\n\n"
            "Scaricalo da https://ollama.com e riavvia il terminale."
        )

    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        # Stacca il processo dalla console, così non muore con questa app.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    print("[BACKEND] Avvio `ollama serve` in background...")
    try:
        subprocess.Popen([executable, "serve"], **kwargs)
    except OSError as exc:
        return False, f"Non sono riuscito ad avviare Ollama: {exc}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(1.0)
        ok, message = check_ollama()
        if ok:
            return True, message

    return False, (
        f"Ollama è stato avviato ma non risponde entro {timeout:.0f} secondi. "
        "Controlla il terminale, oppure avvialo a mano con `ollama serve`."
    )


def init_llm(verify: bool = True) -> OllamaLLM:
    """
    Inizializza il modello locale servito da Ollama.

    Con verify=True controlla subito la disponibilità del server e del modello,
    sollevando OllamaUnavailable con un messaggio azionabile invece di lasciare
    che il problema emerga alla prima domanda.
    """
    if verify:
        ok, message = check_ollama()
        if not ok and AUTOSTART_OLLAMA:
            print("[BACKEND] Ollama non risponde, provo ad avviarlo (UNIBOT_AUTOSTART_OLLAMA).")
            ok, message = start_ollama()
        if not ok:
            raise OllamaUnavailable(message)
        print(f"[BACKEND] {message}")

    return OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# === RETRIEVAL E RISPOSTA ====================================================

def _extract_file_hint(query: str) -> str | None:
    """
    Riconosce un riferimento esplicito a un file nella domanda, per restringere
    la ricerca a quel documento.

      "nel file CSEMAS ..."   -> "csemas"
      "cerca in CSEMAS.pdf"   -> "csemas"

    Restituisce None se la domanda non nomina alcun file.
    """
    match = re.search(r"file\s+([a-z0-9_.-]+)", query.lower())
    if match:
        name = match.group(1)
        return name[:-4] if name.endswith(".pdf") else name

    caps = re.findall(r"\b[A-Z0-9]{3,}\b", query)
    return caps[0].lower() if caps else None


PROMPT_TEMPLATE = """Sei un assistente che aiuta a studiare usando SOLO il seguente contesto,
che è preso dai miei appunti universitari. Rispondi in italiano chiaro, conciso ma completo.
Se qualcosa non è nel contesto, dillo esplicitamente.

Contesto:
{context}

Domanda: {query}

Risposta (in italiano):
"""


def answer_question(query: str, vectordb: Chroma, llm, k: int = 4):
    """
    Retrieval + generazione. Restituisce (risposta, documenti_usati).

    Se la domanda nomina un file, la ricerca viene prima ristretta a quel
    documento; se non produce risultati, si ricade sull'intero indice.
    """
    docs = []
    file_hint = _extract_file_hint(query)

    if file_hint:
        print(f"[BACKEND] Restringo la ricerca al file: {file_hint}")
        try:
            docs = vectordb.similarity_search(
                query, k=k, filter={"file_name_noext": file_hint}
            )
        except (TypeError, ValueError) as exc:
            print(f"[BACKEND] Filtro per file non applicabile ({exc}).")

        if not docs:
            print("[BACKEND] Nessun risultato in quel file, cerco ovunque.")

    if not docs:
        docs = vectordb.similarity_search(query, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    return str(llm.invoke(prompt)), docs


# === DIAGNOSTICA =============================================================

def debug_print_vectordb_files(vectordb: Chroma) -> None:
    """Elenca i file presenti nell'indice e quanti chunk occupa ciascuno."""
    from collections import Counter

    print("\n[DEBUG] Leggo i metadati dall'indice...")
    try:
        data = vectordb.get(include=["metadatas"])
    except (AttributeError, TypeError):
        data = vectordb._collection.get(include=["metadatas"])

    metadatas = data.get("metadatas") or []
    if not metadatas:
        print("[DEBUG] Nessun metadato: l'indice sembra vuoto.")
        return

    counter = Counter(
        (md or {}).get("rel_path") or (md or {}).get("source", "sconosciuto")
        for md in metadatas
    )

    print(f"[DEBUG] {len(counter)} file nell'indice, {len(metadatas)} chunk totali:\n")
    for source, n_chunks in sorted(counter.items()):
        print(f"- {source}   ({n_chunks} chunk)")
