"""Interfaccia Streamlit del chatbot RAG sugli appunti universitari."""

import streamlit as st

from study_rag import (
    BASE_PATH,
    LLM_MODEL,
    OllamaUnavailable,
    answer_question,
    build_vectordb,
    check_ollama,
    start_ollama,
    init_llm,
)

st.set_page_config(page_title="UniBot — Chatbot sugli appunti", page_icon="🎓")
st.title("🎓 Chatbot per studiare con i tuoi appunti")

# --- SIDEBAR: upload e controlli -------------------------------------------
with st.sidebar:
    st.header("Impostazioni")

    st.markdown("### Aggiungi documenti")
    uploaded_files = st.file_uploader(
        "Carica PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    save_uploaded = st.button("📥 Salva e indicizza")

    st.markdown("---")
    rebuild_db = st.button(
        "🔁 Ricostruisci l'indice da zero",
        help=(
            "Non serve per aggiungere o modificare documenti: quelli vengono "
            "rilevati e reindicizzati da soli. Usalo solo dopo aver cambiato "
            "modello di embedding o parametri di chunking."
        ),
    )

    st.markdown("### Stato")
    st.markdown(f"Cartella documenti: `{BASE_PATH.name}`")
    ollama_ok, ollama_msg = check_ollama()
    if ollama_ok:
        st.success(f"`{LLM_MODEL}` pronto")
    else:
        st.warning(ollama_msg)
        if st.button("▶ Avvia Ollama"):
            with st.spinner("Avvio Ollama e attendo che risponda..."):
                started, detail = start_ollama()
            if started:
                st.rerun()
            else:
                st.error(detail)

# --- Salvataggio dei file caricati ------------------------------------------
if save_uploaded and uploaded_files:
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    for up_file in uploaded_files:
        (BASE_PATH / up_file.name).write_bytes(up_file.getbuffer())
    st.success(f"{len(uploaded_files)} file salvati. Aggiorno l'indice...")
    st.session_state.pop("vectordb", None)  # forza la risincronizzazione

# --- Indice vettoriale ------------------------------------------------------
if "vectordb" not in st.session_state:
    with st.spinner("Carico e sincronizzo l'indice..."):
        try:
            st.session_state["vectordb"] = build_vectordb(rebuild=False)
        except Exception as exc:
            st.error(f"Errore nell'apertura dell'indice: {exc}")
            st.stop()

if rebuild_db:
    with st.spinner("Ricostruisco l'indice da zero..."):
        try:
            st.session_state["vectordb"] = build_vectordb(rebuild=True)
        except Exception as exc:
            st.error(f"Errore nella ricostruzione dell'indice: {exc}")
            st.stop()

if "llm" not in st.session_state:
    with st.spinner(f"Mi connetto a Ollama ({LLM_MODEL})..."):
        try:
            st.session_state["llm"] = init_llm()
        except OllamaUnavailable as exc:
            st.error(str(exc))
            st.info(
                "L'indice dei documenti è già caricato: appena Ollama sarà attivo, "
                "ricarica la pagina e potrai fare domande senza reindicizzare nulla."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Non riesco a inizializzare il modello: {exc}")
            st.stop()

vectordb = st.session_state["vectordb"]
llm = st.session_state["llm"]

# --- Cronologia della chat --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Ciao! Fammi una domanda sui tuoi appunti 😊"}
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input e risposta -------------------------------------------------------
user_input = st.chat_input("Fai una domanda sui tuoi appunti...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Cerco nei tuoi appunti..."):
            try:
                answer, sources = answer_question(user_input, vectordb, llm)
            except Exception as exc:
                st.error(f"Errore durante la risposta: {exc}")
                st.stop()

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        if sources:
            with st.expander(f"Fonti utilizzate ({len(sources)} passaggi)"):
                for i, doc in enumerate(sources, start=1):
                    meta = doc.metadata
                    name = meta.get("rel_path") or meta.get("source", "sconosciuta")
                    page = meta.get("page")
                    label = f"[{i}] {name}"
                    if page is not None:
                        label += f" — pagina {page + 1}"
                    st.write(label)
