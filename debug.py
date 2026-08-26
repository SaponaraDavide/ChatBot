"""
Strumento di diagnostica dell'indice.

Uso:
    python debug.py                      # elenca i file indicizzati
    python debug.py "la mia domanda"     # mostra i chunk recuperati per una query
"""

import sys

from study_rag import build_vectordb, debug_print_vectordb_files


def preview_retrieval(vectordb, query: str, k: int = 5) -> None:
    docs = vectordb.similarity_search(query, k=k)
    print(f"\nTrovati {len(docs)} chunk simili a {query!r}:\n")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"=== CHUNK {i} ===")
        print("File:  ", meta.get("rel_path") or meta.get("source"))
        print("Pagina:", meta.get("page", "n/d"))
        print("Testo:\n", doc.page_content[:1000], "\n")


def main() -> None:
    vectordb = build_vectordb(rebuild=False)

    if len(sys.argv) > 1:
        preview_retrieval(vectordb, " ".join(sys.argv[1:]))
    else:
        debug_print_vectordb_files(vectordb)


if __name__ == "__main__":
    main()
