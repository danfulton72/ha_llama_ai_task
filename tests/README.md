# Tests from the original ZIP

The original ZIP included `test_client_http.py` and `test_llama_cpp.py`, but referenced a `tests/hastub` fixture tree that was not present in the archive. They are retained here for reference and have had their import domain updated to `llama_cpp_ai_task`.

They are not currently wired into CI. The HACS and hassfest validation jobs cover repository/integration metadata; a proper Home Assistant pytest harness should replace these legacy stub-based tests before relying on them as regression tests.
