# Your knowledge base goes here

This folder ships empty on purpose - there is no bundled demo corpus.
Drop your own `.txt` files here (any plain-text documents you want the
pipeline to answer questions about), then run the app.

Only `*.txt` files are picked up (this `README.md` is ignored), and
nothing is indexed until you put something here or upload files through
the Streamlit sidebar.

**Nothing persists between runs.** Every time you start `cli.py` or
`streamlit run app_streamlit.py`, the index is rebuilt from scratch, in
memory, from whatever `.txt` files are in this folder at that moment -
there is no saved/cached index file to go stale or to accidentally carry
old data into a new session. Add or remove files here, restart, and the
pipeline reflects exactly what's currently in this folder - nothing more.
