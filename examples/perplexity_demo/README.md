# Perplexity-style citation demo

This example pairs a FastAPI backend with a minimal HTML UI to show how
Cite-Right aligns answer spans to source text.

## What it shows
- FastAPI endpoint at `/api/citations` that runs `align_citations` on a static
  DeepSeek mHC excerpt.
- HTML page at `/` that displays the question, answer, and footnote-style
  citations with source context, plus a selection-driven "Check sources" pane.
  The right pane highlights evidence with a few characters of surrounding text.

## Run the demo
From the project root:

```bash
uv venv --python 3.11
uv pip install -r examples/perplexity_demo/requirements.txt
uv pip install -e .
uv run uvicorn examples.perplexity_demo.app:app --reload --port 8000
```

Then open http://localhost:8000 to see the UI.

Optional (for tighter span segmentation):

```bash
uv pip install "cite-right[spacy,pysbd]"
uv run python -m spacy download en_core_web_sm
```

## Testing the pipeline without a server
You can print the payload directly:

```bash
uv run python -m examples.perplexity_demo.app
```
