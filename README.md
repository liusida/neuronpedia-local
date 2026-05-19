# Neuronpedia Local

Unofficial local prompt explorer for SAE activations, using Neuronpedia annotations.

This is a standalone version of the SAE probe page from the ICA probe server. It runs a local FastAPI server, loads Neuronpedia's public model/source catalog, probes arbitrary text through Neuronpedia's inference API, and links/labels active features with Neuronpedia annotations when available.

## Quick start

```bash
uv sync
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

Models and layers/sources come from Neuronpedia's public resources list. A `*` in the layer/source dropdown marks sources that have Neuronpedia annotation/dashboard data; disabled entries are present on Neuronpedia but do not currently support inference probing.

## Notes

- The model/source catalog is cached in memory and persisted in `data/cache.sqlite` for 24 hours.
- Probe requests are proxied to `https://www.neuronpedia.org/api/search-topk-by-token` and cached in SQLite for 24 hours by model, source, prompt, and probe settings.
- Individual feature annotations are also cached for 24 hours by model, source, and component so labels can be reused across prompts.
- No local model or SAE checkpoint download is needed.
