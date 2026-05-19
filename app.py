#!/usr/bin/env python3
"""Standalone prompt explorer for Neuronpedia SAE activations and annotations."""

from __future__ import annotations

import html
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
NEURONPEDIA_URL = "https://www.neuronpedia.org"
CATALOG_TTL_SECONDS = 60 * 60


@dataclass(frozen=True)
class SourceRow:
    model_id: str
    source_id: str
    inference_enabled: bool


class ProbeRequest(BaseModel):
    model_id: str
    source: str
    text: str = Field(..., min_length=1, max_length=4_000)
    top_k: int = Field(5, ge=1, le=20)
    ignore_bos: bool = True
    density_threshold: float = Field(0.9999, gt=0, lt=1)


def create_app() -> FastAPI:
    app = FastAPI(title="Neuronpedia Local Prompt Explorer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.catalog_cache = None
    app.state.catalog_cache_time = 0.0

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/neuronpedia/catalog")
    def neuronpedia_catalog(refresh: bool = False) -> dict[str, Any]:
        if not refresh and app.state.catalog_cache and time.time() - app.state.catalog_cache_time < CATALOG_TTL_SECONDS:
            return app.state.catalog_cache
        try:
            payload = build_catalog(fetch_text(f"{NEURONPEDIA_URL}/available-resources"))
        except RuntimeError as exc:
            if app.state.catalog_cache:
                return {**app.state.catalog_cache, "stale": True, "warning": str(exc)}
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        app.state.catalog_cache = payload
        app.state.catalog_cache_time = time.time()
        return payload

    @app.post("/api/neuronpedia/probe")
    def neuronpedia_probe(body: ProbeRequest) -> dict[str, Any]:
        try:
            payload = post_json(
                f"{NEURONPEDIA_URL}/api/search-topk-by-token",
                {
                    "modelId": body.model_id,
                    "source": body.source,
                    "text": body.text,
                    "numResults": body.top_k,
                    "ignoreBos": body.ignore_bos,
                    "densityThreshold": body.density_threshold,
                },
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return normalize_probe_payload(payload, model_id=body.model_id, source=body.source, top_k=body.top_k)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "neuronpedia-local/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="replace")
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        try:
            return curl_text(url)
        except RuntimeError as curl_exc:
            raise RuntimeError(f"Could not fetch {url}: {exc}; curl fallback failed: {curl_exc}") from exc


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "neuronpedia-local/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Neuronpedia returned HTTP {exc.code}: {detail[:500]}") from exc
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        try:
            return json.loads(curl_text(url, payload=payload))
        except (RuntimeError, json.JSONDecodeError) as curl_exc:
            raise RuntimeError(f"Could not call {url}: {exc}; curl fallback failed: {curl_exc}") from exc


def curl_text(url: str, *, payload: dict[str, Any] | None = None) -> str:
    cmd = ["curl", "-sS", "--max-time", "60", "-H", "User-Agent: neuronpedia-local/1.0"]
    if payload is not None:
        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json", "--data", json.dumps(payload)])
    cmd.append(url)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=65)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"curl exited {completed.returncode}").strip())
    return completed.stdout


def build_catalog(resources_html: str) -> dict[str, Any]:
    rows = parse_available_resource_rows(resources_html)
    if not rows:
        raise RuntimeError("Could not find Neuronpedia model/source rows in the available resources page.")

    by_model: dict[str, list[SourceRow]] = {}
    for row in rows:
        by_model.setdefault(row.model_id, []).append(row)

    models = []
    for model_id in sorted(by_model):
        sources = sorted(by_model[model_id], key=lambda row: source_sort_key(row.source_id))
        annotated_count = len(sources)
        inference_count = sum(1 for source in sources if source.inference_enabled)
        models.append(
            {
                "id": model_id,
                "label": format_model_label(model_id),
                "source_count": len(sources),
                "annotated_source_count": annotated_count,
                "inference_source_count": inference_count,
                "sources": [
                    {
                        "id": source.source_id,
                        "label": format_source_label(source.source_id),
                        "has_annotation": True,
                        "inference_enabled": source.inference_enabled,
                        "url": f"{NEURONPEDIA_URL}/{urllib.parse.quote(model_id)}/{urllib.parse.quote(source.source_id)}",
                    }
                    for source in sources
                ],
            }
        )

    return {
        "models": models,
        "model_count": len(models),
        "source_count": sum(model["source_count"] for model in models),
        "inference_source_count": sum(model["inference_source_count"] for model in models),
        "annotation_marker": "*",
        "fetched_at": int(time.time()),
    }


def parse_available_resource_rows(markup: str) -> list[SourceRow]:
    parser = AvailableResourcesParser()
    parser.feed(markup)
    rows = parser.rows
    if rows:
        return rows
    return parse_resource_rows_with_regex(markup)


class AvailableResourcesParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_td = False
        self.current_cells: list[str] = []
        self.current_text: list[str] = []
        self.rows: list[SourceRow] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_cells = []
        elif tag == "td":
            self.in_td = True
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.in_td:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_td:
            self.current_cells.append(html.unescape(" ".join(self.current_text)).strip())
            self.in_td = False
            self.current_text = []
        elif tag == "tr" and len(self.current_cells) >= 3:
            model_id, source_id, status = self.current_cells[:3]
            if model_id and source_id and model_id != "Model ID":
                self.rows.append(SourceRow(model_id=model_id, source_id=source_id, inference_enabled="✅" in status))


def parse_resource_rows_with_regex(markup: str) -> list[SourceRow]:
    rows: list[SourceRow] = []
    pattern = re.compile(
        r'href="https://www\.neuronpedia\.org/([^"/#?]+)"[^>]*>\s*([^<]+)</a>.*?'
        r'href="https://www\.neuronpedia\.org/([^"/#?]+)/([^"/#?]+)"[^>]*>\s*([^<]+)</a>.*?'
        r"(✅|❌)",
        re.DOTALL,
    )
    for match in pattern.finditer(markup):
        model_href, model_text, source_model_href, source_href, source_text, status = match.groups()
        model_id = html.unescape(urllib.parse.unquote(model_text or model_href)).strip()
        source_id = html.unescape(urllib.parse.unquote(source_text or source_href)).strip()
        if model_id and source_id and model_href == source_model_href:
            rows.append(SourceRow(model_id=model_id, source_id=source_id, inference_enabled=status == "✅"))
    return rows


def normalize_probe_payload(payload: dict[str, Any], *, model_id: str, source: str, top_k: int) -> dict[str, Any]:
    tokens = []
    for item in payload.get("results") or []:
        top = []
        for feature in item.get("topFeatures") or []:
            feature_blob = feature.get("feature") or {}
            explanations = feature_blob.get("explanations") or []
            label = ""
            if explanations and isinstance(explanations[0], dict):
                label = str(explanations[0].get("description") or "").strip()
            top.append(
                {
                    "component": int(feature.get("featureIndex", -1)),
                    "score": float(feature.get("activationValue", 0)),
                    "label": label,
                    "density": feature_blob.get("frac_nonzero"),
                    "source_url": f"{NEURONPEDIA_URL}/{urllib.parse.quote(model_id)}/{urllib.parse.quote(source)}/{feature.get('featureIndex')}",
                }
            )
        tokens.append(
            {
                "position": int(item.get("position", item.get("tokenPosition", 0))),
                "token": str(item.get("token", "")),
                "token_text": str(item.get("token", "")),
                "top": top,
            }
        )
    return {
        "model_name": model_id,
        "layer": source,
        "source": source,
        "top_k": int(top_k),
        "tokens": tokens,
        "seq_len": len(tokens),
        "truncated": False,
        "n_components": None,
    }


def format_model_label(model_id: str) -> str:
    family = model_id.split("-", maxsplit=1)[0].upper() if "-" in model_id else model_id.upper()
    return f"{model_id} ({family})"


def format_source_label(source_id: str) -> str:
    return source_id.replace("_", " ")


def source_sort_key(source_id: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)", source_id)
    if match:
        return (0, int(match.group(1)), source_id)
    if source_id.startswith("e-"):
        return (0, -1, source_id)
    return (1, 0, source_id)


app = create_app()
