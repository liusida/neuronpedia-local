#!/usr/bin/env python3
"""Standalone prompt explorer for Neuronpedia SAE activations and annotations."""

from __future__ import annotations

import html
import hashlib
import json
import re
import sqlite3
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
DATA_DIR = ROOT / "data"
CACHE_DB_PATH = DATA_DIR / "cache.sqlite"
NEURONPEDIA_URL = "https://www.neuronpedia.org"
CACHE_TTL_SECONDS = 24 * 60 * 60
CATALOG_CACHE_KEY = "catalog:v1"


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
    app.state.cache_db_path = CACHE_DB_PATH
    init_cache_db(app.state.cache_db_path)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/neuronpedia/catalog")
    def neuronpedia_catalog(refresh: bool = False) -> dict[str, Any]:
        if not refresh and app.state.catalog_cache and time.time() - app.state.catalog_cache_time < CACHE_TTL_SECONDS:
            return app.state.catalog_cache
        if not refresh:
            cached_payload = cache_get(app.state.cache_db_path, CATALOG_CACHE_KEY)
            if cached_payload:
                app.state.catalog_cache = cached_payload
                app.state.catalog_cache_time = time.time()
                return cached_payload
        try:
            payload = build_catalog(fetch_text(f"{NEURONPEDIA_URL}/available-resources"))
        except RuntimeError as exc:
            if app.state.catalog_cache:
                return {**app.state.catalog_cache, "stale": True, "warning": str(exc)}
            stale_payload = cache_get(app.state.cache_db_path, CATALOG_CACHE_KEY, allow_expired=True)
            if stale_payload:
                return {**stale_payload, "stale": True, "warning": str(exc)}
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        app.state.catalog_cache = payload
        app.state.catalog_cache_time = time.time()
        cache_set(app.state.cache_db_path, CATALOG_CACHE_KEY, payload, ttl_seconds=CACHE_TTL_SECONDS)
        return payload

    @app.post("/api/neuronpedia/probe")
    def neuronpedia_probe(body: ProbeRequest) -> dict[str, Any]:
        request_payload = {
            "modelId": body.model_id,
            "source": body.source,
            "text": body.text,
            "numResults": body.top_k,
            "ignoreBos": body.ignore_bos,
            "densityThreshold": body.density_threshold,
        }
        cache_key = probe_cache_key(request_payload)
        cached_payload = cache_get(app.state.cache_db_path, cache_key)
        if cached_payload:
            return {**hydrate_feature_annotations(app.state.cache_db_path, cached_payload), "cached": True}
        try:
            payload = post_json(f"{NEURONPEDIA_URL}/api/search-topk-by-token", request_payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        normalized_payload = normalize_probe_payload(payload, model_id=body.model_id, source=body.source, top_k=body.top_k)
        cache_feature_annotations(app.state.cache_db_path, normalized_payload, ttl_seconds=CACHE_TTL_SECONDS)
        normalized_payload = hydrate_feature_annotations(app.state.cache_db_path, normalized_payload)
        cache_set(app.state.cache_db_path, cache_key, normalized_payload, ttl_seconds=CACHE_TTL_SECONDS)
        return normalized_payload

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def init_cache_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at ON cache_entries(expires_at)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_annotations (
                model_id TEXT NOT NULL,
                source TEXT NOT NULL,
                component INTEGER NOT NULL,
                label TEXT NOT NULL,
                density_json TEXT,
                source_url TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                PRIMARY KEY (model_id, source, component)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_annotations_expires_at ON feature_annotations(expires_at)")


def cache_get(path: Path, key: str, *, allow_expired: bool = False) -> dict[str, Any] | None:
    now = int(time.time())
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT value_json, expires_at FROM cache_entries WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        value_json, expires_at = row
        if int(expires_at) < now and not allow_expired:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            return None
    try:
        value = json.loads(str(value_json))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def cache_set(path: Path, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
    now = int(time.time())
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries (key, value_json, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, json.dumps(value, separators=(",", ":"), sort_keys=True), now, now + ttl_seconds),
        )


def probe_cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"probe:v1:{hashlib.sha256(blob).hexdigest()}"


def cache_feature_annotations(path: Path, probe_payload: dict[str, Any], *, ttl_seconds: int) -> None:
    model_id = str(probe_payload.get("model_name") or "")
    source = str(probe_payload.get("source") or probe_payload.get("layer") or "")
    if not model_id or not source:
        return

    annotations: dict[int, tuple[str, Any, str]] = {}
    for token in probe_payload.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        for feature in token.get("top") or []:
            if not isinstance(feature, dict):
                continue
            try:
                component = int(feature.get("component", -1))
            except (TypeError, ValueError):
                continue
            label = str(feature.get("label") or "").strip()
            if component < 0 or not label:
                continue
            annotations[component] = (label, feature.get("density"), str(feature.get("source_url") or ""))

    if not annotations:
        return

    now = int(time.time())
    rows = [
        (
            model_id,
            source,
            component,
            label,
            json.dumps(density, separators=(",", ":"), sort_keys=True) if density is not None else None,
            source_url,
            now,
            now,
            now + ttl_seconds,
        )
        for component, (label, density, source_url) in annotations.items()
    ]
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO feature_annotations (
                model_id, source, component, label, density_json, source_url, created_at, updated_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id, source, component) DO UPDATE SET
                label = excluded.label,
                density_json = excluded.density_json,
                source_url = excluded.source_url,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            """,
            rows,
        )


def hydrate_feature_annotations(path: Path, probe_payload: dict[str, Any]) -> dict[str, Any]:
    model_id = str(probe_payload.get("model_name") or "")
    source = str(probe_payload.get("source") or probe_payload.get("layer") or "")
    if not model_id or not source:
        return probe_payload

    component_set: set[int] = set()
    for token in probe_payload.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        for feature in token.get("top") or []:
            if not isinstance(feature, dict):
                continue
            try:
                component = int(feature.get("component", -1))
            except (TypeError, ValueError):
                continue
            if component >= 0:
                component_set.add(component)
    components = sorted(component_set)
    if not components:
        return probe_payload

    now = int(time.time())
    placeholders = ",".join("?" for _ in components)
    params: list[Any] = [model_id, source, now, *components]
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT component, label, density_json, source_url
            FROM feature_annotations
            WHERE model_id = ? AND source = ? AND expires_at >= ? AND component IN ({placeholders})
            """,
            params,
        ).fetchall()

    annotations: dict[int, dict[str, Any]] = {}
    for component, label, density_json, source_url in rows:
        density = None
        if density_json is not None:
            try:
                density = json.loads(str(density_json))
            except json.JSONDecodeError:
                density = None
        annotations[int(component)] = {"label": str(label), "density": density, "source_url": str(source_url)}

    if not annotations:
        return probe_payload

    for token in probe_payload.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        for feature in token.get("top") or []:
            if not isinstance(feature, dict):
                continue
            try:
                component = int(feature.get("component", -1))
            except (TypeError, ValueError):
                continue
            cached = annotations.get(component)
            if not cached:
                continue
            if not str(feature.get("label") or "").strip():
                feature["label"] = cached["label"]
            if feature.get("density") is None:
                feature["density"] = cached["density"]
            if not feature.get("source_url"):
                feature["source_url"] = cached["source_url"]
    return probe_payload


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
        categories = sorted(
            {source_category(source.source_id) for source in sources},
            key=source_category_sort_key,
        )
        models.append(
            {
                "id": model_id,
                "label": format_model_label(model_id),
                "source_count": len(sources),
                "annotated_source_count": annotated_count,
                "inference_source_count": inference_count,
                "categories": [
                    {
                        "id": category,
                        "label": format_category_label(category),
                        "source_count": sum(1 for source in sources if source_category(source.source_id) == category),
                        "inference_source_count": sum(
                            1 for source in sources if source_category(source.source_id) == category and source.inference_enabled
                        ),
                    }
                    for category in categories
                ],
                "sources": [
                    {
                        "id": source.source_id,
                        "label": format_source_label(source.source_id),
                        "layer_label": format_layer_label(source.source_id),
                        "category": source_category(source.source_id),
                        "category_label": format_category_label(source_category(source.source_id)),
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


def format_layer_label(source_id: str) -> str:
    if source_id.startswith("e-"):
        return "embedding"
    layer = source_layer(source_id)
    return f"layer {layer}" if layer is not None else source_id


def source_category(source_id: str) -> str:
    if re.match(r"^\d+$", source_id):
        return "default"
    match = re.match(r"^(?:\d+|e)-(.+)$", source_id)
    return match.group(1) if match else source_id


def format_category_label(category: str) -> str:
    return category.replace("_", " ")


def source_layer(source_id: str) -> int | None:
    match = re.match(r"^(\d+)", source_id)
    return int(match.group(1)) if match else None


def source_category_sort_key(category: str) -> tuple[int, str]:
    preferred_order = {
        "res-jb": 0,
        "resid": 1,
        "res": 2,
        "res_post": 3,
        "res_mid": 4,
        "mlp": 10,
        "att": 20,
        "gemmascope-res-16k": 30,
        "gemmascope-mlp-16k": 31,
        "gemmascope-att-16k": 32,
        "default": 1000,
    }
    return (preferred_order.get(category, 100), category)


def source_sort_key(source_id: str) -> tuple[int, str, int, str]:
    category = source_category(source_id)
    layer = source_layer(source_id)
    if layer is not None:
        return (*source_category_sort_key(category), layer, source_id)
    if source_id.startswith("e-"):
        return (*source_category_sort_key(category), -1, source_id)
    return (*source_category_sort_key(category), 0, source_id)


app = create_app()
