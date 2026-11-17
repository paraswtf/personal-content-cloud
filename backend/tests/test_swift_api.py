"""The iPhone app's generated API layer (ios/Sources/API/*.generated.swift) matches the server's routes and
schemas: scripts/gen_swift_api.py turns the OpenAPI description into Swift, and this test regenerates it
in memory and compares it with the committed files."""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "gen_swift_api.py"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_swift_api", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _generator()


@pytest.fixture(scope="module")
def openapi():
    from app.main import create_app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the HEAD twins of the file routes share an operation id
        return create_app().openapi()


def test_committed_swift_matches_the_server(gen, openapi):
    files = gen.generate(openapi)
    stale = []
    for name, text in files.items():
        path = gen.OUT_DIR / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(name)
    assert not stale, (
        f"{', '.join(stale)} out of date with the server's API: run "
        "`backend/.venv/bin/python scripts/gen_swift_api.py` from the repository root and commit the result"
    )


def test_output_is_stable(gen, openapi):
    # Two runs over the same description give the same text (sorted, no set order leaking through).
    assert gen.generate(openapi) == gen.generate(openapi)


def test_every_route_is_generated_or_listed(gen, openapi):
    endpoints = gen.generate(openapi)[gen.ENDPOINTS_FILE]
    header = endpoints.split("import Foundation", 1)[0]
    for path, methods in openapi["paths"].items():
        for method in methods:
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            listed = f"{method.upper()} {path}"
            assert f"/// {listed}\n" in endpoints or f"// - {listed}:" in header, f"{listed} neither generated nor listed"


def test_unknown_enum_values_and_unions_are_handled(gen):
    # A tiny description with the shapes the generator must handle beyond plain objects.
    spec = {
        "paths": {
            "/api/things/{thing_id}": {
                "get": {
                    "operationId": "get_thing_api_things__thing_id__get",
                    "parameters": [
                        {"name": "thing_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "in", "in": "query", "required": False,
                         "schema": {"type": "string", "enum": ["in", "out"]}},
                    ],
                    "responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Thing"}}}}},
                },
            },
        },
        "components": {
            "schemas": {
                "Thing": {
                    "type": "object",
                    "required": ["id", "box"],
                    "properties": {
                        "id": {"type": "integer"},
                        "box": {"anyOf": [{"$ref": "#/components/schemas/A"}, {"$ref": "#/components/schemas/B"}]},
                        "flag": {"type": "boolean", "default": False},
                        "point": {"type": "array", "prefixItems": [{"type": "number"}, {"type": "string"}]},
                        "extra": {"type": "object", "additionalProperties": True},
                    },
                },
                "A": {"type": "object", "required": ["a_only"], "properties": {"a_only": {"type": "integer"}}},
                "B": {"type": "object", "properties": {"b": {"type": "string"}}},
            },
        },
    }
    files = gen.generate(spec, hand_written=set())
    models, endpoints = files[gen.MODELS_FILE], files[gen.ENDPOINTS_FILE]
    assert "case unknown(String)" in models
    assert "case `in`" in models
    assert "enum AOrB: Codable, Hashable, Sendable" in models
    assert 'keys.contains(AnyKey("a_only"))' in models
    assert "var flag: Bool = false" in models  # an answer's plain default is not optional
    assert "?? false" in models
    assert "struct ThingPointTuple" in models
    assert "var extra: [String: JSONValue]?" in models
    assert "nonisolated func getThing(thingId: Int, `in`: API.In? = nil) async throws -> API.Thing" in endpoints
    assert 'query.add("in", `in`)' in endpoints
