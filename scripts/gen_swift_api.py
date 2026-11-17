#!/usr/bin/env python3
"""Generate the iPhone app's API layer from the server's OpenAPI description.

    backend/.venv/bin/python scripts/gen_swift_api.py           # writes ios/Sources/API/*.generated.swift
    backend/.venv/bin/python scripts/gen_swift_api.py --check   # exits 1 when the committed files are stale

It builds the FastAPI app (no models, a throwaway data folder), reads `app.openapi()` and writes two files:

- Models.generated.swift: every component schema as a Swift type inside the `API` namespace. Objects become
  `Codable, Hashable, Sendable` structs with lowerCamelCase properties and explicit CodingKeys. A property is
  optional when the schema does not require it or allows null. In answers (schemas only the server sends), a
  property the server fills with a plain default (false, 0, "", [], {}) is not optional: it decodes with that
  default when missing. In request bodies a property left nil is not sent, so the server applies its default;
  `nullKeys` lists the properties to send as an explicit null (the server tells "clear it" from "leave it").
  String literals become enums with an `unknown(String)` case, so a value a newer server adds never breaks
  decoding. A union of objects becomes an enum that picks its case by the keys only one option requires.
  Anything else (free-form objects, mixed unions) is `API.JSONValue`.
- Endpoints.generated.swift: one `nonisolated` async method per operation on `APIClient`, named after the
  route's Python function. Path and query parameters are typed arguments (optional ones default to nil, so the
  server's defaults apply), the JSON body is the generated body type, and the answer is the generated answer
  type, `Void` without a body, or `Data` for files. File routes also get a `...Request` builder that returns a
  signed `URLRequest` for image loaders. The transport they call lives in APIClient+Core.swift (hand-written).

Operations it cannot express are listed in the endpoints file's header with the reason (multipart forms,
raw bodies). Run the backend test tests/test_swift_api.py to see whether the committed files are current.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "ios" / "Sources" / "API"
MODELS_FILE = "Models.generated.swift"
ENDPOINTS_FILE = "Endpoints.generated.swift"
HAND_WRITTEN_CLIENT = REPO / "ios" / "Sources" / "Services" / "APIClient.swift"

# Routes whose body is not described by the schema: the server reads the raw request stream.
RAW_BODY_ROUTES = {
    ("put", "/api/uploads/{upload_id}/chunk"): "raw chunk body; the background uploader sends it from a file",
}

# Names the generated code must not take: Swift keywords (escaped with backticks where they are used) ...
SWIFT_KEYWORDS = {
    "associatedtype", "class", "deinit", "enum", "extension", "fileprivate", "func", "import", "init", "inout",
    "internal", "let", "open", "operator", "private", "precedencegroup", "protocol", "public", "rethrows",
    "static", "struct", "subscript", "typealias", "var", "break", "case", "catch", "continue", "default", "defer",
    "do", "else", "fallthrough", "for", "guard", "if", "in", "repeat", "return", "throw", "switch", "where",
    "while", "as", "false", "is", "nil", "self", "super", "throws", "true", "try", "await", "async", "any",
    "some", "Self", "Type", "Protocol",
}
# ... and type names inside the API namespace that would hide a Swift type the generated code uses.
RESERVED_TYPE_NAMES = {
    "Any", "AnyObject", "AnyKey", "Array", "Bool", "Character", "Codable", "CodingKey", "CodingKeys", "Data",
    "Date", "Decodable", "Decoder", "Dictionary", "Double", "Encodable", "Encoder", "Error", "Float", "Hashable",
    "Int", "Int64", "JSONValue", "Never", "Optional", "Protocol", "RawRepresentable", "Result", "Self",
    "Sendable", "Set", "String", "Type", "UInt", "URL", "URLRequest", "Void", "CaseIterable",
    # Standard library and Foundation names a hand-written `extension API` might use.
    "Range", "ClosedRange", "Task", "Element", "Index", "Key", "Value", "Collection", "Sequence", "Comparable",
    "Equatable", "Identifiable", "Substring", "UUID", "Decimal", "Locale", "Calendar", "TimeInterval",
    "Notification", "JSONDecoder", "JSONEncoder", "URLQueryItem", "URLComponents", "DecodingError",
    "EncodingError", "CGFloat", "Float", "Unicode", "Mirror", "API",
}
# Names of the transport in APIClient+Core.swift that generated methods must not take.
CORE_NAMES = {"perform", "performVoid", "performData", "urlRequest", "connection", "mediaURL", "mediaRequest",
              "mediaHeaders"}

MISSING = object()


# ----------------------------------------------------------------------------------------------- naming
def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[^0-9A-Za-z]+", name) if w]


def lower_camel(name: str) -> str:
    ws = _words(name)
    if not ws:
        return "empty" if name == "" else "value"
    out = ws[0][0].lower() + ws[0][1:] + "".join(w[0].upper() + w[1:] for w in ws[1:])
    return "n" + out if out[0].isdigit() else out


def upper_camel(name: str) -> str:
    ws = _words(name)
    if not ws:
        return "Value"
    out = "".join(w[0].upper() + w[1:] for w in ws)
    return "N" + out if out[0].isdigit() else out


def escape(ident: str) -> str:
    return f"`{ident}`" if ident in SWIFT_KEYWORDS else ident


def swift_string(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) > 0x7E:
            out.append(f"\\u{{{ord(ch):x}}}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def doc_line(text: str | None, limit: int = 160) -> str | None:
    """The first sentence of a docstring or description, on one line, cut at a word near `limit`."""
    if not text:
        return None
    flat = " ".join(text.split())
    m = re.match(r"(.+?[.!?])(\s|$)", flat)
    first = m.group(1) if m else flat
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0] + " ..."
    return first


def unique(name: str, taken: set[str]) -> str:
    if name not in taken:
        taken.add(name)
        return name
    n = 2
    while f"{name}{n}" in taken:
        n += 1
    taken.add(f"{name}{n}")
    return f"{name}{n}"


def ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def refs_in(x: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "$ref" and isinstance(v, str):
                found.add(ref_name(v))
            else:
                found |= refs_in(v)
    elif isinstance(x, list):
        for v in x:
            found |= refs_in(v)
    return found


# --------------------------------------------------------------------------------------------- generator
class Generator:
    def __init__(self, spec: dict[str, Any], hand_written_methods: set[str] | None = None) -> None:
        self.spec = spec
        self.schemas: dict[str, dict[str, Any]] = spec.get("components", {}).get("schemas", {})
        self.hand_written = set(hand_written_methods or ())
        self.type_names: dict[str, str] = {}
        taken: set[str] = set()
        for name in sorted(self.schemas):
            swift = upper_camel(name) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else name
            if swift in RESERVED_TYPE_NAMES or swift in SWIFT_KEYWORDS:
                swift += "Schema"
            self.type_names[name] = unique(swift, taken)
        self.taken_types = taken
        self.enum_names: dict[tuple[str, ...], str] = {}
        self.enum_values: dict[tuple[str, ...], list[str]] = {}
        self.extra_types: dict[str, str] = {}  # name -> rendered declaration (tuples, unions, inline objects)
        self.extra_after: dict[str, str] = {}  # name -> rendered extension outside the namespace
        self.operations = self._operations()
        self._name_enums()
        self.inputs, self.outputs = self._directions()

    # ------------------------------------------------------------------ operations
    def _operations(self) -> list[dict[str, Any]]:
        ops = []
        for path, methods in self.spec.get("paths", {}).items():
            for method, op in methods.items():
                if method not in ("get", "post", "put", "patch", "delete"):
                    continue
                ops.append({"path": path, "method": method, "op": op, "py": self._py_name(op, path, method)})
        ops.sort(key=lambda o: (o["path"], o["method"]))
        return ops

    @staticmethod
    def _py_name(op: dict[str, Any], path: str, method: str) -> str:
        """FastAPI's operation id is <function><path>_<method> with non-word characters as underscores. A route
        serving GET and HEAD takes either method in its id (the first of a set), so any method is stripped."""
        oid = op.get("operationId") or ""
        for verb in ("get", "head", "post", "put", "patch", "delete", "options"):
            suffix = re.sub(r"\W", "_", path) + "_" + verb
            if oid.endswith(suffix) and len(oid) > len(suffix):
                return oid[: -len(suffix)]
        return oid or re.sub(r"\W", "_", f"{method}{path}")

    # ------------------------------------------------------------------ enums
    def _walk_enums(self, x: Any, owner: str, prop: str, uses: dict[tuple[str, ...], list[tuple[str, str]]],
                    values: dict[tuple[str, ...], list[str]]) -> None:
        if not isinstance(x, dict):
            return
        if "enum" in x and all(isinstance(v, str) for v in x["enum"]) and x["enum"]:
            key = tuple(sorted(set(x["enum"])))
            uses[key].append((owner, prop))
            values.setdefault(key, list(dict.fromkeys(x["enum"])))
            return
        for k in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for sub in x.get(k, []) or []:
                self._walk_enums(sub, owner, prop, uses, values)
        if isinstance(x.get("items"), dict):
            self._walk_enums(x["items"], owner, prop, uses, values)
        if isinstance(x.get("additionalProperties"), dict):
            self._walk_enums(x["additionalProperties"], owner, prop, uses, values)
        for pname, ps in (x.get("properties") or {}).items():
            self._walk_enums(ps, owner, pname, uses, values)

    def _name_enums(self) -> None:
        """Each distinct set of string values becomes one enum. It is named after the property that most often
        carries it ("item_kind" -> ItemKind); when two sets want one name, the more used set keeps it and the
        other is prefixed with its owner (ListPeopleSort)."""
        uses: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
        values: dict[tuple[str, ...], list[str]] = {}
        for name in sorted(self.schemas):
            self._walk_enums(self.schemas[name], name, "", uses, values)
        for o in self.operations:
            owner = upper_camel(o["py"])
            for p in o["op"].get("parameters", []) or []:
                self._walk_enums(p.get("schema", {}), owner, p["name"], uses, values)
            for content in ((o["op"].get("requestBody") or {}).get("content") or {}).values():
                self._walk_enums(content.get("schema", {}), owner, "body", uses, values)
            for r in (o["op"].get("responses") or {}).values():
                for content in (r.get("content") or {}).values():
                    self._walk_enums(content.get("schema", {}), owner, "answer", uses, values)
        self.enum_values = values

        def best_prop(key: tuple[str, ...]) -> str:
            counts = Counter(prop for _, prop in uses[key])
            return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        by_name: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for key in uses:
            by_name[upper_camel(best_prop(key))].append(key)
        for name in sorted(by_name):
            keys = sorted(by_name[name], key=lambda k: (-len(uses[k]), k))
            for i, key in enumerate(keys):
                if i == 0 and name not in self.taken_types and name not in RESERVED_TYPE_NAMES \
                        and name not in SWIFT_KEYWORDS:
                    self.taken_types.add(name)
                    self.enum_names[key] = name
                    continue
                prop = best_prop(key)
                owner = sorted(o for o, p in uses[key] if p == prop)[0]
                candidate = upper_camel(owner) + upper_camel(prop)
                if candidate in RESERVED_TYPE_NAMES or candidate in SWIFT_KEYWORDS:
                    candidate += "Value"
                self.enum_names[key] = unique(candidate, self.taken_types)

    def enum_type(self, values: list[str]) -> str:
        return self.enum_names[tuple(sorted(set(values)))]

    # ------------------------------------------------------------------ input or output
    def _directions(self) -> tuple[set[str], set[str]]:
        def closure(roots: set[str]) -> set[str]:
            seen: set[str] = set()
            todo = list(roots)
            while todo:
                n = todo.pop()
                if n in seen or n not in self.schemas:
                    continue
                seen.add(n)
                todo.extend(refs_in(self.schemas[n]) - seen)
            return seen

        ins: set[str] = set()
        outs: set[str] = set()
        for o in self.operations:
            op = o["op"]
            ins |= refs_in(op.get("requestBody", {}))
            for code, r in (op.get("responses") or {}).items():
                if str(code).startswith("2"):
                    outs |= refs_in(r)
        return closure(ins), closure(outs)

    # ------------------------------------------------------------------ types
    def type_of(self, ps: Any, owner: str, prop: str) -> tuple[str, bool]:
        """(Swift type inside the API namespace, whether null is allowed)."""
        if not isinstance(ps, dict) or not ps:
            return "JSONValue", True
        if "$ref" in ps:
            return self.type_names[ref_name(ps["$ref"])], False
        for key in ("anyOf", "oneOf"):
            if key in ps:
                options = ps[key]
                rest = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
                nullable = len(rest) < len(options)
                if len(rest) == 1:
                    t, n = self.type_of(rest[0], owner, prop)
                    return t, nullable or n
                if rest and all(isinstance(o, dict) and "$ref" in o for o in rest):
                    return self.union_type([ref_name(o["$ref"]) for o in rest]), nullable
                return "JSONValue", True
        if "allOf" in ps and len(ps["allOf"]) == 1:
            return self.type_of(ps["allOf"][0], owner, prop)
        t = ps.get("type")
        if isinstance(t, list):
            rest = [x for x in t if x != "null"]
            nullable = len(rest) < len(t)
            if len(rest) == 1:
                inner, n = self.type_of({**ps, "type": rest[0]}, owner, prop)
                return inner, nullable or n
            return "JSONValue", True
        if "enum" in ps and ps["enum"] and all(isinstance(v, str) for v in ps["enum"]):
            return self.enum_type(ps["enum"]), False
        if t == "integer":
            return "Int", False
        if t == "number":
            return "Double", False
        if t == "boolean":
            return "Bool", False
        if t == "string":
            return "String", False
        if t == "null":
            return "JSONValue", True
        if t == "array":
            if "prefixItems" in ps:
                return self.tuple_type(ps["prefixItems"], owner, prop), False
            items = ps.get("items")
            if not isinstance(items, dict) or not items:
                return "[JSONValue]", False
            inner, n = self.type_of(items, owner, prop)
            return f"[{inner}{'?' if n and inner != 'JSONValue' else ''}]", False
        if t == "object" or "properties" in ps or "additionalProperties" in ps:
            if ps.get("properties"):
                return self.inline_object(ps, owner, prop), False
            ap = ps.get("additionalProperties")
            if not isinstance(ap, dict) or not ap:
                return "[String: JSONValue]", False
            inner, n = self.type_of(ap, owner, prop)
            return f"[String: {inner}{'?' if n and inner != 'JSONValue' else ''}]", False
        return "JSONValue", True

    def union_type(self, refs: list[str]) -> str:
        names = [self.type_names[r] for r in refs]
        name = "Or".join(names)
        if name in self.extra_types:
            return name
        if name in self.taken_types:
            name = unique(name + "Union", self.taken_types)
        else:
            self.taken_types.add(name)
        cases = []
        taken: set[str] = set()
        for r, n in zip(refs, names):
            cases.append((unique(escape(lower_camel(n)), taken), n, r))
        props = {r: set((self.schemas[r].get("properties") or {}).keys()) for r in refs}
        reqs = {r: set(self.schemas[r].get("required") or []) for r in refs}
        keyed, tried = [], []
        for case, n, r in cases:
            others = set().union(*(props[o] for o in refs if o != r)) if len(refs) > 1 else set()
            marks = sorted(reqs[r] - others)
            (keyed if marks else tried).append((case, n, r, marks))
        tried.sort(key=lambda c: (-len(reqs[c[2]]), c[1]))
        lines = [f"    /// One of: {', '.join(names)}.",
                 f"    enum {name}: Codable, Hashable, Sendable {{"]
        for case, n, _, _ in [*keyed, *tried]:
            lines.append(f"        case {case}({n})")
        lines += ["", "        init(from decoder: Decoder) throws {",
                  "            let keys = try decoder.container(keyedBy: AnyKey.self)"]
        if not keyed:
            lines.append("            _ = keys")
        for case, n, _, marks in keyed:
            cond = " && ".join(f"keys.contains(AnyKey({swift_string(m)}))" for m in marks)
            lines += [f"            if {cond} {{", f"                self = .{case.strip('`')}(try {n}(from: decoder))",
                      "                return", "            }"]
        for case, n, _, _ in tried:
            lines += [f"            if let value = try? {n}(from: decoder) {{",
                      f"                self = .{case.strip('`')}(value)", "                return", "            }"]
        lines += ["            throw DecodingError.dataCorrupted(DecodingError.Context(",
                  f"                codingPath: decoder.codingPath, debugDescription: {swift_string('None of ' + ', '.join(names))}))",
                  "        }", "", "        func encode(to encoder: Encoder) throws {", "            switch self {"]
        for case, n, _, _ in [*keyed, *tried]:
            lines.append(f"            case .{case.strip('`')}(let value): try value.encode(to: encoder)")
        lines += ["            }", "        }", "    }"]
        self.extra_types[name] = "\n".join(lines)
        return name

    def tuple_type(self, items: list[Any], owner: str, prop: str) -> str:
        name = upper_camel(self.type_names.get(owner, owner)) + upper_camel(prop) + "Tuple"
        if name in self.extra_types:
            return name
        name = unique(name, self.taken_types)
        fields = []
        for i, it in enumerate(items):
            t, n = self.type_of(it, owner, f"{prop}{i}")
            fields.append((f"v{i}", t, n and t != "JSONValue"))
        decl = [f"    /// A fixed-length array: {', '.join(t + ('?' if n else '') for _, t, n in fields)}.",
                f"    struct {name}: Codable, Hashable, Sendable {{"]
        decl += [f"        var {f}: {t}{'?' if n else ''}" for f, t, n in fields]
        decl.append("    }")
        self.extra_types[name] = "\n".join(decl)
        ext = [f"extension API.{name} {{", "    init(from decoder: Decoder) throws {",
               "        var c = try decoder.unkeyedContainer()"]
        for f, t, n in fields:
            ext.append(f"        {f} = try c.{'decodeIfPresent' if n else 'decode'}({self.qualify(t)}.self)")
        ext += ["    }", "", "    func encode(to encoder: Encoder) throws {", "        var c = encoder.unkeyedContainer()"]
        for f, t, n in fields:
            if n:
                ext.append(f"        if let value = {f} {{ try c.encode(value) }} else {{ try c.encodeNil() }}")
            else:
                ext.append(f"        try c.encode({f})")
        ext += ["    }", "}"]
        self.extra_after[name] = "\n".join(ext)
        return name

    def inline_object(self, ps: dict[str, Any], owner: str, prop: str) -> str:
        cache = self.__dict__.setdefault("_inline", {})
        if (owner, prop) in cache:
            return cache[(owner, prop)]
        name = unique(upper_camel(self.type_names.get(owner, owner)) + upper_camel(prop), self.taken_types)
        cache[(owner, prop)] = name
        decl, ext = self.render_struct(name, ps, is_input=owner in self.inputs, output_only=owner not in self.inputs)
        self.extra_types[name] = decl
        if ext:
            self.extra_after[name] = ext
        return name

    # ------------------------------------------------------------------ literals
    def literal(self, value: Any, swift_type: str) -> str | None:
        if value is MISSING or value is None:
            return None
        if isinstance(value, bool):
            if swift_type != "Bool":
                return None
            return "true" if value else "false"
        if isinstance(value, int) and swift_type in ("Int", "Double"):
            return str(value)
        if isinstance(value, float) and swift_type == "Double":
            return repr(value)
        if isinstance(value, str):
            if swift_type == "String":
                return swift_string(value)
            key = next((k for k, n in self.enum_names.items() if n == swift_type), None)
            if key is not None and value in key:
                return "." + self.enum_case_names(self.enum_values[key])[value].strip("`")
            return None
        if value == [] and swift_type.startswith("["):
            return "[]" if ":" not in swift_type else None
        if value == {} and swift_type.startswith("[String:"):
            return "[:]"
        return None

    # ------------------------------------------------------------------ structs
    def render_struct(self, name: str, sc: dict[str, Any], is_input: bool, output_only: bool) -> tuple[str, str]:
        props = sc.get("properties") or {}
        required = set(sc.get("required") or [])
        taken: set[str] = {"nullKeys"}
        fields = []
        for pname, ps in props.items():
            t, nullable = self.type_of(ps, name, pname)
            req = pname in required
            default = ps.get("default", MISSING) if isinstance(ps, dict) else MISSING
            lit = self.literal(default, t) if not nullable else None
            if t == "JSONValue":
                kind = "optional"
            elif not nullable and lit is not None and (req or output_only):
                kind = "defaulted"
            elif nullable or not req:
                kind = "optional"
            else:
                kind = "required"
            fields.append({
                "json": pname, "swift": unique(escape(lower_camel(pname)), taken), "type": t, "kind": kind,
                "nullable": nullable, "required": req, "literal": lit,
                "doc": doc_line(ps.get("description") if isinstance(ps, dict) else None),
            })
        null_keys = is_input and any(f["kind"] == "optional" and f["nullable"] and not f["required"] for f in fields)
        lines = []
        doc = doc_line(sc.get("description"))
        if doc:
            lines.append(f"    /// {doc}")
        lines.append(f"    struct {name}: Codable, Hashable, Sendable {{")
        for f in fields:
            if f["doc"]:
                lines.append(f"        /// {f['doc']}")
            if f["kind"] == "optional":
                lines.append(f"        var {f['swift']}: {f['type']}?")
            elif f["kind"] == "defaulted":
                lines.append(f"        var {f['swift']}: {f['type']} = {f['literal']}")
            else:
                lines.append(f"        var {f['swift']}: {f['type']}")
        if null_keys:
            lines += ["        /// Properties to send as an explicit null while they are nil (nil alone is not sent).",
                      "        var nullKeys: Set<CodingKeys> = []"]
        if fields:
            lines += ["", "        enum CodingKeys: String, CodingKey {"]
            for f in fields:
                bare = f["swift"]
                lines.append(f"            case {bare}" if bare.strip("`") == f["json"] else
                             f"            case {bare} = {swift_string(f['json'])}")
            lines.append("        }")
        lines.append("    }")
        ext: list[str] = []
        if any(f["kind"] == "defaulted" for f in fields):
            ext += [f"    init(from decoder: Decoder) throws {{",
                    "        let c = try decoder.container(keyedBy: CodingKeys.self)"]
            for f in fields:
                key = "." + f["swift"].strip("`")
                if f["kind"] == "required":
                    ext.append(f"        {f['swift']} = try c.decode({self.qualify(f['type'])}.self, forKey: {key})")
                elif f["kind"] == "optional":
                    ext.append(f"        {f['swift']} = try c.decodeIfPresent({self.qualify(f['type'])}.self, forKey: {key})")
                else:
                    ext.append(f"        {f['swift']} = try c.decodeIfPresent({self.qualify(f['type'])}.self, forKey: {key}) "
                               f"?? {f['literal']}")
            ext.append("    }")
        if is_input and any(f["nullable"] for f in fields):
            if ext:
                ext.append("")
            ext += ["    func encode(to encoder: Encoder) throws {",
                    "        var c = encoder.container(keyedBy: CodingKeys.self)"]
            for f in fields:
                key = "." + f["swift"].strip("`")
                s = f["swift"]
                if f["kind"] in ("required", "defaulted"):
                    ext.append(f"        try c.encode({s}, forKey: {key})")
                elif f["nullable"] and f["required"]:
                    ext.append(f"        if let value = {s} {{ try c.encode(value, forKey: {key}) }} "
                               f"else {{ try c.encodeNil(forKey: {key}) }}")
                elif f["nullable"]:
                    ext.append(f"        if let value = {s} {{ try c.encode(value, forKey: {key}) }} "
                               f"else if nullKeys.contains({key}) {{ try c.encodeNil(forKey: {key}) }}")
                else:
                    ext.append(f"        try c.encodeIfPresent({s}, forKey: {key})")
            ext.append("    }")
        ext_text = "\n".join([f"extension API.{name} {{", *ext, "}"]) if ext else ""
        return "\n".join(lines), ext_text

    # ------------------------------------------------------------------ enums
    @staticmethod
    def enum_case_names(values: list[str]) -> dict[str, str]:
        taken: set[str] = {"unknown"}
        out = {}
        for v in values:
            base = lower_camel(v)
            if base == "unknown":
                base = "unknownValue"
            out[v] = escape(unique(base, taken))
        return out

    def render_enum(self, name: str, values: list[str]) -> str:
        cases = self.enum_case_names(values)
        lines = [f"    enum {name}: RawRepresentable, Codable, Hashable, Sendable, CaseIterable {{"]
        lines += [f"        case {c}" for c in cases.values()]
        lines += ["        /// A value this app does not know yet (a newer server).", "        case unknown(String)", "",
                  f"        static let allCases: [{name}] = [{', '.join('.' + c.strip('`') for c in cases.values())}]", "",
                  "        init(rawValue: String) {", "            switch rawValue {"]
        lines += [f"            case {swift_string(v)}: self = .{c.strip('`')}" for v, c in cases.items()]
        lines += ["            default: self = .unknown(rawValue)", "            }", "        }", "",
                  "        var rawValue: String {", "            switch self {"]
        lines += [f"            case .{c.strip('`')}: return {swift_string(v)}" for v, c in cases.items()]
        lines += ["            case .unknown(let value): return value", "            }", "        }", "",
                  "        init(from decoder: Decoder) throws {",
                  "            self.init(rawValue: try decoder.singleValueContainer().decode(String.self))", "        }", "",
                  "        func encode(to encoder: Encoder) throws {", "            var c = encoder.singleValueContainer()",
                  "            try c.encode(rawValue)", "        }", "    }"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ models file
    def models(self) -> str:
        decls: dict[str, str] = {}
        afters: dict[str, str] = {}
        for name in sorted(self.schemas):
            swift = self.type_names[name]
            decl, ext = self.render_struct(swift, self.schemas[name], is_input=name in self.inputs,
                                           output_only=name not in self.inputs)
            decls[swift] = decl
            if ext:
                afters[swift] = ext
        # Operations may introduce more enums and unions (query parameters, bodies, answers).
        for o in self.operations:
            self.endpoint(o)
        for key, name in self.enum_names.items():
            decls[name] = self.render_enum(name, self.enum_values[key])
        decls.update(self.extra_types)
        afters.update(self.extra_after)
        out = [HEADER.format(file=MODELS_FILE).rstrip(), "", "import Foundation", "", "extension API {"]
        out.append("\n\n".join(decls[n] for n in sorted(decls)))
        out.append("}")
        for n in sorted(afters):
            out += ["", afters[n]]
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ endpoints
    def endpoint(self, o: dict[str, Any]) -> dict[str, Any]:
        """The method of one operation: its name, parameters, body and answer (or why it is skipped)."""
        op, path, method = o["op"], o["path"], o["method"]
        # The docstring only: FastAPI's summary is the function name in title case, which says nothing more.
        info: dict[str, Any] = {"path": path, "method": method, "py": o["py"], "doc": doc_line(op.get("description"))}
        if (method, path) in RAW_BODY_ROUTES:
            info["skip"] = RAW_BODY_ROUTES[(method, path)]
            return info
        body = op.get("requestBody")
        if body:
            content = body.get("content") or {}
            if "application/json" not in content:
                info["skip"] = f"{', '.join(sorted(content)) or 'unknown'} body"
                return info
            bt, bn = self.type_of(content["application/json"].get("schema", {}), upper_camel(o["py"]), "body")
            info["body"] = (bt, bn or not body.get("required", False))
        params = []
        for p in op.get("parameters", []) or []:
            t, n = self.type_of(p.get("schema", {}), upper_camel(o["py"]), p["name"])
            params.append({"name": p["name"], "in": p["in"], "type": t,
                           "optional": n or not p.get("required", False)})
        info["params"] = params
        answer = None
        for code in sorted((op.get("responses") or {}).keys()):
            if not str(code).startswith("2"):
                continue
            content = (op["responses"][code].get("content") or {})
            sch = (content.get("application/json") or {}).get("schema")
            if sch is None and not content:
                answer = ("Void", None)
            elif not sch:
                answer = ("Data", None)
            else:
                t, n = self.type_of(sch, upper_camel(o["py"]), "answer")
                answer = (t, n)
            break
        info["answer"] = answer or ("Void", None)
        info["file"] = path.startswith("/files/") or info["answer"][0] == "Data"
        return info

    @staticmethod
    def qualify(t: str) -> str:
        """A namespace-relative type as seen from outside `API` (APIClient's extension)."""
        builtin = {"Int", "Double", "Bool", "String", "Data", "Void"}
        return re.sub(r"[A-Za-z_][A-Za-z0-9_]*",
                      lambda m: m.group(0) if m.group(0) in builtin else f"API.{m.group(0)}", t)

    def endpoints(self) -> str:
        infos = [self.endpoint(o) for o in self.operations]
        names: set[str] = set(self.hand_written) | CORE_NAMES
        for info in infos:
            if "skip" in info:
                continue
            base = lower_camel(info["py"])
            if base in names or base + "Request" in names:
                base = lower_camel(info["method"] + "_" + info["py"])
            info["name"] = unique(base, names)
            if info["file"] and info["method"] == "get":
                names.add(info["name"] + "Request")
        out = [HEADER.format(file=ENDPOINTS_FILE).rstrip()]
        skipped = [i for i in infos if "skip" in i]
        if skipped:
            out.append("//")
            out.append("// Not generated (call them by hand through `perform` or the background uploader):")
            for i in skipped:
                out.append(f"// - {i['method'].upper()} {i['path']}: {i['skip']}")
        out += ["", "import Foundation", "", "extension APIClient {"]
        blocks = []
        for info in infos:
            if "skip" in info:
                continue
            blocks.append(self.render_endpoint(info))
        out.append("\n\n".join(blocks))
        out.append("}")
        return "\n".join(out) + "\n"

    def render_endpoint(self, info: dict[str, Any]) -> str:
        taken: set[str] = {"query", "headers", "path"}
        args: list[tuple[str, str, str | None]] = []  # (declaration, use, default)
        path_expr = info["path"]
        query_lines, header_lines = [], []
        ordered = [p for p in info["params"] if p["in"] == "path"] + \
                  [p for p in info["params"] if p["in"] != "path" and not p["optional"]]
        optional = [p for p in info["params"] if p["in"] != "path" and p["optional"]]
        decl_args = []
        for p in ordered:
            swift = unique(escape(lower_camel(p["name"])), taken)
            t = self.qualify(p["type"])
            decl_args.append(f"{swift}: {t}")
            if p["in"] == "path":
                path_expr = path_expr.replace("{" + p["name"] + "}", f"\\(APIPath.segment({swift}))")
            elif p["in"] == "query":
                query_lines.append(f"        query.add({swift_string(p['name'])}, {swift})")
            elif p["in"] == "header":
                header_lines.append(f"        headers[{swift_string(p['name'])}] = {swift}")
        if "body" in info:
            bt, bn = info["body"]
            bswift = unique("body", taken)
            decl_args.append(f"{bswift}: {self.qualify(bt)}{'? = nil' if bn else ''}")
            body_arg = f", body: {bswift}"
        else:
            body_arg = ""
        for p in optional:
            swift = unique(escape(lower_camel(p["name"])), taken)
            t = self.qualify(p["type"])
            decl_args.append(f"{swift}: {t}? = nil")
            if p["in"] == "query":
                query_lines.append(f"        query.add({swift_string(p['name'])}, {swift})")
            elif p["in"] == "header":
                header_lines.append(f"        if let {swift} {{ headers[{swift_string(p['name'])}] = {swift} }}")
        method = info["method"].upper()
        signature = ", ".join(decl_args)
        prelude = []
        if query_lines:
            prelude.append("        var query = APIQuery()")
            prelude += query_lines
        if header_lines:
            prelude.append("        var headers: [String: String] = [:]")
            prelude += header_lines
        q_arg = ", query: query.items" if query_lines else ""
        h_arg = ", headers: headers" if header_lines else ""
        path_lit = f'"{path_expr}"'
        doc = []
        if info["doc"]:
            doc.append(f"    /// {info['doc']}")
        doc.append(f"    /// {method} {info['path']}")
        answer, nullable = info["answer"]
        blocks = []
        if info["file"] and info["method"] == "get":
            blocks.append("\n".join([
                *doc[:-1], f"    /// A signed request for {method} {info['path']}, for image and video loaders.",
                f"    nonisolated func {info['name']}Request({signature}) throws -> URLRequest {{",
                *prelude,
                f"        return try urlRequest(\"{method}\", {path_lit}{q_arg}{h_arg})",
                "    }",
            ]))
        if info["file"] and info["method"] == "get":
            doc.insert(len(doc) - 1, f"    /// Loads the whole file into memory: stream large files through `{info['name']}Request`.")
        if answer == "Void":
            ret, call = "", f"try await performVoid(\"{method}\", {path_lit}{q_arg}{body_arg}{h_arg})"
        elif answer == "Data":
            ret, call = " -> Data", f"return try await performData(\"{method}\", {path_lit}{q_arg}{body_arg}{h_arg})"
        else:
            t = self.qualify(answer) + ("?" if nullable and answer != "JSONValue" else "")
            ret = f" -> {t}"
            call = f"return try await perform(\"{method}\", {path_lit}{q_arg}{body_arg}{h_arg}, as: {t}.self)"
        blocks.append("\n".join([
            *doc,
            f"    nonisolated func {info['name']}({signature}) async throws{ret} {{",
            *prelude,
            f"        {call}",
            "    }",
        ]))
        return "\n\n".join(blocks)


HEADER = """// {file}
// Generated by scripts/gen_swift_api.py from the server's OpenAPI description. Do not edit by hand:
// change the server, then run `backend/.venv/bin/python scripts/gen_swift_api.py` from the repository root.
// backend/tests/test_swift_api.py fails while this file is out of date.
"""


# ------------------------------------------------------------------------------------------------- entry
def hand_written_methods(path: Path = HAND_WRITTEN_CLIENT) -> set[str]:
    """Method names APIClient.swift already declares, so generated names never clash with them."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(re.findall(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)", text))


def generate(spec: dict[str, Any], hand_written: set[str] | None = None) -> dict[str, str]:
    """The generated files by name, from an OpenAPI description."""
    gen = Generator(spec, hand_written if hand_written is not None else hand_written_methods())
    endpoints = gen.endpoints()  # first: operations add enums and unions the models file must declare
    models = gen.models()
    return {MODELS_FILE: models, ENDPOINTS_FILE: endpoints}


def server_spec() -> dict[str, Any]:
    """The OpenAPI description of the app as the server builds it (models off, a throwaway data folder)."""
    os.environ.setdefault("PCC_NO_MODELS", "1")
    backend = str(REPO / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    with tempfile.TemporaryDirectory(prefix="pcc-swift-api-") as tmp:
        os.environ["DATA_DIR"] = tmp
        from app.main import create_app  # noqa: E402  (imported after the environment is set)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # FastAPI warns about the HEAD twins of file routes
            return json.loads(json.dumps(create_app().openapi()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="folder to write into (default ios/Sources/API)")
    ap.add_argument("--check", action="store_true", help="exit 1 when the files in --out differ")
    args = ap.parse_args(argv)
    files = generate(server_spec())
    stale = []
    for name, text in files.items():
        target = args.out / name
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != text:
            stale.append(name)
            if not args.check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print(f"out of date: {', '.join(stale)}; run scripts/gen_swift_api.py", file=sys.stderr)
            return 1
        print("up to date")
        return 0
    print(f"wrote {', '.join(stale)}" if stale else "nothing changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
