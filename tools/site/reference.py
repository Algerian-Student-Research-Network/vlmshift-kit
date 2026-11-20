"""The page of every key of experiment.yaml and analysis.yaml, read from their JSON Schema in docs/.

The schemas are exported from the code (python -m vlmshift.config.schema docs), so this page cannot
say a key exists when the code refuses it, or forget one the code accepts.
"""
import html
import json
from pathlib import Path

FILES = {
    "experiment": ("experiment.yaml", "What to ask, to which models, on which table. Read by import, audit, prepare and run.",
                   "documentation.html#4-describing-an-experiment-experimentyaml"),
    "analysis": ("analysis.yaml", "What to compute on a run. Read by analyze; it never calls a model.",
                 "documentation.html#8-describing-an-analysis-analysisyaml"),
}
RAW = "https://raw.githubusercontent.com/Algerian-Student-Research-Network/vlmshift-kit/main/docs/{name}.schema.json"


def kind(spec: dict, defs: dict) -> tuple[str, dict | None]:
    """A readable type, and the object whose keys go below it when there is one."""
    if "$ref" in spec:
        target = defs[spec["$ref"].split("/")[-1]]
        return ("object", target) if target.get("properties") else kind(target, defs)
    if "anyOf" in spec:
        parts = [kind(option, defs) for option in spec["anyOf"] if option.get("type") != "null"]
        inner = next((sub for _, sub in parts if sub), None)
        return " or ".join(dict.fromkeys(name for name, _ in parts)), inner
    if "enum" in spec:
        return "choice", None
    if "const" in spec:
        return "fixed", None
    if spec.get("type") == "array":
        name, inner = kind(spec.get("items", {}), defs)
        return f"list of {name}", inner
    if spec.get("type") == "object" and "additionalProperties" in spec and isinstance(spec["additionalProperties"], dict):
        name, inner = kind(spec["additionalProperties"], defs)
        return f"map of name → {name}", inner
    return {"string": "text", "integer": "integer", "number": "number", "boolean": "true / false",
            "object": "object"}.get(spec.get("type"), "any"), None


def values(spec: dict, defs: dict) -> str:
    """The allowed values, when the schema lists them."""
    if "$ref" in spec:
        return values(defs[spec["$ref"].split("/")[-1]], defs)
    for option in spec.get("anyOf", []):
        if found := values(option, defs):
            return found
    if "enum" in spec:
        return ", ".join(f"<code>{html.escape(str(v))}</code>" for v in spec["enum"])
    if "const" in spec:
        return f"<code>{spec['const']}</code>"
    if spec.get("type") == "array":
        return values(spec.get("items", {}), defs)
    return ""


def rows(obj: dict, defs: dict, prefix: str, seen: tuple = ()) -> list[str]:
    """One table row per key, then the keys of the objects it holds, with a dotted path."""
    found = []
    required = set(obj.get("required", []))
    for key, spec in obj.get("properties", {}).items():
        name, inner = kind(spec, defs)
        path = prefix + key + ("" if not inner else "[]" if name.startswith("list") else ".<name>" if name.startswith("map") else "")
        default = spec.get("default", "")
        default = "" if default in ("", None) else f"<code>{html.escape(json.dumps(default, ensure_ascii=False))}</code>"
        note = html.escape(spec.get("description", "") or (inner or {}).get("description", ""))
        found.append(
            f"<tr><td><code>{html.escape(path)}</code></td><td>{html.escape(name)}</td>"
            f"<td>{'yes' if key in required else ''}</td><td>{default}</td><td>{values(spec, defs)}</td><td>{note}</td></tr>"
        )
        title = (inner or {}).get("title")
        if inner and title not in seen:
            found += rows(inner, defs, path + ".", seen + (title,))
    return found


def section(name: str, label: str, about: str, guide: str) -> str:
    schema = json.loads((Path(__file__).parents[2] / "docs" / f"{name}.schema.json").read_text(encoding="utf-8"))
    head = "<tr><th>Key</th><th>Type</th><th>Required</th><th>Default</th><th>Allowed values</th><th>Note</th></tr>"
    body = "\n".join(rows(schema, schema.get("$defs", {}), ""))
    return (
        f'<h2 id="{name}">{label}</h2>\n<p>{about} Explained with examples in the <a href="{guide}">documentation</a>. '
        f'JSON Schema: <a href="{RAW.format(name=name)}">{name}.schema.json</a>.</p>\n'
        f'<div class="table"><table class="keys"><thead>{head}</thead><tbody>\n{body}\n</tbody></table></div>\n'
    )


def page() -> str:
    intro = (
        "<h1>Configuration reference</h1>\n"
        "<p>Every key the two configuration files accept, with its type, its default and its allowed values. "
        "This page is written from the schemas the code exports, so it always matches it: a key that is not here is refused.</p>\n"
        "<p>Your editor can check the files as you type and suggest the keys. With the YAML extension of VS Code, "
        "write this first line in <code>experiment.yaml</code> (and the same with <code>analysis</code> in "
        "<code>analysis.yaml</code>):</p>\n"
        f"<pre><code># yaml-language-server: $schema={RAW.format(name='experiment')}</code></pre>\n"
        "<p>A path such as <code>models[].revision</code> is the key <code>revision</code> of each entry of the list "
        "<code>models</code>; <code>axes.&lt;name&gt;.type</code> is the key <code>type</code> of each axis, under its name.</p>\n"
    )
    return intro + "".join(section(name, *info) for name, info in FILES.items())
