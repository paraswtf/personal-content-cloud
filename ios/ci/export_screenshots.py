"""Copy the screenshots the UI tests kept (ScreenTestCase.snap) out of a test result bundle.

    python3 export_screenshots.py Tests.xcresult out/

Uses `xcrun xcresulttool export attachments` (Xcode 16 and later), which writes every attachment plus a
manifest.json naming the test each came from. The screenshots are renamed to the name the test gave them
("sync.png"); a name used twice gets the test's name added. Also writes out/summary.json (the result
summary) and out/tests.txt (one line per test with its result), and prints that list.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.run(["xcrun", "xcresulttool", *args], check=True, capture_output=True, text=True).stdout


def main() -> int:
    bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        print(f"no result bundle at {bundle}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run("export", "attachments", "--path", str(bundle), "--output-path", tmp)
        except subprocess.CalledProcessError as e:
            print(f"could not export attachments: {e.stderr.strip()}")
            return 0
        manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        used: dict[str, int] = {}
        for test in manifest:
            test_name = re.sub(r"[^A-Za-z0-9]+", "-", test.get("testIdentifier", "test")).strip("-")
            for att in test.get("attachments", []):
                src = Path(tmp) / att["exportedFileName"]
                human = att.get("suggestedHumanReadableName") or att["exportedFileName"]
                # "sync_0_8F2C1E5A-....png": the name the test gave, an index, a UUID
                name = re.sub(r"_\d+_[0-9A-Fa-f-]{36}(?=\.\w+$)", "", human)
                stem, ext = Path(name).stem, Path(name).suffix or src.suffix
                # Only what upload-artifact accepts in a file name (XCTest names its debug texts with quotes and colons).
                stem = re.sub(r'[\\/"*:<>?|\r\n]+', "-", stem).strip("-. ") or "attachment"
                ext = re.sub(r"[^A-Za-z0-9.]+", "", ext)
                if stem in used:
                    stem = f"{stem}-{test_name}"
                used[stem] = used.get(stem, 0) + 1
                shutil.copy(src, out / f"{stem}{ext}")
    try:
        (out / "summary.json").write_text(run("get", "test-results", "summary", "--path", str(bundle)))
        tests = json.loads(run("get", "test-results", "tests", "--path", str(bundle)))
        lines: list[str] = []

        def walk(node: dict) -> None:
            if node.get("nodeType") == "Test Case":
                lines.append(f"{node.get('result', '?'):>8}  {node.get('nodeIdentifier', node.get('name'))}")
            for child in node.get("children", []):
                walk(child)

        for node in tests.get("testNodes", []):
            walk(node)
        (out / "tests.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"could not read the test results: {e}")
    print(f"{len(list(out.glob('*.png')))} screenshots in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
