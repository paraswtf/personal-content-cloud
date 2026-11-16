"""Print the id of the simulator the iOS tests run on: an available iPhone.

    xcrun simctl list devices available -j | python3 pick_simulator.py [SDK version]

With an SDK version (for example 18.5, from `xcrun --sdk iphonesimulator --show-sdk-version`) the runtime
of that version is taken when the machine has it: the app is built with that SDK, and a much newer runtime
booted far more slowly on the CI machine. Otherwise the newest runtime. Among its iPhones a plain
"iPhone NN Pro" is preferred, then a plain "iPhone NN", then any iPhone, so a run picks the same kind of
phone from one machine image to the next. Prints "<udid> <name> <runtime>"; exits 1 when there is none.
"""
import json
import re
import sys


def version(runtime: str) -> tuple[int, ...]:
    m = re.search(r"iOS-(\d+)-(\d+)(?:-(\d+))?", runtime)
    return tuple(int(x) for x in m.groups() if x is not None) if m else ()


def rank(name: str) -> int:
    if re.fullmatch(r"iPhone \d+ Pro", name):
        return 0
    if re.fullmatch(r"iPhone \d+", name):
        return 1
    return 2


def main() -> int:
    devices = json.load(sys.stdin).get("devices", {})
    runtimes = sorted((rt for rt in devices if version(rt)), key=version, reverse=True)
    if len(sys.argv) > 1:
        want = tuple(int(x) for x in sys.argv[1].split(".") if x.isdigit())
        runtimes.sort(key=lambda rt: version(rt)[: len(want)] != want)  # stable: the SDK's runtime first
    for rt in runtimes:
        phones = [d for d in devices[rt] if d.get("isAvailable", True) and d.get("name", "").startswith("iPhone")]
        if phones:
            best = sorted(phones, key=lambda d: (rank(d["name"]), d["name"]))[0]
            print(best["udid"], best["name"].replace(" ", "_"), rt.rsplit(".", 1)[-1])
            return 0
    print("no available iPhone simulator", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
