"""Private subprocess entry point. See verifier.py for execution limitations."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    payload = json.load(sys.stdin)
    from sandbox_runner.verifier import _verify_in_process
    if sys.platform == "linux":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = _verify_in_process(**payload)
    print(json.dumps(result.to_dict(), allow_nan=False))


if __name__ == "__main__":
    main()
