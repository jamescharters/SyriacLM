#!/usr/bin/env python3
"""Mass-download SEDRA entries from the public Beth Mardutho REST API.

The SEDRA online database exposes one JSON document per record at

    https://sedra.bethmardutho.org/api/<endpoint>/<id>.json

for three id-addressed endpoints, each a JSON array of one object:

* ``word``   -- inflected forms (Syriac spelling, western/eastern pointing,
                stem, part of speech, morphology, glosses, owning lexeme). Ids
                run ``1``..~``65163`` (sparse).
* ``lexeme`` -- dictionary headwords (Syriac, multi-language glosses,
                **etymologies**, owning root, member words, category). Ids run
                ``1``..~``38500`` (sparse).
* ``root``   -- consonantal roots (Syriac, member lexemes). Ids run
                ``1``..~``3500`` (sparse).

(The ``etymology`` data is *embedded* in each lexeme record; SEDRA has no
standalone ``etymology`` endpoint.) This script downloads a whole endpoint
massively in parallel and stores each response as an individual ``<id>.json``
file so the local archive mirrors the API one-to-one.

LICENSE (important). SEDRA is distributed for academic/personal use with
restrictions (no redistribution of altered versions, must cite Kiraz -- see
``corpora.SEDRA_CITATION``). Accordingly the download target defaults to a
**git-ignored** directory under ``corpora/sedra_cache/`` and this repo never
commits SEDRA-derived data.

Robustness / etiquette
----------------------
* **Resumable.** Re-running skips ``<id>.json`` files that already exist and
  ids previously recorded as 404 (in ``_missing.txt``), so an interrupted run
  continues where it stopped. Ids that errored are *not* recorded and are simply
  retried next run.
* **Atomic writes.** Each file is written to ``<id>.json.tmp`` then
  ``os.replace``-d into place, so a crash never leaves a half-written file.
* **Retries with backoff** for timeouts / 429 / 5xx (honours ``Retry-After``).
* **Rate-limited.** The server throttles aggressive clients, so a global token
  bucket (``--rate``, default 8 req/s shared across all workers) spaces request
  starts, and any 429/503 backs the *whole* pool off via ``Retry-After``. Tune
  ``--rate``/``--workers`` up only if the server tolerates it.

Corporate TLS note: on networks that intercept TLS, Python's bundled OpenSSL CA
set rejects the proxy root. If the ``truststore`` package is importable we route
verification through the OS trust store (matches ``curl``); otherwise we fall
back to the default context.

Examples
--------
    # full word archive into corpora/sedra_cache/api/word/ (git-ignored)
    .venv/bin/python -m corpora.sedra_scrape

    # the lexeme and root endpoints
    .venv/bin/python -m corpora.sedra_scrape --endpoint lexeme
    .venv/bin/python -m corpora.sedra_scrape --endpoint root

    # quick smoke test
    .venv/bin/python -m corpora.sedra_scrape --start 1 --end 20 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_API_ROOT = "https://sedra.bethmardutho.org/api"
# Per-endpoint last populated id (with margin); the id spaces are sparse and end
# well before these, but 404s are cheap and recorded so resume skips them.
ENDPOINTS: dict[str, int] = {
    "word": 65999,
    "lexeme": 39999,
    "root": 4999,
}
USER_AGENT = (
    "syriac-sedra-scraper/1.0 (academic use; +https://sedra.bethmardutho.org)"
)
# Transient HTTP statuses worth retrying (everything else is treated as fatal for
# that id; 404 is handled separately as a normal "no such entry" outcome).
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Required acknowledgement for any publication using SEDRA (kept here so the
# scraper is standalone; mirrors neural.sedra.SEDRA_CITATION).
SEDRA_CITATION = (
    'This work makes use of the Syriac Electronic Data Retrieval Archive (SEDRA) '
    'by George A. Kiraz, distributed by the Syriac Computing Institute.'
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


# git-ignored (corpora/.gitignore ignores the whole sedra_cache/ tree). Each
# endpoint gets its own subdirectory: sedra_cache/api/<endpoint>/.
DEFAULT_OUT_BASE = _script_dir() / "sedra_cache" / "api"


class RateLimiter:
    """Thread-safe global throttle shared by every worker.

    Spaces request *starts* by ``1/rate`` seconds across the whole pool (a
    token-bucket with one reserved slot): a thread claims the next slot under a
    lock, then sleeps *outside* the lock so the limiter never serialises useful
    work. ``rate <= 0`` disables throttling. ``penalize`` lets a thread that saw
    a server 429/503 push the next slot forward for *all* workers, so the pool
    backs off together instead of hammering the server one thread at a time.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self.min_interval = (1.0 / rate_per_sec) if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
            wait = slot - now
        if wait > 0.0:
            time.sleep(wait)

    def penalize(self, seconds: float) -> None:
        """Delay the next allowed slot for the whole pool by ``seconds``."""
        if seconds <= 0.0:
            return
        with self._lock:
            self._next = max(self._next, time.monotonic()) + seconds


def enable_system_tls() -> bool:
    """Route TLS verification through the OS trust store if ``truststore`` is
    available (needed behind corporate TLS-intercepting proxies). Returns True
    if injected, False if we fall back to the default SSL context."""
    try:
        import truststore  # type: ignore
    except Exception:
        return False
    truststore.inject_into_ssl()
    return True


def fetch_one(
    wid: int,
    base_url: str,
    out_dir: Path,
    timeout: float,
    retries: int,
    force: bool,
    limiter: RateLimiter,
) -> tuple[str, int]:
    """Download a single word id. Returns ``(status, wid)`` where status is one
    of ``ok`` / ``skip`` / ``missing`` / ``error``. Writes ``<wid>.json`` on a
    200 response (atomically); never holds the body in the caller's memory.
    Every HTTP request passes through ``limiter`` so the pool stays under the
    configured global rate."""
    dest = out_dir / f"{wid}.json"
    if not force and dest.exists() and dest.stat().st_size > 0:
        return ("skip", wid)

    url = f"{base_url}/{wid}.json"
    backoff = 1.0
    for attempt in range(retries + 1):
        wait = backoff
        limiter.acquire()
        try:
            req = Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = out_dir / f"{wid}.json.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            return ("ok", wid)
        except HTTPError as exc:
            if exc.code == 404:
                return ("missing", wid)
            if exc.code not in RETRY_STATUSES:
                return ("error", wid)
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after and retry_after.isdigit():
                wait = float(retry_after)
            # Server is rate-limiting: back the whole pool off, not just us.
            limiter.penalize(wait)
        except (URLError, socket.timeout, TimeoutError, ConnectionError):
            pass
        except Exception:
            pass

        if attempt < retries:
            time.sleep(wait + random.uniform(0.0, 0.5))
            backoff = min(backoff * 2.0, 30.0)
    return ("error", wid)


def _load_missing(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out: set[int] = set()
    for tok in path.read_text(encoding="utf-8").split():
        if tok.strip().isdigit():
            out.add(int(tok))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--endpoint",
        choices=sorted(ENDPOINTS),
        default="word",
        help="which SEDRA endpoint to scrape (default: word)",
    )
    ap.add_argument("--start", type=int, default=1, help="first id (inclusive)")
    ap.add_argument(
        "--end",
        type=int,
        default=None,
        help="last id (inclusive); defaults to the endpoint's known max + margin",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: sedra_cache/api/<endpoint>/; git-ignored, "
        "never commit SEDRA data)",
    )
    ap.add_argument("--workers", type=int, default=8, help="parallel connections")
    ap.add_argument(
        "--rate",
        type=float,
        default=8.0,
        help="max requests/sec across ALL workers (0 = unlimited); the server "
        "rate-limits, so keep this modest",
    )
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request seconds")
    ap.add_argument("--retries", type=int, default=6, help="retries on transient error")
    ap.add_argument(
        "--base-url",
        default=None,
        help="override API base URL (default: <api-root>/<endpoint>)",
    )
    ap.add_argument(
        "--force", action="store_true", help="re-download even if a file exists"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only fetch the first N ids (testing)"
    )
    ap.add_argument(
        "--progress-every", type=int, default=500, help="print progress every N done"
    )
    args = ap.parse_args(argv)

    if args.end is None:
        args.end = ENDPOINTS[args.endpoint]
    if args.base_url is None:
        args.base_url = f"{DEFAULT_API_ROOT}/{args.endpoint}"
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_BASE / args.endpoint

    if args.start < 1 or args.end < args.start:
        print("error: require 1 <= start <= end", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    missing_path = out_dir / "_missing.txt"

    tls = enable_system_tls()
    limiter = RateLimiter(args.rate)
    rate_txt = f"{args.rate:g}/s" if args.rate > 0 else "unlimited"
    print(
        f"SEDRA {args.endpoint} scrape -> {out_dir}\n"
        f"  range {args.start}..{args.end}  workers {args.workers}  rate {rate_txt}  "
        f"system-TLS {'on' if tls else 'off (default ctx)'}\n"
        f"  {SEDRA_CITATION}",
        file=sys.stderr,
    )

    known_missing = set() if args.force else _load_missing(missing_path)

    ids = range(args.start, args.end + 1)
    todo: list[int] = []
    pre_skipped = 0
    for wid in ids:
        if args.limit is not None and len(todo) + pre_skipped >= args.limit:
            break
        if not args.force:
            dest = out_dir / f"{wid}.json"
            if (dest.exists() and dest.stat().st_size > 0) or wid in known_missing:
                pre_skipped += 1
                continue
        todo.append(wid)

    total = len(todo)
    print(
        f"  {total} ids to fetch ({pre_skipped} already present/missing)",
        file=sys.stderr,
    )
    if total == 0:
        print("nothing to do.", file=sys.stderr)
        return 0

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    new_missing: list[int] = []
    errors: list[int] = []
    interrupted = False
    t0 = time.time()

    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            executor.submit(
                fetch_one,
                wid,
                args.base_url,
                out_dir,
                args.timeout,
                args.retries,
                args.force,
                limiter,
            ): wid
            for wid in todo
        }
        done = 0
        for fut in as_completed(futures):
            status, wid = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "missing":
                new_missing.append(wid)
            elif status == "error":
                errors.append(wid)
            done += 1
            if done % args.progress_every == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                eta = (total - done) / rate if rate else 0.0
                print(
                    f"  {done}/{total}  ok={counts['ok']} missing={counts['missing']} "
                    f"error={counts['error']}  {rate:5.1f}/s  eta {eta:5.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- shutting down workers...", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=True)
    finally:
        all_missing = sorted(known_missing | set(new_missing))
        missing_path.write_text("\n".join(map(str, all_missing)) + "\n", encoding="utf-8")
        summary = {
            "endpoint": args.endpoint,
            "base_url": args.base_url,
            "range": [args.start, args.end],
            "out_dir": str(out_dir),
            "requested": total,
            "ok": counts["ok"],
            "missing_this_run": len(new_missing),
            "missing_total": len(all_missing),
            "error": counts["error"],
            "interrupted": interrupted,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        (out_dir / "_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(
        f"done: ok={counts['ok']} missing={len(new_missing)} error={len(errors)} "
        f"in {time.time() - t0:.0f}s -> {out_dir}",
        file=sys.stderr,
    )
    if errors:
        sample = ", ".join(map(str, sorted(errors)[:10]))
        print(
            f"  {len(errors)} ids errored (re-run to retry). e.g. {sample}",
            file=sys.stderr,
        )
    if interrupted:
        print("  (partial run; re-run the same command to resume)", file=sys.stderr)
        return 130
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
