#!/usr/bin/env python3
"""Dormant command line.

    python cli.py run            scan, bench, score, brief
    python cli.py brief          the two-line morning brief
    python cli.py top --limit 20 the files it thinks are dead
    python cli.py archive --top 50 --max-gb 5
    python cli.py undo
    python cli.py demo --seed 7 && python cli.py serve --demo
    python cli.py agent "clear off ten gigabytes"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _set_db(args) -> None:
    """Point the whole app at a different data directory before dormant imports."""
    if getattr(args, "demo", False):
        os.environ["DORMANT_DATA"] = str(ROOT / "data" / "demo")
    elif getattr(args, "db_dir", None):
        os.environ["DORMANT_DATA"] = str(Path(args.db_dir).expanduser().resolve())


def _gb(n) -> str:
    return f"{(n or 0) / 1e9:.2f} GB"


def cmd_scan(args):
    from dormant import scanner
    from dormant.store import connect, init_db, set_meta

    conn = connect()
    init_db(conn)
    set_meta(conn, "data_mode", "demo" if args.demo else "real")
    roots = [Path(r).expanduser().resolve() for r in args.roots] if args.roots else None
    from dormant.config import DEFAULT_ROOTS
    roots = roots or [Path(r).expanduser().resolve() for r in DEFAULT_ROOTS]
    roots = [r for r in roots if r.exists()]
    res = scanner.scan(conn, roots, workers=args.workers)
    print(json.dumps(res.as_dict(), indent=2))
    print(f"indexed {res.files_stored:,} files, {res.with_history:,} with recorded open history")


def cmd_demo(args):
    from dormant import demo
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    res = demo.seed(conn, rng_seed=args.seed)
    print(f"synthetic corpus: {res['files']:,} files, {res['open_events']:,} open events, {_gb(res['bytes'])}")
    print("no real paths were touched; this database is safe to screenshot")


def cmd_run(args):
    from dormant import engine
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    out = engine.full_run(roots=args.roots or None, mode="demo" if args.demo else "real", conn=conn)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return
    b = out["brief"]
    print(b["line1"])
    print(b["line2"])
    print()
    s = out.get("scoring", {})
    print(f"scored {s.get('scored', 0):,} stale files in {s.get('seconds', 0)}s  "
          f"(dormancy threshold {s.get('threshold')}, holdout precision "
          f"{round(s.get('precision_at_threshold') or 0, 4)})")
    if args.bench:
        for r in out["bench_detail"]["results"]:
            print(f"  {r['model']:14s} auc {r['auc']}  prec {r['precision']}  rec {r['recall']}  "
                  f"fit {r['fit_s']}s  predict {r['predict_s']}s")
        for r in out["bench_detail"].get("scarcity", []):
            print(f"  n={r['n_train']:>5}  tabpfn {r['tabpfn_auc']}  forest {r['random_forest_auc']}  "
                  f"gb {r['hist_gb_auc']}  best {r['best']}")


def cmd_brief(args):
    from dormant import engine
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    st = engine.state(conn)
    print(st["brief"]["line1"])
    print(st["brief"]["line2"])


def cmd_bench(args):
    from dormant import model
    from dormant.store import connect, init_db, set_meta

    conn = connect()
    init_db(conn)
    res = model.bench(conn)
    set_meta(conn, "scarcity", json.dumps(res.get("scarcity") or []))
    set_meta(conn, "eval_split", json.dumps(res.get("split") or {}))
    print(json.dumps({"split": res["split"], "results": res["results"],
                      "scarcity": res.get("scarcity")}, indent=2, default=str))


def cmd_top(args):
    from dormant import engine
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    rows = engine.top_flagged(conn, limit=args.limit)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        print(f"{1 - r['p_reopen']:>6.1%}  {_gb(r['size_bytes']):>10}  {r['last_used_at'] or 'never'!s:>10}  "
              f"{r['name'][:44]:44s} {r['path'][-42:]}")


def cmd_forecast(args):
    from dormant.forecast import forecast
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    print(json.dumps({k: v for k, v in forecast(conn, volume=args.volume).items() if k != "curve"},
                     indent=2, default=str))


def cmd_archive(args):
    from dormant import action, engine
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    rows = engine.top_flagged(conn, limit=args.top)
    ids = [r["id"] for r in rows]
    total = sum(r["size_bytes"] for r in rows)
    print(f"about to move {len(ids)} files ({_gb(total)}) into {_gb(args.max_gb * 1e9)} budget")
    for r in rows[:10]:
        print(f"  {1 - r['p_reopen']:>6.1%}  {_gb(r['size_bytes']):>10}  {r['name'][:52]}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")
    if not args.yes:
        print("\nnothing moved. re-run with --yes when you actually want it staged.")
        return
    res = action.archive(conn, ids, max_bytes=int(args.max_gb * 1e9), reason="cli")
    print(f"staged {res['count']} files, {_gb(res['bytes'])}, manifest {res['manifest']}")
    print(f"recoverable with: python cli.py undo --manifest {res['manifest']} --yes")


def cmd_undo(args):
    from dormant import action
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    if not args.yes:
        print("nothing restored. re-run with --yes.")
        return
    res = action.undo(conn, args.manifest)
    print(f"restored {res['restored']} files, {_gb(res.get('bytes'))}, manifest {res.get('manifest')}")
    if res.get("failed"):
        print(f"{len(res['failed'])} could not be restored: {res['failed'][0]['why']}")


def cmd_preview(args):
    from dormant import action
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    if args.ids:
        paths = []
        for i in args.ids:
            row = conn.execute("SELECT path FROM files WHERE id = ?", (i,)).fetchone()
            if row is None:
                print(f"id {i} is not in the index")
                continue
            info = action.preview_info(conn, i)
            print(f"{i:>6}  {info['kind']:<6} {_gb(info['size_bytes']):>10}  {info['path']}")
            paths.append(row["path"])
    else:
        paths = args.paths or []
    if not paths:
        print("nothing to preview. pass file ids or paths.")
        return
    if args.reveal:
        print(action.reveal(paths))
        return
    print(action.quicklook(paths))


def cmd_trash(args):
    from dormant import action, engine
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    ids = args.ids or [r["id"] for r in engine.top_flagged(conn, limit=args.top)]
    rows = conn.execute("SELECT path, size_bytes FROM files WHERE id IN (%s)" % ",".join("?" * len(ids)), ids).fetchall()
    print(f"about to send {len(ids)} files ({_gb(sum(r['size_bytes'] for r in rows))}) to the macOS Trash")
    for r in rows[:10]:
        print(f"  {r['path'][-78:]}")
    if not args.yes:
        print("\nnothing moved. re-run with --yes. Put Back in Finder will still reverse this.")
        return
    res = action.trash(conn, ids, reason="cli")
    print(f"trashed {res['count']} files, {_gb(res['bytes'])}, manifest {res['manifest']}")


def cmd_purge(args):
    from dormant import action, engine
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    ids = args.ids or [r["id"] for r in engine.top_flagged(conn, limit=args.top)]
    rows = conn.execute("SELECT path, size_bytes FROM files WHERE id IN (%s)" % ",".join("?" * len(ids)), ids).fetchall()
    total = sum(r["size_bytes"] for r in rows)
    print(f"PERMANENT DELETE: {len(ids)} files, {_gb(total)}. These will not go to the Trash.")
    for r in rows[:10]:
        print(f"  {r['path'][-78:]}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")
    if args.phrase != action.DELETE_PHRASE:
        print(f"\nnothing deleted. re-run with --phrase {action.DELETE_PHRASE} --yes to confirm.")
        return
    if not args.yes:
        print("\nnothing deleted. re-run with --yes as well.")
        return
    res = action.purge(conn, ids, confirm=args.phrase, reason="cli")
    print(f"deleted {res['removed']} files, freed {_gb(res['bytes_freed'])}, manifest {res['manifest']}")
    for s in res["skipped"][:5]:
        print(f"  skipped: {s.get('why')} {s.get('path', '')[-60:]}")
    if res["failed"]:
        print(f"{len(res['failed'])} could not be removed")


def cmd_agent(args):
    from dormant.agent import handle

    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    result = handle(conn, " ".join(args.instruction))
    print(json.dumps(result, indent=2, default=str))


def cmd_serve(args):
    import uvicorn

    if args.demo:
        os.environ["DORMANT_DATA"] = str(ROOT / "data" / "demo")
    from dormant.config import API_HOST, API_PORT
    from dormant.store import connect, init_db

    conn = connect()
    init_db(conn)
    host = args.host or API_HOST
    port = args.port or API_PORT
    print(f"Dormant mission control on http://{host}:{port}  (data: {os.environ.get('DORMANT_DATA', ROOT / 'data')})")
    uvicorn.run("api.app:app", host=host, port=port, log_level="warning")


def action_phrase() -> str:
    from dormant.action import DELETE_PHRASE

    return DELETE_PHRASE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dormant", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", help="override where the index lives")
    p.add_argument("--demo", action="store_true", help="use the synthetic demo database")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="index your files from Spotlight and stat")
    s.add_argument("--roots", nargs="*")
    s.add_argument("--workers", type=int, default=12)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("demo", help="generate a synthetic corpus")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("run", help="scan, bench, score, brief")
    s.add_argument("--roots", nargs="*")
    s.add_argument("--json", action="store_true")
    s.add_argument("--bench", action="store_true", help="also print the model scoreboard")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("brief", help="print the two-line brief")
    s.set_defaults(func=cmd_brief)

    s = sub.add_parser("bench", help="time-split benchmark, all providers")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("top", help="list the files it considers dead")
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_top)

    s = sub.add_parser("forecast", help="disk space projection")
    s.add_argument("--volume", default="/")
    s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("archive", help="stage the top candidates into a recoverable folder")
    s.add_argument("--top", type=int, default=50)
    s.add_argument("--max-gb", type=float, default=10.0)
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_archive)

    s = sub.add_parser("undo", help="reverse an archive manifest")
    s.add_argument("--manifest")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_undo)

    s = sub.add_parser("preview", help="Quick Look indexed files, or show them in Finder")
    s.add_argument("ids", nargs="*", type=int, help="file ids from `dormant top`")
    s.add_argument("--paths", nargs="*", help="raw paths instead of ids")
    s.add_argument("--reveal", action="store_true", help="reveal in Finder rather than preview")
    s.set_defaults(func=cmd_preview)

    s = sub.add_parser("trash", help="move files to the macOS Trash (recoverable)")
    s.add_argument("--ids", nargs="*", type=int)
    s.add_argument("--top", type=int, default=50)
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_trash)

    s = sub.add_parser("purge", help="permanently delete files (irreversible)")
    s.add_argument("--ids", nargs="*", type=int)
    s.add_argument("--top", type=int, default=0)
    s.add_argument("--phrase", default="", help=f"must be {action_phrase()} to proceed")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_purge)

    s = sub.add_parser("agent", help="one sentence in, tool calls out")
    s.add_argument("instruction", nargs="+")
    s.set_defaults(func=cmd_agent)

    s = sub.add_parser("serve", help="run the mission control UI")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _set_db(args)
    sys.path.insert(0, str(ROOT))
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
