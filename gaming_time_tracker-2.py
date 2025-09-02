"""
Labor Day Hackathon
2025
Isabella Chen
"""

import argparse
import contextlib
import datetime as dt
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

from tzlocal import get_localzone
import tldextract  # optional domain cleanup if present in titles

# Lazy imports for report so track-only users don't need matplotlib right away
with contextlib.suppress(Exception):
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore

try:
    import pywinctl  # type: ignore
except Exception:
    print("pywinctl is required. Install with: pip install pywinctl", file=sys.stderr)
    raise

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaming_time.db")

KEYWORDS: List[str] = ["roblox","steam","epic games","league of legends","minecraft"]
KEYWORDS_LC: List[str] = [k.lower() for k in KEYWORDS if k.strip()]
# Minutes per day (Mon..Sun)
GOALS_MIN = [120,120,120,120,180,240,180]

@dataclass(frozen=True)
class Sample:
    ts: dt.datetime
    matched: bool
    dur: float  # seconds duration represented by this sample


def now_local() -> dt.datetime:
    tz = get_localzone()
    return dt.datetime.now(tz)


def ensure_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            ts TEXT NOT NULL,
            matched INTEGER NOT NULL,
            dur REAL NOT NULL DEFAULT 1.0
        )
        """
    )
    # add missing 'dur' column if migrating from older version
    cur.execute("PRAGMA table_info(samples)")
    cols = [r[1] for r in cur.fetchall()]
    if "dur" not in cols:
        cur.execute("ALTER TABLE samples ADD COLUMN dur REAL NOT NULL DEFAULT 1.0")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts)")
    conn.commit()


def get_active_title() -> str:
    try:
        w = pywinctl.getActiveWindow()
        if not w:
            return ""
        title = (w.title or "").lower()
        return title
    except Exception:
        return ""


def title_matches(title: str) -> bool:
    t = title.lower()
    if any(kw in t for kw in KEYWORDS_LC):
        return True
    parts = tldextract.extract(t)
    core = parts.domain
    if core and any(core in kw or kw in core for kw in KEYWORDS_LC):
        return True
    return False


def track(poll_sec: float = 1.0, flush_every: int = 20) -> None:
    print("Tracking started. Press Ctrl+C to stop.")
    print(f"Keywords: {', '.join(KEYWORDS_LC)}")
    print("Sampling active window title every", poll_sec, "seconds")

    buf: List[Sample] = []
    with sqlite3.connect(DB_PATH) as conn:
        ensure_db(conn)
        try:
            last_ts = now_local()
            prev_matched = title_matches(get_active_title())
            last_heartbeat = time.time()
            while True:
                time.sleep(poll_sec)
                now_ts = now_local()
                title = get_active_title()
                matched = title_matches(title)
                dur = (now_ts - last_ts).total_seconds()
                dur = max(0.1, min(dur, poll_sec * 2.5))
                buf.append(Sample(last_ts, bool(prev_matched), float(dur)))
                last_ts = now_ts
                prev_matched = matched

                if len(buf) >= flush_every:
                    cur = conn.cursor()
                    cur.executemany(
                        "INSERT INTO samples (ts, matched, dur) VALUES (?, ?, ?)",
                        [(s.ts.isoformat(), int(s.matched), s.dur) for s in buf],
                    )
                    conn.commit()
                    buf.clear()

                if time.time() - last_heartbeat > 60:
                    last_heartbeat = time.time()
                    print("...tracking...", now_local().strftime("%Y-%m-%d %H:%M"))
        except KeyboardInterrupt:
            if buf:
                cur = conn.cursor()
                cur.executemany(
                    "INSERT INTO samples (ts, matched, dur) VALUES (?, ?, ?)",
                    [(s.ts.isoformat(), int(s.matched), s.dur) for s in buf],
                )
                conn.commit()
            print("
Tracking stopped.")


def weekly_goals_series() -> Dict[str, int]:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {d: int(GOALS_MIN[i]) for i, d in enumerate(days)}


def aggregate_daily_minutes(conn: sqlite3.Connection, since_days: int = 7) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, matched, COALESCE(dur, 1.0) FROM samples WHERE ts >= ?",
        ((now_local() - dt.timedelta(days=since_days)).isoformat(),),
    )
    rows = cur.fetchall()
    buckets: Dict[str, float] = {d: 0.0 for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]}
    for ts_str, matched, dur in rows:
        if int(matched) != 1:
            continue
        try:
            ts = dt.datetime.fromisoformat(ts_str)
        except Exception:
            continue
        d = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][ts.weekday()]
        buckets[d] += float(dur)
    return {k: int(round(v / 60.0)) for k, v in buckets.items()}


def report() -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        print("matplotlib and pandas are required for reporting. Install with: pip install matplotlib pandas", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(DB_PATH) as conn:
        ensure_db(conn)
        actual_min = aggregate_daily_minutes(conn, since_days=7)

    goals = weekly_goals_series()
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    df = pd.DataFrame({
        "day": days,
        "goal": [goals[d] for d in days],
        "actual": [actual_min.get(d, 0) for d in days],
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(days))
    ax.bar([i - 0.2 for i in x], df["goal"], width=0.4, label="Goal", color="#6d4aff")
    ax.bar([i + 0.2 for i in x], df["actual"], width=0.4, label="Actual", color="#62d0ff")
    ax.set_xticks(list(x))
    ax.set_xticklabels(days)
    ax.set_ylabel("Minutes")
    ax.set_title("Gaming time vs goal (last 7 days)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaming_time_report.png")
    plt.savefig(out_path, dpi=160)
    print("Saved:", out_path)
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaming time tracker")
    parser.add_argument("mode", choices=["track", "report"], help="track = record usage, report = show chart")
    parser.add_argument("--poll", type=float, default=1.0, help="sampling interval in seconds (default: 1.0)")
    args = parser.parse_args()
    if args.mode == "track":
        track(poll_sec=args.poll)
    else:
        report()


if __name__ == "__main__":
    main()
