"""Plain, flushed progress for long CPU preparation loops and saved logs."""
import time


def progress(items, label, total=None, interval=15.0):
    """Yield unchanged items and count completed iterations, without RNG use."""
    if total is None:
        total = len(items) if hasattr(items, "__len__") else None
    start = last = time.monotonic()
    count = 0

    def report(now, status):
        elapsed = max(0.0, now - start)
        extent = str(count) if total is None else f"{count}/{total}"
        rate = count / elapsed if elapsed else 0.0
        print(f"[prepare] {label}: {extent}; elapsed={elapsed:.1f}s; "
              f"rate={rate:.2f}/s; {status}", flush=True)

    report(start, "started")
    for item in items:
        yield item
        count += 1
        now = time.monotonic()
        if count == 1 or now - last >= interval:
            report(now, "running")
            last = now
    report(time.monotonic(), "complete")
