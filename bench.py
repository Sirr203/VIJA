"""Benchmark sample audio recordings through the pipeline, measuring latency and WER/CER.

    python bench.py refs_vi.json ./audio --src Vietnamese --dst Japanese --tag groq
    python bench.py refs_ja.json ./audio --src Japanese --dst Vietnamese --tag groq
    python bench.py refs_vi.json ./audio --src Vietnamese --no-translate --tag asr_only

refs: {"vi_01.m4a": "", "vi_16.m4a": "ground truth text...", ...}
If reference transcript is empty (""), WER/CER is not computed (error column left blank),
recognition and translation still run, and latency is recorded.
Results are saved to bench_<mode>_<src>2<dst>[_<tag>].csv.
"""
import argparse
import csv
import json
import math
import mimetypes
import os
import statistics
import sys
import time
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from app.metrics import metric_for
from app.services.factory import _recognizer, build_interpreter


def lang_code(lang: str) -> str:
    """Convert language name to 2-character code (vi, ja, en,...)."""
    l = (lang or "").strip().lower()
    mapping = {
        "vietnamese": "vi",
        "vi": "vi",
        "japanese": "ja",
        "ja": "ja",
        "english": "en",
        "en": "en",
    }
    return mapping.get(l, l[:2] if len(l) >= 2 else l)


def build_output_filename(mode: str | None, src: str, dst: str, tag: str = "") -> str:
    """Generate result filename: bench_<mode>_<src>2<dst>[_<tag>].csv
    Example: bench_two_step_vi2ja_groq.csv
    """
    mode_str = mode or "two_step"
    direction = f"{lang_code(src)}2{lang_code(dst)}"
    tag_clean = (tag or "").strip().lstrip("_")
    tag_suffix = f"_{tag_clean}" if tag_clean else ""
    return f"bench_{mode_str}_{direction}{tag_suffix}.csv"


def calculate_percentile(values: list[float], p: float) -> float:
    """Calculate p-th percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def compute_bench_summary(rows: list[dict], metric_name: str) -> dict:
    """Aggregate benchmark results:
    - (a) Files with ground truth references: count, mean WER/CER, warning if n < 20.
    - (b) ALL successfully processed files: median and p90 for ASR, MT, total latency, and top 3 slowest files.
    """
    total_files = len(rows)
    success_rows = [r for r in rows if not r.get("loi")]
    failed_rows = [r for r in rows if r.get("loi")]

    # (a) Files with valid ground truth references
    scored_rows = [
        r for r in success_rows
        if r.get("nguyen_van") and str(r.get("nguyen_van")).strip() and isinstance(r.get("score"), (int, float))
    ]
    scores = [r["score"] for r in scored_rows]
    n_scored = len(scores)
    avg_score = statistics.mean(scores) if scores else None

    # (b) All successful files
    asr_times = [float(r["asr_ms"]) for r in success_rows]
    mt_times = [float(r["mt_ms"]) for r in success_rows]
    total_times = [float(r["giay"]) for r in success_rows]

    # Top 3 slowest files by ASR time, then total time
    slowest_rows = sorted(
        success_rows,
        key=lambda r: (float(r.get("asr_ms") or 0), float(r.get("giay") or 0)),
        reverse=True,
    )[:3]
    slowest_files = [
        {
            "file": r.get("file", ""),
            "asr_ms": int(r.get("asr_ms") or 0),
            "size_kb": float(r.get("size_kb") or 0.0),
            "giay": float(r.get("giay") or 0.0),
        }
        for r in slowest_rows
    ]

    return {
        "total_files": total_files,
        "failed_count": len(failed_rows),
        "success_count": len(success_rows),
        "metric_name": metric_name,
        "n_scored": n_scored,
        "avg_score": avg_score,
        "warning_low_n": n_scored < 20,
        "asr_median_ms": statistics.median(asr_times) if asr_times else 0.0,
        "asr_p90_ms": calculate_percentile(asr_times, 90),
        "mt_median_ms": statistics.median(mt_times) if mt_times else 0.0,
        "mt_p90_ms": calculate_percentile(mt_times, 90),
        "total_median_s": statistics.median(total_times) if total_times else 0.0,
        "total_p90_s": calculate_percentile(total_times, 90),
        "slowest_files": slowest_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure WER/CER and end-to-end conversation translation latency")
    ap.add_argument("refs", help="Path to json file containing ground truth references")
    ap.add_argument("audio_dir", help="Directory containing audio recordings")
    ap.add_argument("--mode", default=None, help="two_step (default) | one_step")
    ap.add_argument("--src", default="Vietnamese", help="Source language (default: Vietnamese)")
    ap.add_argument("--dst", default=None, help="Target language (default: auto-inverted with --src)")
    ap.add_argument("--tag", default="", help="Suffix tag for output file: bench_<mode>_<src>2<dst>_<tag>.csv")
    ap.add_argument("--pace", type=float, default=3.5, help="Pause duration (seconds) between files to avoid rate limits (default: 3.5)")
    ap.add_argument("--asr-model", default=None, help="Override speech recognition (ASR) model")
    ap.add_argument("--mt-model", default=None, help="Override translation (MT) model")
    ap.add_argument("--asr-prompt", default=None, help="Vocabulary/prompt hint for ASR")
    ap.add_argument("--no-translate", action="store_true", help="ASR recognition only, skip translation")
    args = ap.parse_args()

    # Automatically set complementary target language if not specified
    if not args.dst:
        args.dst = "Japanese" if args.src.lower() in ("vietnamese", "vi") else "Vietnamese"

    # Configure environment overrides
    if args.asr_model:
        os.environ["GROQ_ASR_MODEL"] = args.asr_model
        os.environ["ASR_MODEL"] = args.asr_model
    if args.mt_model:
        os.environ["GROQ_MT_MODEL"] = args.mt_model
        os.environ["MT_MODEL"] = args.mt_model

    if args.asr_prompt is not None:
        src_lower = args.src.lower()
        if src_lower in ("vietnamese", "vi"):
            os.environ["GROQ_ASR_PROMPT_VI"] = args.asr_prompt
            os.environ.pop("GROQ_ASR_PROMPT_JA", None)
            os.environ.pop("GROQ_ASR_PROMPT", None)
        elif src_lower in ("japanese", "ja"):
            os.environ["GROQ_ASR_PROMPT_JA"] = args.asr_prompt
            os.environ.pop("GROQ_ASR_PROMPT_VI", None)
            os.environ.pop("GROQ_ASR_PROMPT", None)
        else:
            os.environ["GROQ_ASR_PROMPT"] = args.asr_prompt
            os.environ.pop("GROQ_ASR_PROMPT_VI", None)
            os.environ.pop("GROQ_ASR_PROMPT_JA", None)

    metric_name, metric_fn = metric_for(args.src)

    if args.no_translate:
        asr_provider = os.getenv("ASR_PROVIDER", "groq")
        recognizer = _recognizer(asr_provider)
        interp = None
    else:
        recognizer = None
        interp = build_interpreter(args.mode)

    refs = json.loads(Path(args.refs).read_text(encoding="utf-8"))
    items = list(refs.items())
    rows = []

    for idx, (name, raw_ref) in enumerate(items):
        ref_text = (raw_ref or "").strip()
        path = Path(args.audio_dir) / name
        size_kb = round(path.stat().st_size / 1024, 1) if path.exists() else 0.0
        mime = mimetypes.guess_type(path.name)[0] or "audio/mp4"

        if not path.exists():
            print(f"{name}: [MISSING FILE]")
            rows.append({
                "file": name, "size_kb": 0.0, "metric": metric_name, "score": "", "loi": "File not found",
                "giay": 0.0, "asr_ms": 0, "mt_ms": 0,
                "nguyen_van": ref_text, "asr": "", "dich": "", "diem_dich": "",
            })
            continue

        t0 = time.perf_counter()
        try:
            audio_bytes = path.read_bytes()
            if args.no_translate:
                t_asr0 = time.perf_counter()
                original = recognizer.transcribe(audio_bytes, mime, args.src)
                asr_ms = int((time.perf_counter() - t_asr0) * 1000)
                translated = ""
                mt_ms = 0
            else:
                r = interp.interpret(audio_bytes, mime, args.src, args.dst)
                original = r.original
                translated = r.translated
                asr_ms = r.timings_ms.get("asr", 0)
                mt_ms = r.timings_ms.get("mt", 0)
            err_msg = ""
        except Exception as exc:
            err_msg = str(exc)
            original = ""
            translated = ""
            asr_ms = 0
            mt_ms = 0
            print(f"{name}: [ERROR] {exc}")

        wall = time.perf_counter() - t0

        if err_msg:
            rows.append({
                "file": name, "size_kb": size_kb, "metric": metric_name, "score": "", "loi": err_msg,
                "giay": round(wall, 1), "asr_ms": asr_ms, "mt_ms": mt_ms,
                "nguyen_van": ref_text, "asr": "", "dich": "", "diem_dich": "",
            })
        elif not ref_text:
            # Empty reference: skip WER/CER calculation, leave error blank
            rows.append({
                "file": name, "size_kb": size_kb, "metric": metric_name, "score": "", "loi": "",
                "giay": round(wall, 1), "asr_ms": asr_ms, "mt_ms": mt_ms,
                "nguyen_van": "", "asr": original, "dich": translated, "diem_dich": "",
            })
            print(f"{name} ({size_kb} KB): {metric_name}=--  {wall:.1f}s (ASR: {asr_ms}ms, MT: {mt_ms}ms)\n  ASR: {original}\n  DST: {translated}")
        else:
            score = round(metric_fn(ref_text, original), 3)
            rows.append({
                "file": name, "size_kb": size_kb, "metric": metric_name, "score": score, "loi": "",
                "giay": round(wall, 1), "asr_ms": asr_ms, "mt_ms": mt_ms,
                "nguyen_van": ref_text, "asr": original, "dich": translated, "diem_dich": "",
            })
            print(f"{name} ({size_kb} KB): {metric_name}={score:.1%}  {wall:.1f}s (ASR: {asr_ms}ms, MT: {mt_ms}ms)\n  ASR: {original}\n  DST: {translated}")

        if args.pace > 0 and idx < len(items) - 1:
            time.sleep(args.pace)

    if rows:
        out = build_output_filename(args.mode, args.src, args.dst, args.tag)
        fieldnames = [
            "file", "size_kb", "metric", "score", "loi", "giay", "asr_ms", "mt_ms",
            "nguyen_van", "asr", "dich", "diem_dich",
        ]
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        summary = compute_bench_summary(rows, metric_name)

        print("\n" + "=" * 55)
        print(f"BENCHMARK RESULTS ({args.src} -> {args.dst})")
        print("=" * 55)
        print(f"Total files processed     : {summary['total_files']}")
        print(f"Successful files          : {summary['success_count']}")
        print(f"Failed/missing files       : {summary['failed_count']}")

        print(f"\n(a) RECOGNITION ACCURACY ({metric_name}):")
        if summary["n_scored"] > 0:
            print(f"  • Files with ground truth (n) : {summary['n_scored']}")
            print(f"  • Average {metric_name}       : {summary['avg_score']:.1%}")
            if summary["warning_low_n"]:
                print(f"  ⚠️  WARNING: Sample size n = {summary['n_scored']} (< 20). Insufficient statistical confidence!")
        else:
            print("  • No clips have ground truth references to evaluate.")

        print("\n(b) PROCESSING LATENCY (ALL SUCCESSFUL FILES):")
        if summary["success_count"] > 0:
            print(f"  • Speech Recognition (ASR) : median {summary['asr_median_ms']:.0f} ms | p90 {summary['asr_p90_ms']:.0f} ms")
            if not args.no_translate:
                print(f"  • Translation (MT)         : median {summary['mt_median_ms']:.0f} ms | p90 {summary['mt_p90_ms']:.0f} ms")
            print(f"  • Total Latency            : median {summary['total_median_s']:.2f} s  | p90 {summary['total_p90_s']:.2f} s")
            if summary.get("slowest_files"):
                print("  • Top 3 slowest files:")
                for r in summary["slowest_files"]:
                    print(f"    - {r['file']}: asr_ms={r['asr_ms']} ms, size_kb={r['size_kb']:.1f} KB (total {r['giay']:.1f}s)")
        else:
            print("  • No successful files to compute latency.")

        print(f"\nResult file: {out}")


if __name__ == "__main__":
    main()
