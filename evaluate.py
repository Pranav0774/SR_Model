"""Measure WER on the Google FLEURS test split.

    python evaluate.py --langs en hi ta --engines r2t2 indic whisper --n 100
    python evaluate.py --langs ta --engines indic --n 50

Each engine is only run on the languages it supports. Results go to results/.
Add --apply to write the lowest-WER engine for each language into config.json.
"""
import argparse
import csv
import json
import tarfile
import time
from pathlib import Path

import jiwer
from huggingface_hub import hf_hub_download

from stt import Transcriber
from stt.audio import load_audio
from stt.engines import ENGINES
from stt.normalize import normalize

ROOT = Path(__file__).resolve().parent
FLEURS = {"en": "en_us", "hi": "hi_in", "ta": "ta_in"}
TARGET_WER = 0.10


def fetch_fleurs(lang: str, n: int) -> list[tuple[Path, str]]:
    """Return up to n (wav_path, reference_text) pairs, downloading and extracting on first use."""
    code = FLEURS[lang]
    out = ROOT / "data" / "fleurs" / code
    out.mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download("google/fleurs", f"data/{code}/test.tsv", repo_type="dataset")
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            # columns: id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
            rows.append((line[1], line[3]))
    seen, picked = set(), []
    for name, ref in rows:  # a sentence is read by several speakers; keep one clip per sentence
        if ref not in seen:
            seen.add(ref)
            picked.append((name, ref))
        if len(picked) == n:
            break
    missing = [name for name, _ in picked if not (out / name).exists()]
    if missing:
        tar_path = hf_hub_download("google/fleurs", f"data/{code}/audio/test.tar.gz", repo_type="dataset")
        wanted = set(missing)
        with tarfile.open(tar_path) as tar:
            for member in tar:
                base = Path(member.name).name
                if base in wanted:
                    (out / base).write_bytes(tar.extractfile(member).read())
                    wanted.discard(base)
                    if not wanted:
                        break
    return [(out / name, ref) for name, ref in picked if (out / name).exists()]


def score(refs, hyps, lang):
    """WER/CER with the same normalisation on both sides. Also returns the strict (unnormalised-numbers) WER."""
    kw = {"spell_numbers": lang == "en", "indic": lang in ("hi", "ta")}
    r = [normalize(x, **kw) for x in refs]
    h = [normalize(x, **kw) for x in hyps]
    r0, h0 = [normalize(x) for x in refs], [normalize(x) for x in hyps]
    return jiwer.wer(r, h), jiwer.cer(r, h), jiwer.wer(r0, h0)


def record(summary, lang, label, refs, hyps, seconds):
    wer, cer, strict = score(refs, hyps, lang)
    verdict = "PASS" if wer < TARGET_WER else "FAIL"
    print(f"  {label}: WER {wer:.2%} (basic normalisation only: {strict:.2%})  CER {cer:.2%}  [{verdict} < {TARGET_WER:.0%}]")
    summary[(lang, label)] = {"language": lang, "engine": label, "clips": len(refs), "wer": round(wer, 4),
                              "wer_basic_normalisation": round(strict, 4), "cer": round(cer, 4), "seconds": seconds}


def save_summary(results_dir, summary):
    path = results_dir / "summary.json"
    old = {}
    if path.exists():
        for e in json.loads(path.read_text(encoding="utf-8")):
            old[(e["language"], e["engine"])] = e
    old.update(summary)
    rows = sorted(old.values(), key=lambda e: (e["language"], e["wer"]))
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAll results so far (target WER < 10%)")
    for e in rows:
        print(f"  {e['language']}  {e['engine']:14} WER {e['wer']:.2%}  CER {e['cer']:.2%}  ({e['clips']} clips)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "ta"], choices=list(FLEURS))
    ap.add_argument("--engines", nargs="+", default=["r2t2", "indic", "whisper"], choices=list(ENGINES))
    ap.add_argument("--n", type=int, default=100, help="clips per language")
    ap.add_argument("--decoder", choices=["rnnt", "ctc"], help="IndicConformer decoder (default from config)")
    ap.add_argument("--whisper-model", help="override the Whisper model name or path")
    ap.add_argument("--rescore", action="store_true", help="recompute scores from saved results/*.tsv, no model runs")
    ap.add_argument("--apply", action="store_true", help="route each language to its best engine")
    args = ap.parse_args()

    tr = Transcriber()
    if args.decoder:
        tr.config["engines"]["indic"]["decoder"] = args.decoder
    if args.whisper_model:
        tr.config["engines"]["whisper"]["model"] = args.whisper_model
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary = {}

    if args.rescore:
        for f in sorted(results_dir.glob("*_*.tsv")):
            lang, label = f.stem.split("_", 1)
            rows = list(csv.DictReader(open(f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
            record(summary, lang, label, [r["reference"] for r in rows], [r["hypothesis"] for r in rows], 0)
        save_summary(results_dir, summary)
        return

    for lang in args.langs:
        clips = fetch_fleurs(lang, args.n)
        print(f"\n== {lang}: {len(clips)} clips ==")
        for eng_name in args.engines:
            if lang not in ENGINES[eng_name].languages:
                continue
            label = eng_name
            if eng_name == "indic" and args.decoder:
                label = f"indic-{args.decoder}"
            if eng_name == "whisper" and args.whisper_model:
                label = "whisper-" + Path(args.whisper_model).name
            try:
                tr.engine(eng_name)
            except Exception as e:  # missing weights, gated repo, missing package...
                print(f"  {label}: skipped ({type(e).__name__}: {e})")
                continue
            refs, hyps, t0 = [], [], time.time()
            for i, (path, ref) in enumerate(clips, 1):
                hyps.append(tr.transcribe_array(load_audio(str(path)), lang, eng_name)["text"])
                refs.append(ref)
                if i % 10 == 0:
                    print(f"  {label}: {i}/{len(clips)}", flush=True)
            with open(results_dir / f"{lang}_{label}.tsv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
                w.writerow(["file", "reference", "hypothesis"])
                for (path, _), r, h in zip(clips, refs, hyps):
                    w.writerow([path.name, r.replace("\t", " "), h.replace("\t", " ").replace("\n", " ")])
            record(summary, lang, label, refs, hyps, round(time.time() - t0, 1))

    rows = save_summary(results_dir, summary)

    if args.apply and summary:
        best = {}
        for e in rows:
            if e["engine"] in ENGINES and (e["language"] not in best or e["wer"] < best[e["language"]]["wer"]):
                best[e["language"]] = e
        cfg = tr.config
        for lang, e in best.items():
            cfg["routes"][lang] = e["engine"]
        tr.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("Routes updated:", {k: v["engine"] for k, v in best.items()})


if __name__ == "__main__":
    main()
