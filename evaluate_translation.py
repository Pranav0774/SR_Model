"""Measure Hindi/Tamil to English translation quality on FLEURS.

    python evaluate_translation.py --langs hi ta

FLEURS has the same sentences in every language, so each Hindi/Tamil clip has a human English
reference. Two numbers are reported per language:

  text -> English   translating the correct (human) transcript: the translation model alone
  speech -> English translating what the speech recogniser produced: what a user actually gets

Speech recognition outputs come from results/<lang>_indic.tsv (run evaluate.py first).
BLEU and chrF++ come from sacrebleu; higher is better (BLEU 0-100).
"""
import argparse
import csv
import json
from pathlib import Path

import sacrebleu
from huggingface_hub import hf_hub_download

from stt.translate import Translator

ROOT = Path(__file__).resolve().parent
FLEURS = {"hi": "hi_in", "ta": "ta_in"}


def fleurs_rows(code):
    path = hf_hub_download("google/fleurs", f"data/{code}/test.tsv", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "ta"], choices=list(FLEURS))
    ap.add_argument("--asr", default="indic", help="which saved recogniser output to use (results/<lang>_<asr>.tsv)")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8")).get("translation", {})
    tr = Translator(cfg)
    english = {r[0]: r[2] for r in fleurs_rows("en_us")}          # sentence id -> English reference
    summary = []

    for lang in args.langs:
        rows = fleurs_rows(FLEURS[lang])
        by_file = {r[1]: (r[0], r[2]) for r in rows}                # wav name -> (sentence id, raw human text)
        saved = ROOT / "results" / f"{lang}_{args.asr}.tsv"
        with open(saved, encoding="utf-8") as f:
            hyp_rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

        refs, gold, hyps = [], [], []
        for r in hyp_rows:
            sid, raw = by_file[r["file"]]
            refs.append(english[sid])
            gold.append(raw)
            hyps.append(r["hypothesis"])

        print(f"\n== {lang}: {len(refs)} sentences ==", flush=True)
        from_text = tr.translate(gold, lang)
        from_speech = tr.translate(hyps, lang)
        for label, out in (("text -> English", from_text), ("speech -> English", from_speech)):
            bleu = sacrebleu.corpus_bleu(out, [refs]).score
            chrf = sacrebleu.corpus_chrf(out, [refs], word_order=2).score
            print(f"  {label:18} BLEU {bleu:5.1f}   chrF++ {chrf:5.1f}")
            summary.append({"language": lang, "path": label, "bleu": round(bleu, 1), "chrf++": round(chrf, 1),
                            "sentences": len(refs)})

        with open(ROOT / "results" / f"translation_{lang}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerow(["file", "english_reference", "from_human_text", "from_speech"])
            for r, ref, a, b in zip(hyp_rows, refs, from_text, from_speech):
                w.writerow([r["file"], ref, a.replace("\t", " "), b.replace("\t", " ")])

    (ROOT / "results" / "translation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
