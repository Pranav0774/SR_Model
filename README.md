# Speech to text: English, Hindi, Tamil

Converts speech (a microphone recording or an audio file) into text.

| Language | Engine | Model |
|---|---|---|
| English | `r2t2` | Confucius4-R2T2 (Qwen3-ASR based) |
| Hindi | `indic` | AI4Bharat IndicConformer 600M |
| Tamil | `indic` | AI4Bharat IndicConformer 600M |

R2T2 does not support Tamil, which is why Tamil (and by default Hindi) goes to IndicConformer.
Which engine handles which language is set in `config.json` (see "Accuracy" below for the measured WER).
Whisper is included as a fallback engine and for automatic language detection.

## Running it on your machine

You need Windows, Python 3.10-3.12, and [ffmpeg](https://ffmpeg.org) on your PATH.
About 20 GB of disk and 16 GB of RAM is comfortable (translation adds about 3 GB of RAM). No GPU is needed; everything runs on the CPU.

1. **Hugging Face access (once).** IndicConformer is a gated model.
   Open <https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual>, accept the terms, create a
   read token at <https://huggingface.co/settings/tokens>, then after step 2 below run
   `.venv\Scripts\huggingface-cli login`.
2. Double-click **`setup.bat`** (or run it in a terminal). It creates `.venv`, installs the packages and
   downloads the models. If it stops at IndicConformer, do step 1 and run `setup.bat` again.
3. Double-click **`run_app.bat`**. The page opens at <http://127.0.0.1:5000>.

When the app starts it loads the models in the background (one to two minutes, roughly 12 GB of RAM). The
small status line at the top of the page turns green when it is ready. If your computer has little memory, set
`"preload": false` in `config.json` and the models will load on first use instead.

On the page you can scroll the language wheel (or leave it on Detect), upload or drop a file, or tap the
microphone button to record (it transcribes as soon as you stop, unless you untick that box). The round face at
the top right switches between light and dark mode, and your choice is remembered. The transcript is editable, and there are
Copy and Save as .txt buttons. Recent transcripts are kept in your browser only, and can be cleared.

### Command line

```
.venv\Scripts\python transcribe.py sample.wav                    # detects the language
.venv\Scripts\python transcribe.py sample.mp3 --language ta
.venv\Scripts\python transcribe.py sample.wav --language en --engine whisper
```

## Accuracy

Measured with `evaluate.py` on 100 clips per language from the Google FLEURS test set.
The target was WER (word error rate, lower is better) below 10%.

| Language | Engine | WER | CER | Result |
|---|---|---|---|---|
| English | R2T2 | 4.02% | 2.08% | meets target |
| Hindi | IndicConformer (RNNT) | 9.16% | 3.45% | meets target |
| Hindi | IndicConformer (CTC) | 9.72% | 3.52% | meets target |
| Tamil | IndicConformer (RNNT) | 29.70% | 13.99% | **does not meet target** |
| Tamil | IndicConformer (CTC) | 30.74% | 13.11% | does not meet target |

How to read these:
- WER is computed after lowercasing and removing punctuation on both the reference and the output.
  English numbers are spelled out on both sides ("20th" = "twentieth"). For Hindi and Tamil, nukta
  variants and digit forms are unified on both sides. With basic normalisation only, English is 5.92%, Hindi is 11.04% (RNNT)
  and 11.57% (CTC), so Hindi only meets the target under the spelling rules above. Run `python evaluate.py --rescore` to see both figures.
- 100 clips is a small sample. Treat the numbers as estimates, not guarantees, and expect different results on
  your own recordings (noise, accents, microphones).
- Tamil is the open problem. The errors are real: wrong word endings, joining or splitting of words. Options
  still to test: Whisper large-v3, and the Tamil fine-tuned Whisper `vasista22/whisper-tamil-large-v2`
  (`prepare_tamil_model.py` downloads and converts it).
- R2T2 on Hindi and Whisper on any language were not measured.

Re-run it yourself:

```
.venv\Scripts\python evaluate.py --langs en --engines r2t2 --n 100
.venv\Scripts\python evaluate.py --langs hi ta --engines indic --n 100
.venv\Scripts\python evaluate.py --langs ta --engines indic --decoder ctc
.venv\Scripts\python evaluate.py --rescore          # rescore saved results, no models run
```

Per-clip references and outputs are saved in `results/`, and totals in `results/summary.json`.

## Translation to English

Tick "Also give me an English translation" on the page (or use `--translate` on the command line) and Hindi or
Tamil speech is also given as English text. It is done in two steps: the speech is transcribed first, then the text
is translated with Meta's NLLB-200 (distilled, 600M). The licence of that model is CC-BY-NC 4.0, so it is for
non-commercial use only. English speech is left as it is.

Measured on 100 FLEURS sentences per language against the human English versions (BLEU is 0-100, higher is better):

| From | Translating the correct transcript | Full path from speech |
|---|---|---|
| Hindi | BLEU 39.2, chrF++ 63.7 | BLEU 30.9, chrF++ 58.1 |
| Tamil | BLEU 32.8, chrF++ 58.0 | BLEU 23.5, chrF++ 50.5 |

The English is readable and usually keeps the meaning, but it makes mistakes on names and specific words (for
example "Aristotle" came out as "Astut", and "baked goods" as "fast-paced dishes"). Tamil is weaker than Hindi,
partly because the Tamil transcript already has more errors. Translation takes roughly 11 s for 17 s of audio on a
laptop CPU. Re-run the test with `python evaluate_translation.py --langs hi ta`.

## Speed

On a 6-core laptop CPU, English (R2T2) runs at roughly 1.5x real time (about 17 s for an 11 s clip), and
Hindi/Tamil (IndicConformer) at roughly real time or faster. The first request after starting the app is slower
while models load. A GPU with enough memory would be much faster; R2T2 needs more than 4 GB.

## Files

- `app.py`, `static/index.html`: web interface
- `transcribe.py`: command line
- `evaluate.py`: speech recognition accuracy test
- `evaluate_translation.py`: translation quality test
- `stt/`: engines, routing, audio decoding
- `config.json`: language to engine routing and model settings
- `download_models.py`, `prepare_tamil_model.py`: model download helpers
- `models/`: downloaded R2T2 weights (not copied when sharing, `setup.bat` recreates it)
