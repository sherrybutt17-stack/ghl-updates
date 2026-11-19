# GHL update videos (AI face + voice)

Short videos about GoHighLevel product updates, voiced by a clone of the owner's voice and
lip-synced onto their recorded face. Free tools only. Full rationale: PLAN.md.

## Rules
- **Two approval gates, never skip:** (1) the owner approves `script.md`, (2) the owner listens to
  `narration.wav` and approves (`bin/ghl approve-voice`). No Kaggle GPU time before gate 2.
- **Commercial-safe models only:** Chatterbox (MIT), MuseTalk 1.5 (MIT/any-purpose weights),
  mlx-whisper (MIT). Never add F5-TTS, XTTS, LatentSync/InsightFace, CodeFormer, GFPGAN.
- **Privacy:** only `avatar.mp4` and each job's `narration.wav`/`job.json` go to Kaggle, always private.
  Blur client data visible in screen recordings (names, emails, phones) before rendering.
- **8 GB Mac:** one heavy step at a time (TTS, Whisper, ffmpeg renders). Harvey may be running too.
- **Disk is tight (~45 GB free):** run `bin/ghl clean` after a video is posted.
- **Network:** IPv6 on this connection is unreliable — downloads may stall; IPv4 works.
- Every video's `out/post.md` must keep the AI-disclosure checklist.

## Layout
- `bin/ghl` — the only entry point (run `bin/ghl help`).
- `pipeline/` — Python steps. `voice_tts.py` runs in `envs/tts` (Chatterbox pins old torch);
  everything else runs in `envs/tools`.
- `kaggle/musetalk_kaggle.py` — template for both Kaggle notebooks (setup + render).
- `jobs/<date>-<slug>/` — one folder per video; `job.json` tracks finished stages so steps resume.
- `assets/voice/ref.wav`, `assets/face/avatar.mp4` — the one-time recordings.
- `config.yaml` (settings), `pronunciations.yaml` (spoken spellings for TTS only).

## Per-video workflow (the /ghl-video skill)
1. `bin/ghl fetch` → pick 3–5 updates (new features first, agency relevance), open each post's
   entry in `data/changelog/entries/` for the facts.
2. `bin/ghl new "<title>" --style head|slides|loom [--screen <rec.mov>]`.
   For loom: read `jobs/<job>/frames/sheet_*.jpg` to see what happens when.
3. Write `jobs/<job>/script.md` (format in `pipeline/script_format.py`):
   - 60–90 s = 160–230 words. Hook sentence → updates → one-line CTA.
   - Only facts from the changelog text; no invented numbers, dates or availability.
   - Short spoken sentences; spell brand words normally (pronunciations.yaml handles TTS).
   - `[slide] Title | subtitle; bullet` before each update (slides style: slide; head: topic label).
   - Loom: start paragraphs with `[mm:ss]` where that part happens on screen.
   - Sources go below a `---` line. Also store them in job.json as `sources: [{title, link}]`.
   Then **STOP — gate 1: show the script and wait for approval.**
4. `bin/ghl voice` → **STOP — gate 2:** give the owner `jobs/<job>/narration.wav`.
   Fixes: `bin/ghl voice --redo N`, edit the sentence, or add to pronunciations.yaml.
   After approval: `bin/ghl approve-voice`, then `bin/ghl fetch --mark-used` style: `bin/ghl mark-used <job> <guid>...`.
5. `bin/ghl render` (Kaggle lip-sync ~15–25 min, then captions + compose).
6. Check frames from the outputs (ffmpeg -ss … -frames:v 1) and report `out/` paths + post.md.

## Troubleshooting
- Words mispronounced → add to `pronunciations.yaml`, `bin/ghl voice --redo N`.
- Kaggle "stale job dataset" → just rerun `bin/ghl render` (dataset version lag).
- Kaggle failures → `jobs/<job>/kaggle_out/log.txt`; env build log → `data/kaggle_setup/log.txt`.
- Mouth looks soft/odd → tune `face.musetalk` (extra_margin, cheek widths) in config.yaml.
- Kaggle unavailable → the same `kaggle/musetalk_kaggle.py` can be run in a free Colab T4.
