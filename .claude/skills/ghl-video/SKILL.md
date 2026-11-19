---
name: ghl-video
description: Make a GHL (GoHighLevel) update video with the owner's AI face and voice — fetch the changelog, write the script, generate voice, lip-sync on Kaggle, compose 16:9 + 9:16 videos. Use for "make this week's GHL video", "new GHL update video", or when given a GHL screen recording to narrate.
---

# Make a GHL update video

Work in the project root. Follow CLAUDE.md rules — especially the two STOP gates.

## Inputs
- Optional style: `head` (talking head, default), `slides` (auto slides + face bubble),
  `loom` (owner's screen recording + face bubble). A screen recording path implies `loom`.
- Optional topic: specific updates the owner wants covered.

## Steps
1. `bin/ghl doctor` — if anything one-time is missing, tell the owner what to do and stop.
2. `bin/ghl fetch` — list unused posts. Choose 3–5 (or the owner's topic). Read the entries in
   `data/changelog/entries/*.json` for facts.
3. `bin/ghl new "<short title>" --style <style> [--screen <path>]`.
   Loom: view every `jobs/<job>/frames/sheet_*.jpg` (timestamps are printed on each frame) and
   note what's on screen when; flag any client data visible.
4. Write `jobs/<job>/script.md` (160–230 words, hook → updates → CTA, facts only, `[slide]`
   lines per update, `[mm:ss]` anchors for loom, sources under `---`). Add
   `sources: [{"title","link","guid"}]` to job.json.
5. **STOP (gate 1).** Show the script, word count and estimated length; wait for approval/edits.
6. `bin/ghl voice`. Report any sentences flagged by the check.
7. **STOP (gate 2).** Ask the owner to listen to `jobs/<job>/narration.wav`. Apply fixes with
   `bin/ghl voice --redo N` / pronunciations.yaml until approved, then `bin/ghl approve-voice`
   and `bin/ghl mark-used <job> <guid> ...`.
8. `bin/ghl render` (runs in the background; ~20–30 min). If Kaggle fails, read
   `kaggle_out/log.txt`, fix, rerun — it resumes.
9. Extract 3 frames from each output video and look at them (captions readable, face bubble
   clean, nothing cut off). Report the `out/` files, thumbnail and post.md (with the
   AI-disclosure checklist). Offer `bin/ghl clean` once posted.
