# Plan: AI videos of you (your face + voice) for GHL updates

## The short answer
Yes, it's possible, and it's free. Your Mac can do most of the work, but it's too weak (M2, 8 GB of RAM) for the lip-sync step, which animates your face. That step runs on **Kaggle's free GPU**: about 30 hours a week of free computing time, and each video uses about 20 minutes of it. I'd control all of it from your Mac, so you'd just type a command.

## The free tools
All of these allow business use, which matters because this is for your agency.

| Job | Tool | Runs on |
|---|---|---|
| Clone your voice | **Chatterbox** (MIT license) | Your Mac |
| Animate your face to the new audio | **MuseTalk 1.5** (MIT license) | Kaggle's free GPU |
| Captions | **Whisper** (MIT license) | Your Mac |
| Editing: face bubble, captions, title cards | **ffmpeg** + Python | Your Mac |
| Scripts | The public **GHL changelog** feed | Your Mac |

Tools I ruled out:
- **F5-TTS and XTTS** (voice): not allowed for commercial use.
- **LatentSync** (face): its face-detection part is non-commercial only.
- **CodeFormer** (face cleanup): non-commercial only.

## What you record once (about 15 minutes)
1. **Your voice: 60–90 seconds of reading aloud.**
   - Quiet room, phone or USB mic about 15–20 cm from your mouth.
   - Use the upbeat "presenter" tone you want in the videos.
   - I cut it into a few short clips and you pick the one that sounds best.
2. **Your face: 2 minutes of talking to the camera.**
   - iPhone on a tripod, turned sideways, 4K, focus and brightness locked.
   - Frame your head and shoulders, with soft light from the front and a plain background.
   - Look into the lens and nod gently. Close your mouth fully at every pause, and keep your hands away from your face.
   - I keep the steadiest 30 seconds, and every future video is made from that.

## One-time setup (mostly done by me)
1. **Install the tools on your Mac.**
   - Your current ffmpeg can't add captions, so it needs a fuller version.
   - About 8–10 GB in total. You only have **24 GB free**, so I'll keep it lean and clean up after each video.
2. **You:** make a free Kaggle account and verify your phone number. Then I connect it to your Mac.
3. **Build the face-animation setup on Kaggle once**, test it, and save it so each video starts quickly.
4. **Build the pipeline and a `/ghl-video` command** so any future session can run it.

## Each video (about 5–10 minutes of your time, 20–35 minutes of waiting)
1. You type `/ghl-video`. To make a Loom-style video, also give me a silent screen recording of you clicking through GHL.
2. **I write the script.** I read the GHL changelog, pick 3–5 notable updates, and write a 60–90 second script with links to the sources.
3. **Check 1: you approve or edit the script.**
4. **I make the voice-over** in your cloned voice, and automatically redo any sentence where words come out wrong.
5. **Check 2: you listen.** You can ask me to redo any single sentence. Both checks come before any GPU time is used.
6. **I make the video.** Your face is lip-synced on Kaggle, then captions are added and everything is put together. Video styles:
   - **Full-screen talking head**: just you.
   - **Loom-style**: your GHL screen recording with your face in a circle in the corner. I time the narration to what's happening on screen, and I flag anything private on screen (client names, emails) so it can be blurred.
   - **Slides with a face bubble**: used when there's no screen recording.
7. **You get:**
   - A horizontal version for YouTube or LinkedIn.
   - A vertical version for Reels, TikTok and Shorts.
   - A `post.md` file with a title, description and AI-disclosure reminder.

## Things to know
- **Quality:** good, not perfect.
  - Full-screen close-ups can look slightly soft around the mouth.
  - The face-bubble style hides this almost completely.
- **Changelog feed:** it only shows the latest 10 posts, so I'll check it every few days so weekly videos don't miss updates.
- **The voice tool is no longer updated** (since August 2026). I'll lock it to a working version and keep a copy on your Mac.
- **AI labels:** YouTube and Meta require you to label realistic AI versions of yourself. Just tick "AI-generated" when you upload.
- **Please skim Kaggle's terms yourself** ([kaggle.com/terms](https://www.kaggle.com/terms)); I couldn't load them.
- **Backup if Kaggle ever fails:** the same face step can run on Google Colab's free GPU.

## How we test it before the real thing
1. **Health check:** confirm all the tools work and there's enough disk space.
2. **Voice test:** the same 2 sentences (including "GoHighLevel" and "A2P 10DLC") with each voice clip. You pick the one that sounds most like you.
3. **10-second test video** in both styles. I check sound sync, size and loudness automatically, and you watch it to judge the lip-sync and likeness.
4. **One full 60–90 second video**, timing each step so you know what to expect.

## Your to-dos before I start
1. Make a free Kaggle account and verify your phone.
2. Record the voice clip and the face clip. I'll provide a script to read.

---

# Appendix: technical details (for building)

**Checked on this Mac:**
- macOS 14.5, M2, 8 GB RAM, about 48 GB of free disk after the cleanup.
- Homebrew `ffmpeg` 9.0.1 has no `ass`, `subtitles` or `drawtext` filters. *(Changed while building:)* `ffmpeg-full` isn't used, because Homebrew compiles it from source on macOS 14, which takes hours. Captions are drawn as transparent PNGs with Pillow and overlaid with the standard ffmpeg instead.
- Python 3.14 is too new for these tools; use `uv` (installed in `envs/bootstrap`, not Homebrew) with Python 3.11.
- This connection's IPv6 route is unreliable, so the changelog fetcher prefers IPv4.
- GHL changelog RSS is reachable without a login: `https://ideas.gohighlevel.com/api/changelog/feed.rss`. It holds 10 items.

**Pinned versions and decisions:**
- **Chatterbox:**
  - Install from git at commit `ab55923c27`.
  - It reads only the first 10 s of the voice reference (6 s for the text model, 15 s for Turbo), so use a 12–15 s reference.
  - One call produces at most about 40 s of audio, so generate one sentence at a time.
  - Try MPS first, with `PYTORCH_ENABLE_MPS_FALLBACK=1`; fall back to CPU.
  - *(Changed while building:)* the original Chatterbox model (3.2 GB) was already downloaded on this Mac, so it's the default. Turbo is only downloaded if the original is too slow.
- **MuseTalk 1.5:**
  - Clone at commit `0a89dec45a`.
  - Needs Python 3.10, torch 2.0.1 (cu118), mmcv 2.0.1, mmdet 3.1.0 and mmpose 1.1.0. Kaggle's built-in Python 3.12 environment can't run it.
  - Kaggle's P100 GPU no longer works with current torch, so use the T4.
  - Set `HF_ENDPOINT=https://huggingface.co`; MuseTalk's script defaults to `hf-mirror.com`.
  - It already plays the base clip forward then backward to fill the audio length, so there's no need to loop it ourselves.
  - Feed it 25–35 s of base clip at 25 fps.
  - Use normal inference mode with `--use_float16`, not real-time mode (which can stop and wait for keyboard input).

**Kaggle:**
- **Kernel A, `ghl-musetalk-env` (runs once):**
  - Builds the Python 3.10 environment with `uv` under `/tmp/mt`, downloads the weights and runs a 3 s test.
  - Saves `/tmp/mt` and the torch hub cache as `mt_bundle.tar` (about 6–8 GB).
- **Kernel B, `ghl-musetalk-render` (one run per video):**
  - Settings in `kernel-metadata.json`: `enable_gpu`, `enable_internet`, `machine_shape: "NvidiaTeslaT4"`.
  - Inputs: `dataset_sources` = `ghl-avatar` and `ghl-job`; `kernel_sources` = `ghl-musetalk-env`.
  - Finds its input files by searching `/kaggle/input`.
  - Stops if the job ID in `job.json` doesn't match `JOB_ID`, so it never renders an old upload.
  - Writes `lipsync_<id>.mp4`, `status.json` and `log.txt`.
- **Mac side:**
  - `kaggle datasets version`, then poll `datasets status` until the upload is ready.
  - `kernels push -t 3600`, then poll `kernels status` every 60 s, giving up after 60 minutes.
  - `kernels output --file-pattern` to download the result.
  - Check the video length matches the narration within 0.1 s, at 25 fps.
  - A lock file stops two renders running at once.

**Directory layout (`~/Documents/GHL updates/`):**
```
CLAUDE.md  .claude/skills/ghl-video/SKILL.md  config.yaml  pronunciations.yaml
bin/ghl    # doctor|fetch|new|voice|lipsync|captions|compose|render|test|clean
pipeline/  common.py fetch_changelog.py voice.py screen_frames.py timeline.py
           kaggle_lipsync.py captions.py graphics.py compose.py
kaggle/    env/  render/  avatar_ds/  job_ds/
assets/    voice/ref.wav  face/avatar.mp4  brand/
data/changelog/   jobs/<date>-<slug>/   envs/tts  envs/tools  models/
```

**Pipeline details:**
- **Voice quality check:** Whisper transcribes each generated sentence. If more than 15% of words are wrong, the sentence is regenerated, up to 3 times.
- **Loudness:** normalised to −14 LUFS.
- **Loom style:**
  - Scene-change frames are put on labelled 3×3 contact sheets, and the script is written as timed segments in `segments.json`.
  - If narration runs longer than its slot, the screen freezes.
  - If the slot is more than 4 s longer than the narration, the screen speeds up (at most 2×) with a ">>" badge.
  - Lip-sync runs on the full narration track, including the gaps.
- **Captions:** mlx-whisper word timings matched to the script's spelling; 2–4 words at a time with the current word highlighted.
- **Face bubble:** a square crop around the face, scaled to 360 px, with a circular mask and white ring, placed bottom-left.
- **Exports:** 1920×1080 and 1080×1920, x264 CRF 18, `+faststart`.
- **Job tracking:** `job.json` records the last finished stage, so `bin/ghl render` resumes after a failure.

**Rules for CLAUDE.md:**
- Never render before both approvals.
- Use only commercially licensed models.
- Only `avatar.mp4` and `narration.wav` go to Kaggle, and everything there stays private.
- Only one heavy job at a time on this 8 GB Mac.
- Always include the disclosure checklist in `post.md`.
