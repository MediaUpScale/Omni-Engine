# Command Reference — Unified Multi-Page Factory

Ready-to-copy CLI commands scanned from `main.py`, scheduler modules, and generation scripts.

> **Note on flags.** `main.py` does **not** accept `--model` or `--qty`. Quantity is `--quantity` / `--count` / `-n`. Image/audio/video providers are selected with `--model-api-flow` (and optional per-media overrides). There is no `youtube_scheduler` package; YouTube uses `agents/posting/publish_existing.py` and `main.py --publish-youtube`.
>
> **Layout.** Run every command from the factory root. `main.py` stays at root. Writer: `agents/writer/`. RAG: `agents/rag/`. Posting CLIs: `agents/posting/` (`pinterest_main.py`, `pinterest_oauth.py`, `publish_existing.py`). Orchestrator: `agents/orchestrator/`. Media: `agents/media/`. Quality: `quality/VisualQA_Agent/`. Core pipelines: `core/`. Ancient Knowledge env/token: `channels_config/ancient_knowledge/`. Principles of Wealth CLI: `channels_config/principles_of_wealth_finance_economics/wealth_main.py`.
>
> **LOFI Stage 1 writer.** Default is one Few-Shot call to `models/gemini-3.8-flash` that returns spoken beats **and** a `visual_concept` per beat. Niche presets (`relationship`, `parenting`) supply golden examples and the Aesthetic Master prefix. Stage 3 wraps `visual_concept` as `{aesthetic_prefix}, {concept}, shot on 35mm lens, moody depth of field` and sends that verbatim to Flux/Together. TTS-first durations drive Stage 4. Set `LOFI_ENABLE_STRICT_JUDGE=1` to restore the archived judge/repair path.
>
> ElevenLabs TTS speed / `voice_settings`: see `docs/elevenlabs_tts.md`. Speed always lives inside `voice_settings`, never as a standalone client kwarg. LOFI production knobs are `TTS_VOICE_ID`, `TTS_MODEL`, `TTS_SPEED` (valid speed **0.7–1.2**) in `core/economic_reel_lofi/config.py`.

### ECONOMIC_REEL_LOFI story-quality gate

Every locked or generated script must clear this **before** stills/TTS:

1. **Causal spine** — each line connects to the next via consequence or contrast (the hope 9-liner is the bar).
2. **Stakes** — something risked, lost, or almost lost; not description alone.
3. **Closing takeaway** — a usable emotional tool or a stated lesson; not an aphorism for its own sake.

Fail → back to writer/validator; no image budget is spent. Implemented as `assess_story_quality` in `script_agent.py`, enforced in the writer narrative gates, `validator_agent.validate_script`, and the locked-script path.

Run every command from the factory root unless noted.

---

## Valid choices (cheat sheet)

### Channels (`--channel`, alias `--page`)

`anna_protocol` · `master_mei` · `wonder_feed` · `down_dirty` · `ancient_knowledge` · `momma_circle` · `principles_of_wealth_finance_economics` · `endless_summer_paradise`

Default `--channel` for `main.py`: `anna_protocol`.

### Post types (`--post-type`)

| Value | What it produces |
|---|---|
| `STANDARD_QUOTE` | Long-form educational caption + image (default) |
| `SMART_BAIT` | 4-layer viral-hook image (bg + mask + bold text + logo) |
| `LONG_CAPTION_IMAGE` | Clean illustration + logo only; long FB-style caption (copyright footer) |
| `CTA_CAPTION_IMAGE` | Anna Protocol only. Clean still + logo; original Holistic Legacy caption (350–750 chars, Comment KEYWORD, no ©) |
| `CAROUSEL` | Three cohesive slides (`slide_01`…`03`) |
| `ECONOMIC_REEL` | Vertical 9:16 MP4 (TTS + Ken Burns / sequence reel) |
| `ECONOMIC_REEL_LOFI` | Isolated LOFI reel (Flux Schnell, no LoRA). Pages: `wonder_feed`, `momma_circle` |
| `WAN_REEL` | Flux LoRA stills → Wan2.2 img2vid → concat → F5 narration (no publish) |
| `REFERENCE_BASED_REELS` | Raw-footage clip + LLM hook overlay + lullaby audio. Designed for `momma_circle` |

### Formats (`--format`)

`IMAGE_AVATAR` · `IMAGE_QUOTE` · `IMAGE_BACKGROUND` · `HYBRID_VIDEO` · `TEXT_QUOTE` · `DYNAMIC_REEL` · `SEQUENCE_REEL` · `REFERENCE_BASED_REELS`

### Avatar (`--avatar`)

`ON` · `OFF`  
Defaults: `anna_protocol`/`master_mei` = ON; `wonder_feed`/`down_dirty` = OFF.

### Draw style (`--draw-style`)

`NATURAL` · `CARTOON` · `SKETCH`  
Forced: `wonder_feed` graphite types → `SKETCH`; `ancient_knowledge` / `master_mei` → `NATURAL`.

### Model API flow presets (`--model-api-flow`)

| Preset | Image | Audio | Video |
|---|---|---|---|
| `lowtier_deepinfra_elevenlabs` | DeepInfra `FLUX.1-schnell` | ElevenLabs | MoviePy |
| `hightier_together_elevenlabs` | Together `FLUX.1-dev` | ElevenLabs | MoviePy |
| `remote_gpu_pod` | Remote GPU Flux LoRA (ComfyUI) | F5-TTS | Wan2.2 img2vid |
| `remote_gpu_serverless` | Remote GPU Flux LoRA (RunPod) | F5-TTS | Wan2.2 img2vid |

Per-media overrides (win over the preset): `--img-production`, `--audio-production`, `--video-production`.

Text brain (captions/research, not image gen): `--economic` (Gemini-only) or `--premium-relay` (Gemini research + Claude 3.5 Sonnet). Do not pass both.

### Scene pacing

`--video-length SECONDS` — target reel duration (wins over page `REEL_DURATION`).  
`--scene-duration SPEC` — `fixed:4` · `progressive` · `progressive:start=4,step_every=3,step=1,cap=7.5` · `equal`  
`WAN_REEL` requires `fixed:N` (default `fixed:7`).

---

# 1. Content Generation Commands

## 1.1 Images

### STANDARD_QUOTE

`python main.py --page anna_protocol --post-type STANDARD_QUOTE --quantity 5 --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 5 long-form educational quote images for anna_protocol using Gemini for captions and Together FLUX Schnell for stills.*

`python main.py --page master_mei --post-type STANDARD_QUOTE --quantity 5 --economic --avatar ON --draw-style NATURAL --model-api-flow hightier_together_elevenlabs`  
*// Generates 5 photorealistic Master Mei quote portraits (avatar ON) with Together FLUX Dev + ElevenLabs unused for stills-only output.*

`python main.py --page down_dirty --post-type STANDARD_QUOTE --quantity 3 --avatar OFF --format IMAGE_BACKGROUND --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 3 atmospheric (no-avatar) quote backgrounds for down_dirty using Together FLUX Schnell.*

`python main.py --page anna_protocol --post-type STANDARD_QUOTE --quantity 1 --premium-relay --format TEXT_QUOTE`  
*// Generates 1 text-only quote on a solid brand backdrop (zero image-API cost); captions via Gemini research + Claude.*

### SMART_BAIT

`python main.py --page wonder_feed --post-type SMART_BAIT --quantity 10 --economic --avatar OFF --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 10 graphite-sketch viral-bait images for wonder_feed (draw-style is forced to SKETCH). Ultra-short hook overlay + sarcastic caption.*

`python main.py --page anna_protocol --post-type SMART_BAIT --quantity 3 --avatar OFF --format IMAGE_BACKGROUND --model-api-flow hightier_together_elevenlabs`  
*// Generates 3 hyper-literal SMART_BAIT backgrounds for anna_protocol with Together FLUX Dev (4-layer stack: bg + 20% mask + bold text + logo).*

### LONG_CAPTION_IMAGE

`python main.py --page anna_protocol --post-type LONG_CAPTION_IMAGE --quantity 1`  
*// Cheap default: Together FLUX Schnell, no Gemini. Logo-only still + long essay caption with copyright footer. For Comment-KEYWORD posts use CTA_CAPTION_IMAGE.*

`python main.py --page anna_protocol --post-type LONG_CAPTION_IMAGE --quantity 1 --avatar ON`  
*// Likeness path: `--avatar ON` switches the image model to Gemini `models/gemini-3-pro-image-preview` and attaches Anna's avatar PNG.*

`python main.py --page anna_protocol --post-type LONG_CAPTION_IMAGE --quantity 8 --avatar ON --image-primary models/gemini-2.5-flash-image`  
*// Same likeness path with cheaper Gemini flash-image.*

`python main.py --page ancient_knowledge --post-type LONG_CAPTION_IMAGE --quantity 40 --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 40 long-format caption images for ancient_knowledge (logo only, no baked text). Photorealistic (NATURAL is forced). Together FLUX Schnell.*

`python main.py --page wonder_feed --post-type LONG_CAPTION_IMAGE --quantity 20 --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 20 long-caption graphite illustrations for wonder_feed (SKETCH lock). Clean image + deep storytelling caption.*

### CTA_CAPTION_IMAGE (anna_protocol only)

`python main.py --page anna_protocol --post-type CTA_CAPTION_IMAGE --quantity 1`  
*// Original Holistic Legacy caption: 60% one paragraph (350–450 chars) / 40% 3–4 short paragraphs (550–750 chars). Ends with Comment KEYWORD (GINGER, SALT, DETOX, PROTOCOL, …). No copyright signature. Clean logo-only still (Together FLUX Schnell).*

`python main.py --page anna_protocol --post-type CTA_CAPTION_IMAGE --quantity 1 --avatar ON`  
*// Same caption rules with Gemini Pro likeness (`models/gemini-3-pro-image-preview`).*

`python main.py --page anna_protocol --post-type CTA_CAPTION_IMAGE --quantity 1 --cta OFF`  
*// Same mid-length format without the Comment KEYWORD invitation.*

### CAROUSEL

`--quantity` = number of **distinct carousel posts** (each pulled from a different topic in the channel topic pool / content queue; recent `content_library.json` subjects are skipped). `--carousel_quantity` = images each carousel hosts (default `3`). All slides of one carousel stay on that carousel's own topic, each from a different original photographic angle — never sub-angles of one shared subject unless you pass `--multi-variant`. Text / G-frame overlays stay **OFF** unless `--text-overlay ON` (alias `--gframe ON`).

`python main.py --page anna_protocol --post-type CAROUSEL --quantity 4 --carousel_quantity 3 --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Produces 4 distinct carousels, each with 3 slides (slide_01…03, distinct viewpoints), using DeepInfra FLUX Schnell.*

`python main.py --page anna_protocol --post-type CAROUSEL --quantity 3 --carousel_quantity 5`  
*// Produces 3 distinct carousels, each with 5 fully theme-related slides.*

`python main.py --page wonder_feed --post-type CAROUSEL --quantity 2 --economic --model-api-flow hightier_together_elevenlabs`  
*// Generates 2 graphite carousels (3 slides each by default) for wonder_feed with Together FLUX Dev.*

`python main.py --page anna_protocol --post-type CAROUSEL --quantity 3 --avatar ON`  
*// CAROUSEL honours `--avatar ON` — every slide uses Gemini Pro image (Gemini 3) with the Anna avatar reference for likeness consistency. Three different topics from the anna_protocol content queue.*

`python main.py --page anna_protocol --post-type CAROUSEL --quantity 3 --multi-variant "Celtic sea salt"`  
*// Opt-in: one core topic exploded into 3 sub-angle carousels (legacy BatchPlanner matrix).*

`python main.py --page anna_protocol --post-type CAROUSEL --quantity 2 --text-overlay ON`  
*// Explicit G-frame / headline burn-in. Default is OFF (clean photographs).*

### Image-only / caption-only / dry-run

`python main.py --page wonder_feed --post-type SMART_BAIT --quantity 5 --skip-image --economic`  
*// Writes captions + planner rows only; skips Gemini/Together image synthesis.*

`python main.py --page anna_protocol --post-type STANDARD_QUOTE --quantity 5 --skip-caption --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Synthesizes images only; skips caption/research LLM calls.*

`python main.py --page master_mei --post-type STANDARD_QUOTE --test`  
*// Dry-run: prints scaffold prompts and inventory without calling Gemini, Anthropic, or image APIs.*

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --test-images 5 --economic`  
*// Debug: writes the production script + first 5 visual prompts, generates 5 stills into outputs/ancient_knowledge/test_previews/, then exits (no TTS, MoviePy, or uploads).*

### Visual QA stills (Together FLUX + Gemini critic)

`python -m quality.VisualQA_Agent.main --list-channels`  
*// Lists registered VisualQA channel DNA keys and their attempts/approved/logs folders, then exits.*

`python -m quality.VisualQA_Agent.main --channel master_mei --beat "A gaunt worker trapped in a CRT dungeon..." --model black-forest-labs/FLUX.1-schnell`  
*// Runs the generate → Gemini-critique → auto-correct loop for a Master Mei still using Together FLUX Schnell. Writes attempts under outputs/master_mei/VisualQA_Agent_Judge/.*

`python -m quality.VisualQA_Agent.main --channel master_mei --max-retries 4 --threshold 0.85 -v`  
*// Same VisualQA loop with retry/threshold overrides and debug logging. Default FLUX model comes from TOGETHER_IMAGE_MODEL / config.*

---

## 1.2 Videos

### ECONOMIC_REEL — Together + ElevenLabs + MoviePy (cloud stack)

`python main.py --page wonder_feed --post-type ECONOMIC_REEL --quantity 3 --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 3 graphite Ken Burns reels for wonder_feed (format forced to DYNAMIC_REEL, style SKETCH). Flux Schnell stills + ElevenLabs TTS + MoviePy compile.*

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --quantity 2 --economic --model-api-flow hightier_together_elevenlabs --video-length 80 --scene-duration progressive`  
*// Generates 2 photorealistic ~80 s sequence reels for ancient_knowledge (NATURAL lock) with Together FLUX Dev, ElevenLabs VO, and progressive scene pacing.*

`python main.py --page master_mei --post-type ECONOMIC_REEL --quantity 1 --economic --avatar ON --model-api-flow hightier_together_elevenlabs --video-length 105`  
*// Generates 1 cinematic Master Mei reel (~105 s) with avatar ON, FLUX Dev stills, ElevenLabs VO, MoviePy assembly.*

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --quantity 1 --economic --cta OFF --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Same ECONOMIC_REEL path but suppresses comment-to-receive CTAs and DM links in captions.*

### ECONOMIC_REEL — Remote GPU (Flux LoRA + F5-TTS + Wan2.2)

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --quantity 1 --economic --model-api-flow remote_gpu_pod --video-length 80`  
*// Generates 1 ancient_knowledge reel on a live ComfyUI pod: Flux LoRA txt2img, F5-TTS narration, Wan2.2 img2vid. Overrides .env ENABLE_REMOTE_GPU_WORKFLOWS.*

`python main.py --page master_mei --post-type ECONOMIC_REEL --quantity 1 --economic --model-api-flow remote_gpu_serverless --avatar ON`  
*// Generates 1 Master Mei reel via RunPod serverless (same three remote workflows as remote_gpu_pod).*

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --quantity 1 --economic --model-api-flow remote_gpu_pod --img-production together/black-forest-labs/FLUX.1-schnell --audio-production elevenlabs --video-production moviepy`  
*// Mixed stack: named remote-GPU preset, then per-media overrides force Together Schnell stills, ElevenLabs VO, and MoviePy compile.*

### ECONOMIC_REEL_LOFI — emotional-first, TTS-timed riso reels

Default writer mode is `emotional`: original micro-philosophical storytelling in 7–12-word spoken beats. Gate 1 approves the narrative, then ElevenLabs fixes actual beat timing; Gate 2 approves atmospheric riso prompts before image spend. Image provider is selected with `--lofi-image-provider together|gemini` (default `together`). `--model-api-flow` does not change LOFI stills. Valid pages: `wonder_feed`, `momma_circle`.

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --quantity 3 --duration 34 --module relationship`  
*// Generates 3 emotional riso reels for wonder_feed. Relationship RAG namespace, TTS-first slide timing, atmospheric stills, captions, and watermark.*

`python main.py --page momma_circle --post-type ECONOMIC_REEL_LOFI --quantity 2 --duration 38 --module parenting`  
*// Generates 2 LOFI parenting-theme reels for momma_circle (explicit escape hatch; otherwise this page is forced to REFERENCE_BASED_REELS).*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --stills-only --module relationship --lofi-no-review`  
*// Pipeline with gates auto-passed. Default holds at Gate 1 before TTS and at Gate 2 before image generation.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --stills-only --lofi-script PATH.json`  
*// Locked script auto-clears Gate 1 only; run still holds at Gate 2 (assembled positive + negative prompts) unless `--lofi-no-review`.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --lofi-resume-from outputs/wonder_feed/clips/lofi_pipeline_….json --lofi-approve-gate 2`  
*// Resume a held run after reviewing Stage 3 assembled prompts; Stage 4 (stills, VO) runs.*

Duration: default 27s / 9 nominal beats. Actual slides follow measured VO, normally 2.5–4.0s; narration is never cut. Larger `--duration` adds beats (max 90s). Visual style is `riso_retro_flat_v4` in `core/economic_reel_lofi/style_modules/`.

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --script-only --module relationship --lofi-theme hope`  
*// One Gemini 3.8 Flash Few-Shot writer call + RAG only. Immediately prints and saves strict SCRIPT_CANDIDATE_JSON with spoken text and visual_concept per beat; no judge, repair, TTS, image, or render APIs.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --script-only --module parenting`
*// Same single-pass writer using the parenting niche preset (golden examples + warm golden-hour aesthetic wrapper).*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --lofi-image-provider gemini`
*// Uses Google Gemini Flash Image for Stage 4 while retaining the same riso prompt and QA pipeline.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --script-only --lofi-mode theme --module relationship --lofi-theme hope`
*// Explicitly selects the slower duration-aware theme composer.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --script-only --lofi-mode quote --lofi-seed-quote "A concept to develop"`
*// Develops a seed concept through a concrete parable: setup → conflict → retreat → workable equilibrium.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --script-only --lofi-mode paraphrase --lofi-aphorism-id more_tears_than_smiles`
*// One light, structure-preserving paraphrase from `store/aphorism_bank.json`; no story expansion. Omit `--lofi-aphorism-id` to pick a random unused bank entry each run.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --test-preview`  
*// Single-image aesthetic review (Flux Schnell + production grading + LOFI caption typography + watermark). No script, video, publish, or RAG history write.*

`python main.py --page wonder_feed --post-type ECONOMIC_REEL_LOFI --test-preview --prompt "A couple sitting in silence at opposite ends of a long kitchen table at dusk"`  
*// Same LOFI still preview with a custom scene description instead of the seed-aligned baseline prompt.*

### WAN_REEL — Flux LoRA → Wan2.2 → F5 (test render, no publish)

Requires `--scene-duration fixed:N` (progressive is ECONOMIC_REEL-only). Default length 70 s / `fixed:7` from ancient_knowledge page config.

`python main.py --page ancient_knowledge --post-type WAN_REEL --quantity 1 --model-api-flow remote_gpu_pod --video-length 70 --scene-duration fixed:7 "Göbekli Tepe — the 12,000-year-old temple that rewrote human history"`  
*// Cost/quality test: Flux LoRA stills → Wan2.2 clips at 7 s each → concat → F5 narration. Does not upload to YouTube/B2. Optional positional topic overrides the default Göbekli Tepe seed.*

`python main.py --page ancient_knowledge --post-type WAN_REEL --model-api-flow remote_gpu_serverless --video-length 70 --scene-duration fixed:7`  
*// Same WAN_REEL test render on RunPod serverless instead of a dedicated ComfyUI pod.*

### REFERENCE_BASED_REELS — raw footage (momma_circle)

`momma_circle` **forces** this post type unless you pass `--post-type ECONOMIC_REEL_LOFI`.

`python main.py --page momma_circle --quantity 5 --clip-duration 37`  
*// Extracts 5 raw-footage clips, overlays an LLM hook, blends lullaby ambient audio. Clip length clamped to page CLIP_DURATION_MIN_S / MAX_S (default midpoint ~37 s). Uploads to B2.*

`python main.py --page momma_circle --post-type REFERENCE_BASED_REELS --quantity 3 --clip-duration 45 --economic`  
*// Same reference-reel path with an explicit post type and a 45 s target clip length.*

### HYBRID_VIDEO (7 s Ken Burns loop from a still)

`python main.py --page anna_protocol --post-type STANDARD_QUOTE --format HYBRID_VIDEO --quantity 2 --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 2 stills then compiles each into a 7-second Ken Burns zoom loop (video_converter), not a full ECONOMIC_REEL.*

### Generate + auto-schedule to YouTube

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --quantity 4 --economic --publish-youtube --interval-hours 12 --random-delay-max-minutes 60 --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Generates 4 reels then uploads them as YouTube Scheduled (private + publishAt). Slot i = now + 12 h × (i+1) plus 0–60 min jitter. Uses credentials/tokens/youtube_token_ancient_knowledge.json. Alias: --upload-youtube.*

`python main.py --page master_mei --post-type ECONOMIC_REEL --quantity 2 --economic --publish-youtube --interval-hours 6 --model-api-flow hightier_together_elevenlabs`  
*// Generates 2 Master Mei reels and drip-schedules them on YouTube at 6-hour intervals.*

`--yt-privacy` is ignored when `--publish-youtube` is set (forced private + publishAt). `--schedule-uploads` is a no-op legacy alias.

### Render gate (inspect stills before compile)

`python main.py --page ancient_knowledge --post-type ECONOMIC_REEL --quantity 1 --economic --render-approval-required --model-api-flow lowtier_deepinfra_elevenlabs`  
*// After scene_01.png… are written to outputs/ancient_knowledge/assets/<episode_id>/, pauses for [ENTER] before audio/video compile. Ctrl+C aborts.*

---

## 1.3 Avatars

Avatar likeness is a generation flag on image/reel post types, not a separate post type. Reference PNGs live in `channels_config/<page>/avatar_reference/`.

`python main.py --page anna_protocol --post-type STANDARD_QUOTE --quantity 3 --avatar ON --format IMAGE_AVATAR`  
*// `--avatar ON` selects Gemini Pro image + Anna likeness. Omit the flag to stay on Together FLUX.*

`python main.py --page master_mei --post-type ECONOMIC_REEL --quantity 1 --avatar ON --economic --model-api-flow remote_gpu_pod`  
*// Generates 1 Master Mei reel with avatar ON (page default if omitted). Remote GPU Flux LoRA stills include the likeness reference when the channel LoRA/trigger is configured.*

`python main.py --page anna_protocol --post-type SMART_BAIT --quantity 5 --avatar OFF --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Bypasses the avatar pipeline — purely atmospheric SMART_BAIT backgrounds (SMART_BAIT still forces avatar OFF; LONG_CAPTION_IMAGE, CTA_CAPTION_IMAGE and CAROUSEL honour `--avatar`).*

`python main.py --page down_dirty --post-type STANDARD_QUOTE --quantity 4 --avatar OFF --draw-style NATURAL --economic --model-api-flow lowtier_deepinfra_elevenlabs`  
*// Atmospheric cinematic stills for down_dirty (page default avatar OFF; no likeness file required).*

---

# 2. Scheduling & Publishing Commands

## 2.1 Facebook

### Reels (local `outputs/<channel>/clips/*.mp4` → Meta Business Suite)

Requires a Dolphin{anty}/Chrome profile already open on `business.facebook.com` (English UI). CDP can be a port (`56076`) or a full URL (`http://127.0.0.1:56076`).

`python -m agents.posting.facebook_scheduler.reels_scheduler --channel master_mei --cdp 56076 --max 10`  
*// Runs the Facebook Reels scheduler for master_mei using CDP port 56076, processing a maximum of 10 clips from the queue.*

`python -m agents.posting.facebook_scheduler.reels_scheduler --channel ancient_knowledge --cdp 9222 --max 5`  
*// Schedules up to 5 pending ancient_knowledge MP4s via the dedicated reels_composer endpoint.*

`python -m agents.posting.facebook_scheduler.reels_scheduler --channel wonder_feed --dry-run`  
*// Scans outputs/wonder_feed/clips/*.mp4, prints the schedule plan and first slot, and exits with no browser clicks or facebook_history.json writes.*

`python -m agents.posting.facebook_scheduler.reels_scheduler --channel master_mei --max 3 --no-move`  
*// Schedules up to 3 Master Mei reels but leaves files in clips/ after success (still writes facebook_history.json). Default is to move completed files to clips/posted_facebook/.*

`python -m agents.posting.facebook_scheduler.reels_scheduler --channel momma_circle`  
*// Live Reels run for momma_circle; CDP auto-detects Dolphin / config.CDP_ENDPOINT (default http://localhost:9222). First slot = now + 25–60 min; later slots = last + 4 h + 0–60 min jitter.*

### Background text posts (Google Sheet queue → Facebook composer)

Sheet-driven (Momma Circle quotes). No `--channel` flag — override the sheet instead.

`python agents/posting/facebook_scheduler/main.py --dry-run`  
*// Simulates the sheet queue: prints pending posts and scheduled times. No browser clicks, no sheet writes.*

`python agents/posting/facebook_scheduler/main.py --fill-times-only --interval 3`  
*// Fills empty Column B datetimes at 3-hour increments, then exits without posting.*

`python agents/posting/facebook_scheduler/main.py --cdp http://localhost:9222`  
*// Live run: attaches to Chrome via CDP, reads the Ready_to_post worksheet, and schedules each PENDING background text post.*

`python agents/posting/facebook_scheduler/main.py --sheet-id YOUR_SHEET_ID --worksheet Ready_to_post --url https://www.facebook.com/MommaCircle`  
*// Live run against an explicit spreadsheet/tab, navigating to the Momma Circle page first.*

`python agents/posting/facebook_scheduler/main.py --record --workflow credentials/recorded_workflow.json`  
*// Records the English Meta Business Suite workflow (Open Backgrounds → … → Open Scheduler) and saves it for live replay.*

`python agents/posting/facebook_scheduler/main.py --set-bg`  
*// Training mode: interactively pick a background tile once and save it to credentials/bg_config.json.*

`python agents/posting/facebook_scheduler/main.py --show-bg`  
*// Prints the currently saved background tile config and exits.*

`python agents/posting/facebook_scheduler/record_mode.py --url https://www.facebook.com/MommaCircle --output recorded_workflow.py`  
*// Launches Playwright Codegen pointed at the Momma Circle page so selectors can be recaptured after a Facebook UI change.*

`python channels_config/momma_circle/tests/test_momma_circle_post.py --dry-run`
*// Verifies the Momma Circle Graph API token/page ID without publishing.*

`python channels_config/momma_circle/tests/test_momma_circle_post.py --message "Connection test from the factory"`
*// Publishes a one-off Graph API test post to the Momma Circle page (uses FB_MOMMA_CIRCLE_* env vars).*

---

## 2.2 YouTube

There is no `youtube_scheduler.shorts_scheduler` module. Use `agents/posting/publish_existing.py` for already-rendered MP4s, or `main.py --publish-youtube` (section 1.2) to generate-and-upload in one shot. Tokens: `credentials/tokens/youtube_token_{page}.json`.

**Global safety cap:** every YouTube scheduling run stops after `MAX_DAILY_UPLOADS` successful uploads (default **20**). Override with `--limit N`. Remaining queue items stay deferred for the next run. Notice: `[GLOBAL SAFETY] Reached daily upload quota cap (20 videos). Remaining queue deferred for next run.`

### Endless Summer Paradise (library ingest)

`python channels_config/endless_summer_paradise/esp_main.py scan`  
*// Scans Endless_Summers_Paradise - Production; keeps only duration > 40s; maps global_video_library.json; stages unmatched files to outputs/endless_summer_paradise/needs_metadata/.*

`python channels_config/endless_summer_paradise/esp_main.py queue`  
*// Lists the ready YouTube schedule queue built by the last scan.*

`python channels_config/endless_summer_paradise/esp_main.py schedule --dry-run`  
*// Previews private+publishAt slots for ready queue items (no upload).*

`python channels_config/endless_summer_paradise/esp_main.py schedule --limit 3`  
*// Uploads up to 3 ready masters to the Endless Summer Paradise YouTube channel (token: youtube_token_endless_summer_paradise.json).*

`python agents/posting/publish_existing.py --page ancient_knowledge --video outputs/ancient_knowledge/clips/reel_example_v01.mp4 --schedule --interval 6`  
*// Uploads one Short and auto-picks the next free slot, then spaces further files by 6 hours (default YT_POST_INTERVAL_HOURS=6). Caption/tags load from content_library.json when omitted.*

`python agents/posting/publish_existing.py --page ancient_knowledge --video reel_v01.mp4 reel_v02.mp4 reel_v03.mp4 --schedule`  
*// Uploads and drip-schedules three Shorts for ancient_knowledge (privacyStatus=private + publishAt).*

`python agents/posting/publish_existing.py --page master_mei --video outputs/master_mei/clips/reel_example_v01.mp4 --publish-at "2026-08-13 14:00"`  
*// Schedules a single Master Mei Short at an explicit UTC datetime (YYYY-MM-DD HH:MM).*

`python agents/posting/publish_existing.py --page wonder_feed --video outputs/wonder_feed/clips/reel_example_v01.mp4 --privacy unlisted --dry-run`  
*// Previews an immediate unlisted upload without calling the YouTube API.*

`python agents/posting/publish_existing.py --page ancient_knowledge --resume-youtube-queue`  
*// Reads credentials/pending_youtube_uploads.json and publishes videos queued after a previous daily-quota (~20/channel/day) stop, scoped to ancient_knowledge.*

`python agents/posting/publish_existing.py --resume-youtube-queue`  
*// Same pending-queue resume for ALL pages (omit --page).*

`python main.py --page ancient_knowledge --resume-youtube-queue`  
*// Factory entry-point alias: resumes the pending YouTube queue then exits (does not generate new content). Pass --page to scope; omit it to resume every page.*

`python agents/posting/publish_existing.py --page ancient_knowledge --update-only --video-id VIDEO_ID --title "Corrected Title"`  
*// Patches title/description/tags of an already-uploaded video (requires --video-id). Library caption is used if --caption is omitted.*

`python -m agents.posting.youtube_publisher path/to/clip.mp4 --page master_mei --title "Test Upload" --privacy unlisted --schedule`  
*// Standalone single-file test upload. --schedule uses the smart next-slot picker; default privacy is unlisted.*

### Principles of Wealth (library ingest)

Source files stay on the production drive. Entry point is `channels_config/principles_of_wealth_finance_economics/wealth_main.py` (not `main.py` generation). Tokens: `credentials/tokens/youtube_token_principles_of_wealth_finance_economics.json`.

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py scan`  
*// Matches `Ray Dalio epN.mp4` longs, up to 10 Shorts per episode (`Short1`-`Short10` for Ep 1, `EpN.1`-`EpN.10` for Ep 2+), and `ThumbN`. Writes `outputs/principles_of_wealth_finance_economics/wealth_asset_map.json`.*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py process --episodes 1-2 --dry-run`  
*// Preview the FFmpeg uniqueness pass (2 px crop, 0.5% timing shift, metadata strip) and thumbnail re-sign.*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py process --episodes 1 --hw-encode`  
*// Re-sign episode 1 into `{source}/Processed/` using NVIDIA nvenc when available (default is libx264 ultrafast).*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py publish --mode longs --episodes 1-10 --dry-run`  
*// Preview weekly long slots (private + publishAt). Default: next Thursday 18:00 UTC, then +7 days. Reads `last_long_scheduled_at` when `--start-date` is omitted.*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py publish --mode longs --episodes 1-10 --start-date 2026-08-27 --time-utc 18:00`  
*// Schedule ACT I longs one per week as private + publishAt. Writes `long_video_id` and `last_long_scheduled_at`.*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py publish --mode shorts --episodes 1 --start-date 2026-08-25 --time-utc 16:00`  
*// Schedule that episode's Shorts one per day. Each Short links to the parent `long_video_id` (works while the long is still private/scheduled).*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py publish --mode shorts --episodes 1 --no-schedule`  
*// Upload Shorts immediately (no publishAt). Scheduled mode is the default (`--schedule`).*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py playlists`  
*// Create/sync ACT I–III playlists and insert longs in chronological episode order (not newest-first).*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py status`  
*// Source-file match vs uploaded IDs from `wealth_publish_state.json`.*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py metadata --dry-run`  
*// Export `outputs/principles_of_wealth_finance_economics/youtube_seo_pack.json` and print the SEO rewrite table. Does not call YouTube.*

`python channels_config/principles_of_wealth_finance_economics/wealth_main.py metadata --apply`  
*// Patch titles, descriptions, and tags on already-uploaded longs/Shorts, and rename existing ACT playlists by ID (no duplicates). Use `--mode longs|shorts|playlists` and `--episodes 1-10` to scope.*

`python agents/posting/publish_existing.py --page principles_of_wealth_finance_economics --video path/to/clip.mp4 --privacy unlisted --dry-run`  
*// Fallback single-file upload through the shared publisher (no ACT catalog / relatedVideoId wiring).*

---

## 2.3 Pinterest

Channel context (`--channel`) loads `channels_config/<id>/` and routes inventory to `outputs/<id>/`. Without `--channel`, pass `--env` / `--inventory-dir`.

`python agents/posting/pinterest_oauth.py`  
*// One-shot OAuth: opens the Pinterest consent page, captures the redirect on localhost:8080, writes PINTEREST_ACCESS_TOKEN / REFRESH_TOKEN to .env, and lists boards.*

`python agents/posting/pinterest_oauth.py --channel ancient_knowledge`  
*// Same OAuth flow, writing tokens to `channels_config/ancient_knowledge/.env.ancient_knowledge` and matching/creating the Ancient Knowledge board.*

`python agents/posting/pinterest_main.py validate-token --channel anna_protocol`  
*// Tests the Pinterest access token for anna_protocol (auto-refresh if expired).*

`python agents/posting/pinterest_main.py validate-token --channel ancient_knowledge`  
*// Tests the isolated Ancient Knowledge token (`channels_config/ancient_knowledge/.env.ancient_knowledge`); refreshes on 401.*

`python agents/posting/pinterest_main.py status --channel anna_protocol`  
*// Prints queue depth, publish history, and last legacy-ledger pins for anna_protocol.*

`python agents/posting/pinterest_main.py status --channel ancient_knowledge`  
*// Isolated queue/history under `outputs/ancient_knowledge/`.*

`python agents/posting/pinterest_main.py check-readiness --channel anna_protocol`  
*// Pre-flight checklist before the first publish (exit 0 = ready).*

`python agents/posting/pinterest_main.py sync --channel anna_protocol --limit 50`  
*// Builds/repairs master_inventory.json: merge library, fix local paths, inject Pinterest title/caption/visual_hook via Claude (max 50 entries).*

`python agents/posting/pinterest_main.py sync --channel wonder_feed --no-ai --dry-run`  
*// Preview inventory sync using fast regex/templates instead of Claude; no writes.*

`python agents/posting/pinterest_main.py sync --channel anna_protocol --force --ai-delay 0.8`  
*// Re-generates Pinterest metadata even when fields already exist, with 0.8 s between Claude calls.*

`python agents/posting/pinterest_main.py transform --channel anna_protocol --method blurred_padding`  
*// Builds a 2:3 sales pin (1000×1500) for the first unposted inventory entry. Methods: blurred_padding (default), center_crop.*

`python agents/posting/pinterest_main.py transform --channel anna_protocol --post-id POST_ID --method center_crop --open`  
*// Transforms a specific inventory post_id and opens the output file in Explorer.*

`python agents/posting/pinterest_main.py schedule --channel anna_protocol --quantity 5 --min-hours 3 --max-hours 6`  
*// Publishes 5 pins live with human-mimic delays randomly between 3 and 6 hours. Default quantity is a random 3–5 if -n is omitted.*

`python agents/posting/pinterest_main.py schedule --channel ancient_knowledge --quantity 20`  
*// Recycles Ancient Knowledge library stills + unpaired clips, auto-creates the "Ancient Knowledge" board if missing, then drip-publishes 20 pins.*

`python agents/posting/pinterest_main.py schedule --channel wonder_feed -n 3 --dry-run --no-wait`  
*// Full schedule pipeline without calling the Pinterest API, skipping sleep intervals (testing only).*

`python agents/posting/pinterest_main.py export --channel anna_protocol --posts-per-day 3 --slots 08:00 13:00 18:00 --start 2026-08-13 --mark-exported`  
*// Writes a Post Planner bulk-upload CSV (3 posts/day at those slots starting 2026-08-13) and marks rows exported in master_inventory.*

`python sync_drive_assets.py --limit 20 --dry-run`  
*// Legacy library repair: preview Pinterest title/caption/visual_hook injection for up to 20 posts (no writes). Prefer `python agents/posting/pinterest_main.py sync` for channel-scoped runs.*

---

## 2.4 Backblaze B2 / PostPlanner

`python upload_clips_to_b2.py --page wonder_feed --clips-dir outputs/wonder_feed/clips/ --interval 60`  
*// Uploads every MP4 in the wonder_feed clips folder to B2 and writes a PostPlanner XLSX with 60-minute posting slots (default page is wonder_feed).*

`python upload_clips_to_b2.py --page ancient_knowledge --dry-run`  
*// Prints the ancient_knowledge clip list and intended B2 keys without uploading or writing XLSX.*

`python upload_clips_to_b2.py --page master_mei --captions-json captions.json --caption "Fallback caption" --bucket MediaupscaleStorage`  
*// Uploads Master Mei clips with per-file captions from JSON (filename → caption); --caption is the fallback. Overrides B2_BUCKET_NAME.*

`python fix_uploaded_titles.py --page ancient_knowledge`  
*// Idempotent: uploads reel MP4s under outputs/ancient_knowledge/ to B2 and patches MEDIA URL columns in postplan_*.xlsx / automated_bulk_posts_import.xlsx from ImgBB image links to live .mp4 URLs.*

`python fix_uploaded_titles.py --page wonder_feed --dry-run`  
*// Preview the B2 upload + Excel MEDIA URL patch for wonder_feed without writing.*

`python fix_postplanner_captions.py --dry-run`  
*// Preview stripping raw LLM JSON/markdown from CAPTION cells in the default ancient_knowledge postplan workbook.*

`python fix_postplanner_captions.py outputs/ancient_knowledge/postplanner/postplan_20260802_212650.xlsx`  
*// Cleans caption_body text in-place for the given PostPlanner XLSX.*

---

# 3. Utility, QA & Maintenance Commands

`python tests/test_remote_gpu.py --dry-run`  
*// Loads ComfyUI workflow JSONs under infra/runpod/workflows/, prints node counts and config, and makes no HTTP calls.*

`python tests/test_remote_gpu.py --image --base-url http://127.0.0.1:8188 --mode comfyui`  
*// Live Flux Dev txt2img smoke test against a local/pod ComfyUI. Saves under outputs/remote_gpu_tests/.*

`python tests/test_remote_gpu.py --audio --ref-audio path/to/voice.wav --base-url http://<POD_IP>:8188`  
*// Live F5-TTS txt2audio test, uploading the reference WAV to ComfyUI input/.*

`python tests/test_remote_gpu.py --video --image-file path/to/frame.png --duration 5 --video-size 512`  
*// Live Wan2.2 img2vid test from a local still (required unless --dry-run).*

`python tests/test_remote_gpu.py --all --base-url http://<POD_IP>:8188 --mode runpod --timeout 600`  
*// Runs image + audio + video remote-GPU tests on RunPod with a 600 s job timeout.*

`python tests/test_audio_engine.py --page master_mei`  
*// Mixes a dummy/test voiceover with page BGM/SFX for master_mei and writes an audio-test artifact (no full reel compile).*

`python patch_last_30_audio.py --n 30 --clips-dir outputs/master_mei/clips`  
*// Remuxes the latest 30 Master Mei MP4s with BGM volume 0.24 (−20%), copying video bitstream (`-c:v copy`). Writes .bak_pre_bgm024 backups unless --no-backup.*

`python patch_last_30_audio.py --n 10 --dry-run`  
*// Resolves the latest 10 clip stems and mix sources without remuxing.*

---

## Common `main.py` flag map

| Flag | Aliases | Default | Purpose |
|---|---|---|---|
| `topic` (positional) | — | AI-chosen | Optional subject string |
| `--page` | — | `anna_protocol` | Persona / output namespace |
| `--post-type` | — | `STANDARD_QUOTE` | Pipeline selector |
| `--quantity` | `--count`, `-n` | `1` | Concurrent variants |
| `--format` | — | page default | Output container |
| `--avatar` | — | page default | Likeness ON/OFF |
| `--draw-style` | — | `SKETCH` | NATURAL / CARTOON / SKETCH |
| `--model-api-flow` | — | `.env` | Provider preset per media type |
| `--img-production` | — | — | Image provider/model override (`gemini[/model]`, `together[/model]`, `remote_gpu`) |
| `--image-primary` | — | Flux / page override | Image SKU. `--avatar ON` implies Gemini Pro unless this flag is set |
| `--audio-production` | — | — | Audio provider override |
| `--video-production` | — | — | Video provider override |
| `--economic` | — | page/env | Gemini-only text brain |
| `--premium-relay` | — | — | Gemini research + Claude captions |
| `--cta` | — | `ON` | Inject/suppress CTA keywords |
| `--duration` | — | `34` | LOFI reel length 30–38 s |
| `--module` | — | `relationship` | LOFI RAG namespace |
| `--clip-duration` | — | page midpoint | REFERENCE_BASED_REELS length |
| `--video-length` | — | page `REEL_DURATION` | Shared scene-pacing target |
| `--scene-duration` | — | page/factory | `fixed:N` / `progressive` / `equal` |
| `--publish-youtube` | `--upload-youtube` | off | Schedule compiled reels on YT |
| `--interval-hours` | — | `12.0` | YT slot spacing with `--publish-youtube` |
| `--limit` | — | `20` (`MAX_DAILY_UPLOADS`) | Global YT upload safety cap per run |
| `--random-delay-max-minutes` | — | `60` | Extra jitter on YT slots |
| `--resume-youtube-queue` | — | off | Drain pending YT uploads, then exit |
| `--skip-image` / `--skip-caption` | — | off | Partial pipeline |
| `--test` | — | off | Print scaffolds, no APIs |
| `--test-images N` | — | — | First-N stills debug, then exit |
| `--test-preview` | — | off | LOFI single-still aesthetic review |
| `--stills-only` | — | off | LOFI script + 9 stills + QA, no TTS/video |
| `--script-only` | — | off | LOFI writer+validator only, no images |
| `--prompt` | — | seed baseline | Custom LOFI `--test-preview` scene |
| `--render-approval-required` | — | off | Pause after stills, before compile |
