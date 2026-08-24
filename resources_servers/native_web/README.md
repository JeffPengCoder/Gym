# Native visual web resource server

This is the native Nano Omni backend for WebArena, VisualWebArena, and
WebVoyager. It is a sibling of `browsergym_web`, not a subclass or replacement:
both implement Gym's common session/step/evaluate protocol, while preserving
different benchmark action and evaluator semantics.

Playwright owns Chromium contexts, pages, navigation, and tabs. PyAutoGUI owns
visible coordinate input and full-display screenshots. The Responses agent
owns the model loop. Benchmark evaluators are selected separately:

- WebVoyager returns browser evidence to the post-close Gemini judge;
- WebArena evaluates string, URL, and `program_html` targets against the live
  self-hosted sites;
- VisualWebArena adds page-image/VQA/SSIM evaluation;
- WebArena-family collision plans merge API snapshots (for example Shopping
  orders and reviews) with live-page snapshots (`program_html` and page-image
  targets) before and after each rollout, so one task does not silently
  validate another task's mutation.

The WebArena-family evaluator source is pinned to the native reference at
`3b775dc538931ead0cb6b4922349da9c6d493dab`. See
`reference_evaluation/PROVENANCE.md`.

One process supports exactly one live session on one X display. Horizontal
parallelism is obtained by launching isolated server replicas with distinct
`DISPLAY` values. This prevents clicks from one rollout entering another
rollout's browser. Distributed Gym workers remain supported; stateful
WebArena-family replicas additionally need isolated site deployments or an
external scheduling/reset policy that prevents conflicting writers.

The server expects Xvfb to be started by the container/job entrypoint before
the Python server starts. It intentionally fails before creating a browser if
`DISPLAY` is absent. The runtime image must also provide `xclip` for Unicode
clipboard input and the fonts required by the benchmark pages; these are OS
packages and are intentionally not hidden behind Python dependencies.

With `CAPSOLVER_API_KEY` and `WA_CAPTCHA_PROVIDER=capsolver`, the built-in
solver handles visible Turnstile and reCAPTCHA v2 widgets. A reviewed custom
solver can instead be injected through
`WA_CAPTCHA_SOLVER=module.path:factory`; the factory returns an object exposing
`maybe_solve(page, phase=...)`.

CAPTCHA/proxy helpers are scoped to WebVoyager. Local WebArena-family tasks use
the `WA_SHOPPING`, `WA_REDDIT`, `WA_GITLAB`, `WA_WIKIPEDIA`, `WA_MAP`,
`WA_CLASSIFIEDS`, and related deployment URLs, plus the public benchmark login
accounts (overridable with `WA_<SITE>_USERNAME/PASSWORD`). Tasks requiring a
model-backed local evaluator also require `WEBARENA_JUDGE_API_KEY`; model,
base URL, and timeout follow the native `WEBARENA_JUDGE_*` environment
contract. The pinned Nano Omni WebArena and VisualWebArena launchers resolve
that contract to `WEBARENA_JUDGE_MODEL=us/azure/openai/gpt-4.1`,
`WEBARENA_JUDGE_BASE_URL=https://inference-api.nvidia.com` (without `/v1`),
and `WEBARENA_JUDGE_TIMEOUT=120`. Keep these deployment values explicit; the
evaluator module's generic fallback model is not the aligned profile.

VisualWebArena reference images stay outside JSONL. Because JSONL paths begin
with `visualwebarena/...`, mount the parent directory containing that tree and
set it as the same `task_image_root` in both the native resource-server and
web-agent profiles. Relative metadata paths are resolved only below that root
and are size/type checked before model or evaluator use. Task images stay in
the first model turn for the full rollout; `max_image_history` compacts only
browser screenshots, matching the pinned native agent.

Profiles:

- `resources_servers/native_web/configs/native_web.yaml`: WebVoyager;
- `resources_servers/native_web/configs/native_webarena.yaml`: WebArena;
- `resources_servers/native_web/configs/native_visualwebarena.yaml`:
  VisualWebArena.

The server logs redacted lifecycle events through the
`nemo_gym.resources_servers.native_web` and
`nemo_gym.resources_servers.native_web.captcha` loggers. Enable Gym verbose
logging for challenge scans. INFO records detections, provider task creation,
successful injection, browser actions, screenshots and failures; DEBUG also
records scans where no challenge was present. Keys, proxy credentials,
provider task IDs, solution tokens, screenshot payloads and URL query strings
are never logged.
