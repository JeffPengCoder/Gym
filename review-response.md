# PR 2295: Remaining Inline Review Responses

This note records the responses and completed local implementation for the
three remaining inline review comments. The six broader PR-scope questions
have already been answered separately.

## 1. Use the standard Gym judge abstraction

Comment: <https://github.com/NVIDIA-NeMo/Gym/pull/2295#discussion_r3852753129>

### Response

Done. WebVoyager judging now uses the standard `/verify` endpoint and
`nemo_gym.judge.call_judge()` for the model call. The custom
`/verify_webvoyager` route and agent-side retry loop have been removed.

Judge transport, timeout, authentication, HTTP, and response-schema failures
are classified as `judge_failed` by `judge_failsafe`, allowing
`gym eval reverify --judge-failed-only` to rerun only the judge. Captured
screenshots, page URLs, and the final answer are retained in the rollout
response so reverification does not repeat live-site interaction. A
successfully received but unparseable verdict remains a benchmark outcome
rather than an infrastructure failure.

The persisted evidence is compact: screenshots already present in the rollout
trajectory are referenced by index, and only boundary screenshots absent from
the trajectory are inlined. Initial `/verify` sends a trajectory-free response
plus one top-level evidence copy; the verifier response does not echo those
top-level screenshot fields.

### Implementation

- Replace the direct judge-model POST with `nemo_gym.judge.call_judge()`.
- Remove the custom `/verify_webvoyager` endpoint and the WebAgent judge retry
  loop.
- Send the completed rollout to the judge resource server through `/verify`.
- Persist compact judge evidence inside the response so generic
  reverification can reconstruct the request from `input + response`.
- Add tests for judge transport failures, response parsing, failure-sidecar
  routing, and judge-only reverification without browser replay.

## 2. Make the locked CLI environment reproducible

Comment: <https://github.com/NVIDIA-NeMo/Gym/pull/2295#discussion_r3852766146>

### Response

Done. The undefined `PINNED_GYM` placeholder is replaced with an exact
uv-lock-based workflow: `uv lock --check`, `uv sync --frozen --extra dev`, and
commands executed through `./.venv/bin/gym`.

The implementation also follows the OSWorld preparation pattern. Directly
invoking `benchmarks/webvoyager/prepare.py` validates the selected
dataset/profile and generates a private, gitignored `env.yaml`. The
documentation covers
both the normal one-shot `gym eval run` workflow, which starts and shuts down
its servers automatically, and the advanced `gym env prefetch` /
`gym env start` / `gym eval run --no-serve` workflow. A foreground
`gym env start` is terminated with Ctrl-C; externally managed vLLM and proxy
services remain outside Gym's lifecycle.

### Implementation

- Document `uv lock --check` and `uv sync --frozen --extra dev` as the source
  of the root Gym CLI environment.
- Extend the direct `prepare.py` entrypoint to select a profile, validate its
  source, and generate a mode-`0600`, gitignored benchmark-local `env.yaml`.
- Keep `gym eval prepare --benchmark webvoyager` as the standard dataset-only
  preparation API.
- Document both the one-shot and persistent-server lifecycles, including how
  each lifecycle ends.
- Keep credentials in environment variables rather than checked-in files.

## 3. Give native WebVoyager a dedicated browser resource server

Comment: <https://github.com/NVIDIA-NeMo/Gym/pull/2295#discussion_r3852840563>

### Response

Done. Native WebVoyager now has a dedicated `webvoyager_browser` resources
server. Proxy selection, CAPTCHA handling, WebVoyager init scripts, and
public-site navigation policy live there.

The browser implementation is not duplicated: headed Chromium lifecycle,
Playwright/PyAutoGUI actions, screenshots, and artifact handling live in the
shared `nemo_gym.web.native_browser` layer. The existing `native_web` server is
scoped to WebArena and VisualWebArena and no longer branches on
`WebBenchmark.WEBVOYAGER`.

### Implementation

- Add a dedicated `resources_servers/webvoyager_browser` component.
- Extract reusable headed-Chromium and PyAutoGUI mechanics behind a shared
  native-browser policy boundary rather than copying the driver.
- Move proxy, CAPTCHA, public-site navigation, and WebVoyager-only lifecycle
  behavior into the dedicated component.
- Keep `native_web` responsible for WebArena and VisualWebArena behavior.
- Give `web_agent` an explicit environment-server reference while retaining
  the conventional resources-server reference for verification.
- Keep the legacy 643-task BrowserGym profile on `browsergym_web` while using
  the same standard WebVoyager judge.

## Planned commit sequence

1. `refactor(web): split the WebVoyager browser environment`
2. `fix(webvoyager): use standard Gym judge verification`
3. `docs(webvoyager): add reproducible uv and environment workflow`

These changes do not intentionally alter benchmark tasks, prompts, tool
schemas, or model-serving recipes.

## Local implementation status

The working tree now implements the plan above:

- `webvoyager_browser` owns native WebVoyager proxy/CAPTCHA/public-site
  behavior; `native_web` accepts only WebArena and VisualWebArena.
- Both resource servers reuse `nemo_gym.web.native_browser`, which contains
  only headed-browser lifecycle, action, screenshot, and artifact mechanics.
- `web_agent` uses an explicit browser `environment_server` when it differs
  from the canonical verification `resources_server`.
- WebVoyager judging uses only `/verify`, `call_judge()`, `judge_failsafe`, and
  a `STATELESS` reverification declaration. The rollout response carries the
  compact evidence required by generic judge-only reverification; trajectory
  screenshots are not duplicated in the stored evidence.
- Direct `benchmarks/webvoyager/prepare.py` creates a private `env.yaml`, while
  `gym eval prepare --benchmark webvoyager` remains dataset-only. The script
  prints absolute, copyable commands for the locked repository CLI.
- The locked `.venv/bin/gym` one-shot and persistent-server workflows are
  documented, including Ctrl-C shutdown semantics.

Local validation completed with:

- `uv lock --check` and `uv sync --frozen --extra dev`;
- 433 web, browser-resource, judge, dataset, and generic-reverification tests
  during implementation, followed by a final 300-test focused regression run
  after the live-run fixes;
- Ruff and `git diff --check`;
- generation of both the 643-task legacy and 552-task native private
  compositions; and
- `gym env validate` for both generated compositions.

The reverify regression builds the standard payload from a materialized input
and persisted rollout response, reconstructs the exact screenshot sequence,
and reaches only the judge model call—no browser session is seeded or replayed.

A hash-sealed version of this review-fix working tree was also run over the
complete maintained native WebVoyager population on the target Linux/Enroot
runtime. It closed **552/552** unique tasks with zero missing, duplicate,
unexpected, invalid, or unresolved infrastructure rows and an empty retry set:

- candidate: **421/552 = 76.27%**;
- previous Gym control: **428/552 = 77.54%** (-7 tasks / -1.27 pp);
- maintained reference golden: **429/552 = 77.72%** (-8 tasks / -1.45 pp).

The full run exercised the dedicated `webvoyager_browser`, the standard judge
`/verify` route, the pinned 16,384-token policy-output cap, proxy/CAPTCHA
handling, and judge-only evidence persistence. Raw outbound audits covered
1,317 policy requests in the corrected waves: every request carried
`max_tokens=16384` and
`chat_template_kwargs.truncate_history_thinking=false`, with no malformed
trace rows or contract mismatch.

The population exposed and motivated regression-covered fixes for CAPTCHA
solver/browser proxy identity, futile retries of deterministic context-limit
errors, a missing output cap in distributed worker launch commands,
post-action browser-target closure classification, and Responses API
`incomplete/max_output_tokens` handling. The final source generation passed
production agent/browser imports plus 5 focused agent and 14 CAPTCHA tests
inside the same Linux/Enroot image used by the benchmark. None of these fixes
rewrites task goals, prompts, tool schemas, judge verdicts, or successful
rollout rows.
