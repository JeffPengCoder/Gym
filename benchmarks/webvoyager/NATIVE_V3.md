# Native Nano Omni v3 WebVoyager

This profile reproduces the maintained screenshot/tool-call WebVoyager route
inside NeMo Gym. It is separate from `benchmarks/webvoyager/config.yaml`, which
continues to provide the public Selenium/BrowserGym-compatible baseline.

## Code path

```text
552-task pinned dataset
  -> WebAgent native profile
     -> Gym Responses boundary
        -> vLLM model proxy
           -> public Nano Omni v3 /v1/chat/completions endpoint (TP8)
     -> native_web resource server
        -> one Xvfb display
        -> headed Chromium managed by Playwright
        -> visible coordinate actions executed by PyAutoGUI
     -> native_v3 WebVoyager Gemini judge
```

The dataset, agent, browser and judge communicate through the common Gym web
models. The agent never imports Playwright or PyAutoGUI, and the browser server
does not own model prompting or scoring.

## Reproducibility contract

- Native recipe source: `osworld_internal` branch `nemotron-v3`, commit
  `3b775dc538931ead0cb6b4922349da9c6d493dab`.
- Dataset: `jayl940712/webarena_benchmarks` commit
  `6a2977939b157b0ab9de7799bb089c721f1ac115`, `webvoyager.jsonl`, 552 tasks,
  SHA-256 `f635a9b27fa1980a63b39bbf64ae8e9e766159cb70fa765451d3d3c0b948ff98`.
- Policy: public `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
  revision `24e67ea000b7c2837fc8f9488aa2008524fac8ba`.
- Generation: temperature 0.1, top-p 0.95, 100 browser steps, three recent
  browser screenshots, structured native browser tools.
- Serving: TP8, 128K context, an independently staged multimodal public-v3
  template, and
  `chat_template_kwargs={"truncate_history_thinking": false}` on every vLLM
  chat-completions request. Layer
  `benchmarks/webvoyager/configs/native_v3_policy.yaml` after the generic
  `vllm_model` config.
- Judge: `gcp/google/gemini-3-flash-preview`, all trajectory screenshots,
  JSON `SUCCESS`/`FAILURE` verdict.
- Browser timing: the start URL settles on `domcontentloaded`, every
  policy-driven navigation settles on `load`, `page.goto` retries transport
  faults after 4/4/4/8 s, and one 45 s context deadline bounds every Playwright
  operation.
- Policy transport: up to 20 attempts at 5 s for a single turn, so an unstable
  endpoint does not mask a task the reference would have completed.

The profile opts into bounded recovery for decoding and executing an action the
policy already chose: inner-JSON and one-missing-bracket repair, unambiguous
public-v3 tool aliases, parse-retry feedback at temperature 0.2, and a failed UI
action left visible for correction. It deliberately does not enable the repeated
-action hint, which would place harness-authored strategy in the policy's
context that the pinned reference never sends.

A browser that exhausts its CAPTCHA budget reports
`native_status=captcha_budget_exhausted`; the agent masks that rollout as
`failure_kind=captcha_budget_exhausted` rather than judging a forced stop, and
the summarizer routes it to the cleanup wave.

## Required run inputs

1. Set `WEBVOYAGER_SOURCE_JSONL` and run the native prepare script.
2. Provide reviewed paths for the tokenizer, chat template and reasoning
   parser plugin in the public v3 model config. Stage the public-v3 multimodal
   template as a separate immutable runtime asset and pass that explicit path
   to vLLM; do not rely on an implicit model-directory fallback. The tokenizer
   is not optional:
   serving directly from the public checkpoint currently emits the Mistral
   regex warning and is not equivalent to the launcher's explicit Nano
   tokenizer input.
3. Start every native resource-server instance under a unique 1920x1080 Xvfb
   display. `max_sessions=1` is intentional: PyAutoGUI is display-global.
   Install `xclip` in the browser image. PyAutoGUI cannot emit non-ASCII
   characters, so the runtime pastes them through the X clipboard instead; a
   missing `xclip` turns every non-ASCII `type` action into a step failure.
4. Provide `WA_BROWSER_PROXY_SERVER` for the selected US-proxy domains.
5. Provide `CAPSOLVER_API_KEY` and set `WA_CAPTCHA_PROVIDER=capsolver`. The
   built-in Turnstile/reCAPTCHA v2 integration is used unless an approved
   `WA_CAPTCHA_SOLVER=module.path:factory` override is supplied.
   If the browser reaches Squid through a node-local SSH tunnel, also set
   `WA_CAPTCHA_PROXY_SERVER` to the public endpoint for that same Squid
   instance. CapSolver runs outside the worker namespace and cannot use a
   browser proxy such as `127.0.0.1:19407`.
6. Configure the judge endpoint/key through the standard Gym secret/config
   channel.

The agent-facing route remains Gym's Responses API so the common rollout
contract is unchanged. The `vllm_model` proxy converts the request to OpenAI
Chat Completions and sends it to the policy endpoint. The captured outbound
payload—not the agent-facing URL—is the parity boundary: it must contain
`messages`, the five native WebVoyager tools and
`chat_template_kwargs.truncate_history_thinking=false`.

## Debug logging contract

Run with `verbose=true` and set `nemo_gym_log_dir` to a run-scoped directory.
Gym then retains one log per subprocess instead of relying only on the parent
terminal. The native route emits redacted `key=value` lifecycle events at the
following boundaries:

| Component | Representative events |
| --- | --- |
| Agent | `web_rollout_start`, `web_model_turn_complete`, `web_action_parsed`, `web_rollout_complete` |
| Browser/session | `web_session_seed_complete`, `native_browser_tool_complete`, `native_browser_screenshot`, `web_session_close` |
| CapSolver | `captcha_detected`, `captcha_task_created`, `captcha_solved`, `captcha_solver_failed` |
| Judge | `webvoyager_judge_model_complete`, `webvoyager_judge_unparseable`, `webvoyager_judge_complete` |

Logs include benchmark/task/session/step identifiers, safe URL origins,
operation names, status, latency, evidence sizes/hashes, token counts and error
classes. They intentionally exclude full URLs with query strings, browser
action arguments, screenshots, answers, judge text, API keys, proxy
credentials, provider task IDs and CAPTCHA solution tokens. Provider and
evidence identifiers appear only as short SHA-256 fingerprints.

For an experiment rooted at `${RUN_ROOT}` use overrides equivalent to:

```bash
++verbose=true \
++nemo_gym_log_dir=${RUN_ROOT}/logs/components
```

The CapSolver account gate is separate from the real browser-challenge gate:

```bash
python benchmarks/webvoyager/smoke_capsolver_account.py \
  --output "${RUN_ROOT}/status/capsolver-account-preflight.json"
```

It calls `getBalance` directly, never through the browser proxy, and persists
only balance/status metadata and a key fingerprint. A comparable run also
requires a real challenge log containing `captcha_detected` followed by
`captcha_solved` and evidence that the target page accepted the injected token.

For the reference topology, run two dataset splits concurrently. Each split
uses one TP8 policy replica and 16 one-session browser resource replicas. A
cleanup/resume pass dispatches only tasks without a terminal result.

Each browser replica must have an independent X display, HOME, temporary
directory, artifact directory and rollout output. Do not set
`gym eval --concurrency=16` against a single native resource server: native
coordinate actions are display-global and that server intentionally permits
only one session. Start 16 isolated, single-concurrency Gym processes per
split instead. The policy endpoints and read-only component environments may
be shared within a split after one serialized prefetch has completed.

Merge worker outputs by task ID and generate the cleanup input from the exact
pinned dataset, rather than from worker exit codes. The summarizer accepts
multiple files or run directories and writes the missing-task JSONL
atomically:

```bash
python benchmarks/webvoyager/summarize_native_v3.py \
  split-0/workers split-1/workers \
  --dataset webvoyager.jsonl \
  --output aggregate/summary.json \
  --missing-output cleanup/missing.jsonl
```

A result is comparable only when all expected task IDs are present exactly
once, no unexpected task IDs are present, and no task is masked as an
infrastructure/configuration failure.

When a cleanup wave reruns masked rows, pass its input with
`--superseded-ids-jsonl` and list the first-wave paths before cleanup paths.
The last result for only those explicitly declared task IDs then supersedes
the preserved first-wave result; undeclared duplicates remain an audit error.

## Validation order

Run one task first, then one task from every retained domain, then a 32-task
coverage subset, and only then all 552 tasks. Every report must distinguish
policy failure from browser, proxy/captcha, model-server and judge failures;
only a fully accounted denominator is comparable to a reference SR.
