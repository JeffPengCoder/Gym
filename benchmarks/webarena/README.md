# WebArena

This benchmark is a thin Gym adapter over BrowserGym WebArena. The prepare
script converts the official 812 task configs into normalized `web_task` rows;
BrowserGym still owns task setup, authentication, Playwright execution, and the
native WebArena evaluator.

By default the script finds the sibling checkout used during development:

```text
../webarena/config_files/test.raw.json
```

For another layout, set `WEBARENA_SOURCE_CONFIG`. Then run:

```bash
gym eval prepare --benchmark webarena
```

Before rollout, deploy the WebArena websites and set all `WA_*` URLs documented
by `resources_servers/browsergym_web/README.md`. Of the official tasks, 82 use
semantic fuzzy matching and another 36 can invoke the unachievable-answer
judge. Pass `webarena_evaluator_model=<model>` and
`web_evaluator_base_url=<openai-compatible-url>`, and place the credential in
`OPENAI_API_KEY` (or select another name with `web_evaluator_api_key_env`). A
missing model on one of those 118 tasks is a masked configuration failure, not
a score of zero.

The BrowserGym implementation is single-session because a fresh browser
context does not isolate mutable site state.

## Native Nano Omni route

`configs/native_v3.yaml` selects the headed-Chromium/PyAutoGUI action surface
and the pinned native WebArena evaluator without changing the BrowserGym
profile above. Prepare its maintained 812-row population with:

```bash
export WEBARENA_NATIVE_SOURCE_JSONL=/path/to/webarena_benchmarks/webarena.jsonl
python benchmarks/webarena/prepare_native_v3.py
```

The native prepare path accepts the 812-row `webarena.jsonl` from
`jayl940712/webarena_benchmarks` commit
`6a2977939b157b0ab9de7799bb089c721f1ac115` and verifies its SHA-256 before
adapting any row. A different 812-row file is rejected rather than silently
creating a different benchmark recipe.

Before launch, provide the required `WA_*` site URLs, reset the self-hosted
deployment, and inject the evaluator's `WEBARENA_JUDGE_*` environment through
the resource-server secret boundary. One native process owns one X display;
parallel workers require separate displays and a site-state isolation policy.
