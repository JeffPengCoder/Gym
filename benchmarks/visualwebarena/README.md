# VisualWebArena

This benchmark uses BrowserGym's VisualWebArena task and evaluator with a
shared Gym rollout loop. Observations contain the screenshot with BrowserGym's
set-of-marks overlay plus a bracketed accessibility tree.

The upstream repository stores site-local task IDs in three files, while
`libvisualwebarena==0.0.15` and BrowserGym use one global range `0..909`. The
prepare script reproduces the package order exactly: Classifieds, Reddit
(including its cross-site tasks), then Shopping, assigning global IDs after
concatenation. For example, upstream Shopping task 0 is BrowserGym task 444.

The default source is the sibling `../visualwebarena/config_files/vwa`
checkout. Set `VISUALWEBARENA_SOURCE_DIR` for another layout, then run:

```bash
gym eval prepare --benchmark visualwebarena
```

Deploy the VWA site stack and configure the required `VWA_*` variables before
collecting rollouts. Tasks with fuzzy or unachievable-answer matching also
need `visualwebarena_evaluator_model=<model>`,
`web_evaluator_base_url=<openai-compatible-url>`, and a credential in
`OPENAI_API_KEY` (or the environment name selected by
`web_evaluator_api_key_env`). The configured judge model is recorded in the
verifier version; missing configuration is masked rather than scored as a task
failure.

## Native Nano Omni route

`configs/native_v3.yaml` uses the shared native visual driver plus the pinned
VisualWebArena evaluator. Prepare the maintained 908-row population with:

```bash
export VISUALWEBARENA_NATIVE_SOURCE_JSONL=/path/to/webarena_benchmarks/visualwebarena.jsonl
python benchmarks/visualwebarena/prepare_native_v3.py
```

The native prepare path accepts the 908-row `visualwebarena.jsonl` from
`jayl940712/webarena_benchmarks` commit
`6a2977939b157b0ab9de7799bb089c721f1ac115` and verifies its SHA-256 before
adapting any row. A different 908-row file is rejected rather than silently
changing task IDs, reference images, or evaluator targets.

The JSONL intentionally retains paths such as
`visualwebarena/shopping/task_86/input_0.png`. Mount the directory that contains
the `visualwebarena/` tree read-only (the reference checkout's `benchmarks/`
directory) and set that parent as the same `task_image_root` override on both
`native_visualwebarena_agent.responses_api_agents.web_agent` and
`native_visualwebarena.resources_servers.native_web`. The resource server also
needs the deployment's `WA_*` URLs and approved `WEBARENA_JUDGE_*` secret
environment. Site reset and split-level deployment isolation remain
orchestration responsibilities, not browser-driver behavior.
