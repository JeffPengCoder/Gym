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
gym eval prepare --config benchmarks/visualwebarena/configs/native_v3.yaml
```

On first use, the native prepare path downloads the complete public
[`jayl940712/webarena_benchmarks`](https://github.com/jayl940712/webarena_benchmarks)
archive at commit `6a2977939b157b0ab9de7799bb089c721f1ac115` into
`cache/webarena_benchmarks/<commit>/`. This includes both the 908-row
`visualwebarena.jsonl` and its 346 reference images. The download is atomic and
reused on later runs. Before adapting any row, prepare verifies the JSONL
SHA-256, task count, and every local image reference; a different or incomplete
source is rejected rather than silently changing task IDs, images, or evaluator
targets.

For a shared or pre-populated cache, point `VISUALWEBARENA_NATIVE_SOURCE_ROOT`
at the directory containing both `visualwebarena.jsonl` and the
`visualwebarena/` image tree. The native config passes that same absolute root
to the agent and resource server:

```bash
export VISUALWEBARENA_NATIVE_SOURCE_ROOT=/shared/webarena_benchmarks
gym eval prepare --config benchmarks/visualwebarena/configs/native_v3.yaml
```

An offline checkout can be created explicitly when the cluster login node is
the only host with network access:

```bash
git clone https://github.com/jayl940712/webarena_benchmarks.git /shared/webarena_benchmarks
git -C /shared/webarena_benchmarks checkout 6a2977939b157b0ab9de7799bb089c721f1ac115
export VISUALWEBARENA_NATIVE_SOURCE_ROOT=/shared/webarena_benchmarks
```

The JSONL intentionally retains paths such as
`visualwebarena/shopping/task_86/input_0.png`. Consequently, the image root is
the public checkout root—not a `benchmarks/` child. Mount that root read-only at
the same absolute path on every distributed worker. The resource server also
needs the deployment's `WA_*` URLs and approved `WEBARENA_JUDGE_*` secret
environment. Site reset and split-level deployment isolation remain
orchestration responsibilities, not browser-driver behavior.
