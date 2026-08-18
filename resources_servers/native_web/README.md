# Native Nano Omni web baseline

This resource-server source snapshot preserves the independently runnable
native WebArena, VisualWebArena, and WebVoyager baseline. It is intentionally
separate from `resources_servers/browsergym_web`.

The baseline retains its original architecture:

- its own Nano Omni model loop and structured tool calls;
- headed Chromium on Xvfb;
- PyAutoGUI screenshots and computer actions;
- Playwright context, navigation, tabs, login, proxy, and CAPTCHA helpers;
- native benchmark evaluators and judges;
- original split/worker launchers and result format.

## Source snapshots

Two source roots are retained because VisualWebArena was maintained on a
separate GitLab branch:

```text
baselines/
  nemotron_v3/       WebArena and WebVoyager
  visualwebarena/    VisualWebArena
```

The exact repository, branch, and commit identities are recorded in
`SOURCE_LOCK.json`. Files below `baselines/` are copied without semantic
changes from those commits; six redundant final blank lines were normalized
to satisfy Gym's diff checks. Do not merge the two copies of `webarena/common`
or `webarena/nvidia`; their evaluator and runtime behavior differs.

## Running

Deployment configuration, credentials, cluster paths, large runtime assets,
and experimental results do not belong in Gym. The corresponding runbook is
maintained in `rl.git` under:

```text
webarena/runs/2026-08-19-gitlab-native-web-runner/
```

Use the launcher in the appropriate pinned source root:

- `nemotron_v3/launch_nemotron_webarena_parallel.sh`
- `nemotron_v3/launch_nemotron_webvoyager_parallel.sh`
- `visualwebarena/launch_nemotron_visualwebarena_parallel.sh`

Start with one task and a fresh result directory. WebArena and
VisualWebArena require a controlled stateful-site reset policy. WebVoyager
requires the configured judge, US egress proxy, and CapSolver contract.

## Relationship to the alignment branch

This branch is a behavioral baseline. It does not use the Gym Responses API
agent loop and is not intended as the final training architecture.
`feature/webarena-alignment` ports only the required behavior into Gym's
agent/driver/evaluator/resource-server boundaries and compares against this
baseline.
