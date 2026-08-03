# BrowserGym Web Resources Server

This stateful server owns the live Playwright context used by WebArena,
VisualWebArena, and the BrowserGym-backed WebVoyager profile. It deliberately
keeps benchmark-specific launch and evaluation logic behind one HTTP contract:

```text
seed_session -> observe -> step* -> evaluate -> close
```

WebArena and VisualWebArena use their BrowserGym task validators in the live
session. WebVoyager uses `browsergym/openended`; its final screenshots are
judged by the separate `webvoyager_judge` resource server.

## Runtime setup

Install the server environment and Chromium:

```bash
uv sync --project resources_servers/browsergym_web
uv run --project resources_servers/browsergym_web playwright install chromium
```

BrowserGym expects the official site-stack URLs in environment variables. For
WebArena these are `WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`,
`WA_GITLAB`, `WA_WIKIPEDIA`, `WA_MAP`, and `WA_HOMEPAGE`. VisualWebArena uses
`VWA_SHOPPING`, `VWA_REDDIT`, `VWA_WIKIPEDIA`, `VWA_CLASSIFIEDS`,
`VWA_CLASSIFIEDS_RESET_TOKEN`, and `VWA_HOMEPAGE`.

## Isolation boundary

The first version exposes `site_pool_mode: unmanaged` and defaults to one live
session. A fresh browser context does not reset mutable websites. This is an
explicit deployment limitation, not an implied isolation guarantee. A future
site-pool implementation can replace the lease boundary without changing the
agent or benchmark row contract.

Step `execution_ok` reports whether the browser action executed; evaluator
score is returned separately as `benchmark_reward`. Browser/evaluator
infrastructure failures are surfaced for masking rather than converted into a
benchmark score of zero.
