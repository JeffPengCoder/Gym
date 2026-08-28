# Native WebArena resource server

This component runs the native Nano Omni WebArena recipe behind Gym's common
session, step, evaluate, and artifact protocol. It is a sibling of
`browsergym_web`, not a subclass: both expose the same Gym boundary while
preserving different browser-action and evaluator semantics. Native
WebVoyager remains in the dedicated `webvoyager_browser` component.

Playwright owns Chromium contexts, pages, navigation, and tabs. PyAutoGUI owns
visible coordinate input and full-display screenshots. The Responses agent
owns the policy loop. The pinned WebArena evaluator scores string, URL, and
`program_html` targets against the live self-hosted sites.

Collision plans merge site/API snapshots with live-page snapshots before and
after each rollout. This prevents one task from silently receiving credit for
another task's mutation. The evaluator source is pinned to the native
reference at `3b775dc538931ead0cb6b4922349da9c6d493dab`; see
`reference_evaluation/PROVENANCE.md`.

One process supports exactly one live session on one X display. Horizontal
parallelism uses isolated server replicas with distinct `DISPLAY` values.
Distributed Gym workers remain supported, but mutable WebArena sites also
need isolated deployments or scheduling/reset policy that prevents conflicting
writers.

The container entrypoint must start Xvfb before the Python server. The image
must also provide `xclip` for Unicode clipboard input and benchmark fonts;
these are OS dependencies rather than hidden Python dependencies.

Tasks resolve `WA_SHOPPING`, `WA_REDDIT`, `WA_GITLAB`, `WA_WIKIPEDIA`,
`WA_MAP`, `WA_CLASSIFIEDS`, and related deployment URLs plus the public
benchmark login accounts. Model-backed evaluator tasks additionally require
the `WEBARENA_JUDGE_*` environment contract. The aligned native profile pins
the judge model and base URL explicitly instead of relying on evaluator
fallbacks.

Use `resources_servers/native_web/configs/native_webarena.yaml` for the
benchmark-specific profile. The server logs redacted lifecycle events through
`nemo_gym.resources_servers.native_web`; screenshots, credentials, and full
URL paths are not logged.
