# BrowserGym web resource server

This resource server implements Gym's common stateful web protocol for
WebArena and the legacy BrowserGym WebVoyager route.

- WebArena registers the pinned `browsergym/webarena.<task_id>` environment,
  exposes accessibility-tree observations and BrowserGym high-level actions,
  and returns the upstream terminal evaluator reward.
- WebVoyager uses BrowserGym's open-ended task, a SoM screenshot, and an
  external screenshot-and-answer judge.

The implementation keeps browser state inside one resource-server session;
the Responses agent owns the policy loop. BrowserGym action mappings are
validated before execution, and exceptions escaping `Env.step()` are reported
as evaluator/runtime failures rather than correctable policy actions.

WebArena model-backed evaluator tasks require an explicit evaluator model and
API key. The compatibility hook changes only the pinned evaluator's model
argument; its prompt, sampling options, and score parsing remain upstream.

The component pins BrowserGym 0.14.3. Its `overrides.txt` selects a Python
3.13-compatible greenlet build without changing the Playwright API version.
Install Chromium for Playwright 1.44.0 in the runtime image before launching
the server.

Browser video is optional. When enabled, finalized recordings are returned in
the standard close response alongside screenshot evidence.
