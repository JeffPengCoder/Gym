# Native web resource server

This server implements the common Gym web protocol with a headed Chromium
driver. Playwright owns contexts, pages, navigation and tabs; PyAutoGUI owns
visible coordinate input and full-display screenshots.

One process supports exactly one live session on one X display. Horizontal
parallelism is obtained by launching isolated server replicas with distinct
`DISPLAY` values. This prevents clicks from one rollout entering another
rollout's browser.

The server expects Xvfb to be started by the container/job entrypoint before
the Python server starts. It intentionally fails before creating a browser if
`DISPLAY` is absent.

With `CAPSOLVER_API_KEY` and `WA_CAPTCHA_PROVIDER=capsolver`, the built-in
solver handles visible Turnstile and reCAPTCHA v2 widgets. A reviewed custom
solver can instead be injected through
`WA_CAPTCHA_SOLVER=module.path:factory`; the factory returns an object exposing
`maybe_solve(page, phase=...)`.

The server logs redacted lifecycle events through the
`nemo_gym.resources_servers.native_web` and
`nemo_gym.resources_servers.native_web.captcha` loggers. Enable Gym verbose
logging for challenge scans. INFO records detections, provider task creation,
successful injection, browser actions, screenshots and failures; DEBUG also
records scans where no challenge was present. Keys, proxy credentials,
provider task IDs, solution tokens, screenshot payloads and URL query strings
are never logged.
