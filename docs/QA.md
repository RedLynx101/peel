# Verification record

Verified September 5, 2026 by the primary agent. **25 Python tests passed**, Ruff lint/format and JavaScript syntax/Prettier checks passed. The bundled 325 KB checkpoint was hash-checked and used for actual CPU inference through the API. The editable package installation also completed successfully.

The Chrome browser run passed **29 checks** in `scripts/browser-qa.js`: playback/pause/restart/scrub/4x, exact-map comparison and unequal episode lengths, checkpoint labels, validation-only chart data, custom-map inference, object/boundary editing, keyboard placement, and recoverable missing-model/server errors. Layout checks passed at 1440, 390, and 360 pixels with no horizontal overflow. Reduced motion disabled autoplay. There were no uncaught JavaScript errors; two network errors were deliberately injected for the recovery checks.

The primary manually inspected [desktop](../artifacts/screenshots/museum-desktop.png), [mobile](../artifacts/screenshots/mobile.png), [comparison](../artifacts/screenshots/comparison.png), and [camera](../artifacts/screenshots/camera-gallery.png) screenshots, plus all three PNG charts. An initially scrolled viewport capture was corrected before delivery. Font assets are local and no external font requests are required.

The repository's pinned [Checks workflow](../.github/workflows/checks.yml) verifies a fresh Linux checkout with CPU PyTorch, including the bundled checkpoint. Its current run status is visible in [GitHub Actions](https://github.com/RedLynx101/peel/actions). Final remote revision, visibility, and CI acceptance are recorded in the delivery audit after pushing.

To repeat the optional browser checks with the local server running, open `http://127.0.0.1:8000` using Playwright CLI, then run `playwright-cli run-code --filename scripts/browser-qa.js`. The recorded run used Playwright CLI with Chrome. The assertions intentionally depend on this release's curated artifacts; update them when publishing a different experiment. They simulate error responses only inside that browser session.

The primary agent owns integrated verification. A passing backend test alone does not establish browser correctness, model competence, or deployment.

## Browser acceptance

- Desktop 1440x1000: readable hierarchy, visible game/controls, no horizontal overflow or clipping.
- Mobile 390x844 and narrow 360x800: stacked layout, usable controls, legible charts/editor, no horizontal overflow.
- Initial load: genuine exported replay and metadata, no invented defaults for missing metrics.
- Play/pause, restart, scrub, and 1x/2x/4x playback: correct frame progression and end-of-episode state.
- Checkpoint selector: loads distinct model artifacts, identifies scripted/imitation/PPO attempts accurately.
- Gallery/Joyce view: hidden tiles do not reveal geometry; inventory/outcome agree with replay.
- Comparison: matched seed when available; different maps explicitly labeled; close returns to main view.
- Experiment selector: plots evaluation-only curves with matching metadata; training rewards are not held-out scores.
- Editor: tool selection, tile placement, boundary preservation, unique-object relocation, reset.
- Room validation: invalid missing objects rejected; geometric-only scope disclosed.
- Custom heist: genuine policy inference; visibly returns success or failure, no scripted fallback disguised as model.
- Error path: missing model/server produces clear recoverable message and re-enables action button.
- Keyboard: visible focus, accessible controls, editor cells operable with Enter, skip link.
- Reduced motion: no automatic replay or decorative motion; controls remain functional.
- Console/network: no uncaught errors or missing required local assets.

## Scientific acceptance

- Train/validation/test map contents differ, not only RNG seeds.
- Saved observation schema matches loaded checkpoints.
- Teacher uses observable history only.
- PPO terminal masks and episode history are correct.
- No repeatable reward farming through interactions.
- Checkpoint round-trip and documented fresh-episode resume verified.
- Evaluation count, split, seed, and model revision recorded with every headline result.
- Controlled memory diagnostic distinguished from heist task performance.
- Report unsuccessful trials and constraints alongside improvements.

## Release acceptance

- Clean checkout tests, syntax/format checks, repeatable setup, MIT and bundled font licenses.
- Screenshots and charts manually reviewed; raw metrics retained.
- GitHub repository private, expected owner/name, pushed commit matches local HEAD, CI checked.
- No public deployment or visibility change.

Scientific measurements and their limits are recorded in [EXPERIMENTS.md](EXPERIMENTS.md). The criteria above define ongoing acceptance; a camera policy that fails its test suite is reported as a failed experiment, not hidden by a passing software test.
