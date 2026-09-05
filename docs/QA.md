# Verification inventory

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

Results are recorded in the final experiment report and Metis evidence archive. This file is an inventory, not a claim that all checks have already passed.
