# Contributing to Peel

Keep the simulator authoritative. Frontend animation may interpolate positions but must not invent actions, outcomes, or observations. Never replace a failed model attempt with a scripted success without labeling it.

For environment changes, update the observation version when the encoded meaning changes, and add a rule or privacy test. For learning changes, save a new config and run directory; preserve the old raw records. Report sample counts, seed namespaces, demonstration cost, and failures. Use validation for selection and reserve final test results for reporting.

Run pytest, Ruff, JavaScript syntax checks, and Prettier before a pull request. Review the museum at desktop and mobile sizes after visual changes. Avoid adding frontend dependencies unless they solve a concrete problem better than the current small modules.

Training files in `runs/` are local and ignored. Curate explicit artifacts for review; do not commit virtual environments, caches, credentials, or large unselected checkpoints. Only load checkpoints from trusted sources: PyTorch training checkpoints contain Python-serialized state.

Changes to public hosting or repository visibility require the owner's explicit decision. MIT covers the original code and artwork; preserve the OFL font notices.
