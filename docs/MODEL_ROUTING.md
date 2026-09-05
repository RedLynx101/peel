# Model routing

Use the smallest model that can complete the task without making review and repair cost larger than the savings. Route by task size, ambiguity, statefulness, and verification cost rather than by a presumed model ranking.

## Runtime descriptions

These are the authoritative model descriptions available in this session (September 5, 2026):

| Model | Runtime description | Good default |
| --- | --- | --- |
| `gpt-5.6-luna` | Fast and affordable agentic coding model | Mechanical documentation or configuration, small fixes, and bounded work with clear checks |
| `gpt-5.6-terra` | Balanced agentic coding model for everyday work | Medium-sized coding modules or tests with a clear contract and ordinary integration points |
| `gpt-5.6-sol` | Reliable agentic workhorse for everyday tasks | Complex or stateful backend work, learning logic, and ML integration where edge cases matter |

These descriptions and the routing below are operational judgments, not benchmark results. They do not establish prices, measured performance, or a tested ranking.

## Routing rules

- Prefer **Luna** for narrow, mechanical work: docs, config, straightforward refactors, and small bug fixes. Require a focused check such as a parser, lint, unit test, or diff review.
- Prefer **Terra** for a self-contained module, a normal test change, or a medium-sized implementation whose inputs, outputs, and acceptance checks are clear. Give it the contract and the relevant test command.
- Prefer **Sol** for stateful backend behavior, learning or scheduling logic, ML integration, cross-module debugging, and tasks where hidden edge cases make repair expensive. Keep the task bounded and specify the evidence required.
- Keep architecture, frontend implementation, graphics, integration decisions, and final evidence with the primary agent. The primary agent independently verifies backend work before treating it as complete.
- Escalate when the task becomes more ambiguous, crosses boundaries, or needs a more expensive verification loop. Do not escalate merely because a task is longer if its contract and checks remain mechanical.

The smaller model is the right choice only when its review cost does not exceed the time or resource savings. For a high-risk change, route the work to the stronger model or keep implementation and verification with the primary agent.

## Actual-use ledger

| Work item | Dispatch | Reason | Verification owner |
| --- | --- | --- | --- |
| Bounded learning backend | `gpt-5.6-sol`, high effort | Stateful backend behavior with higher edge-case and integration cost | Primary agent independently verifies backend behavior |
| This routing guide | `gpt-5.6-luna`, medium effort | Small documentation task with a direct acceptance check | Primary agent reviews the file and diff |
| No Terra dispatch yet | — | No medium-sized clear-contract module has been assigned in this run | — |

## Handoff checklist

Every delegated task should include a narrow scope, the relevant contract or files, the exact validation command or check, and a clear completion artifact. The primary agent owns architecture, frontend, integration, final evidence, and release decisions.
