# Sources and third-party material

Peel's simulator, training implementation, UI, and Canvas/SVG illustrations are original project code released under MIT. The agent Joyce and Bloom Gallery are fictional; the creative references were Joyce and bananas.

## Bundled assets

- DM Sans variable Latin font, distributed through `@fontsource-variable/dm-sans` 5.3.0. SIL Open Font License, bundled in `web/fonts/DM-Sans-LICENSE.txt`.
- Fraunces variable Latin normal/italic fonts, distributed through `@fontsource-variable/fraunces` 5.3.0. SIL Open Font License, bundled in `web/fonts/Fraunces-LICENSE.txt`.
- All museum illustrations and the banana favicon are drawn in source. No stock art or generated-image service is required.

## Implementation references

- [Gymnasium](https://gymnasium.farama.org/): environment interface and terminal semantics.
- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py): reference for the PPO algorithm and readable experiment implementation.
- [CleanRL PPO-TrXL](https://docs.cleanrl.dev/rl-algorithms/ppo-trxl/): memory-policy implementation context.
- [MiniGrid](https://minigrid.farama.org/): initial environment inspiration. Peel ultimately implements its own bounded Gymnasium environment for precise observation and replay semantics.
- [PyTorch](https://pytorch.org/): neural networks, automatic differentiation, and CUDA execution.

Installed Python and Node dependencies retain their own licenses. See the dependency manifests and lock/snapshot files for versions. No affiliation with reference projects is implied.
