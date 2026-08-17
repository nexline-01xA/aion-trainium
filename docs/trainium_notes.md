# Trainium2 / NKI Notes

Working notes on the hardware and tooling. This is background reference,
not a substitute for the official Neuron SDK docs or the competition's
provided NKI documentation — check those first for anything version-specific.

## NeuronCore execution model

- Trainium NeuronCores expose distinct compute engines (Tensor, Vector,
  Scalar, plus GPSIMD for custom/general-purpose work). Efficient kernels
  keep the TensorEngine fed with large, regular matmul tiles rather than
  many small or irregular ones.
- On-chip SBUF is software-managed scratchpad memory — data movement
  between HBM and SBUF is explicit (via DMA), not automatic like a GPU
  cache hierarchy. This is the main mental-model shift coming from CUDA:
  every kernel author is responsible for orchestrating data movement.
- NKI (Neuron Kernel Interface) is a Python, NumPy/Triton-like, tile-level
  programming environment for the Neuron compiler, and it interoperates
  with the Neuron Profiler for identifying bottlenecks and instruction
  latencies.

## Practical implications for AION

1. **Compilation cost is real and can eat into a 30-minute budget.**
   Confirm graph caching is working and that tensor shapes don't change
   step-to-step (which would force recompilation).
2. **SBUF capacity constrains batch/tile size choices.** A batch size that
   spills SBUF will trigger extra DMA traffic — worth profiling explicitly
   rather than assuming "bigger batch = better."
3. **Conditional computation must stay tile-regular.** Per-token gather/
   scatter routing (GPU MoE-style) is likely to hurt more than it helps on
   this architecture — see `docs/research_hypotheses.md` H4 and
   `src/aion/routing/block_router.py`.
4. **Kernel fusion candidates should come from the profiler, not
   intuition.** Norm+residual and activation+projection are typical
   candidates for small models, but this repo does not assume any specific
   op is hot until we've actually profiled the real pipeline.

## Environment / setup (to fill in once we provision a Trainium2 instance)

- Neuron SDK version: `TODO`
- Instance type: `TODO`
- `torch-neuronx` / `torch-xla` version: `TODO`
- NKI version / `nki-samples` commit used as reference: `TODO`

## References

- AWS Neuron Kernel Interface samples: https://github.com/aws-neuron/nki-samples
- AWS Trainium Frontier Competition overview (public listing).
- Official competition NKI documentation and profiling tools (provided to
  registered teams — not yet in our possession).
