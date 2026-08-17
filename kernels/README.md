# Kernels

Empty on purpose.

Custom NKI kernels are written only after profiling the real training
pipeline on real Trainium2 hardware identifies a genuine, measured
bottleneck — never speculatively. See `docs/research_hypotheses.md` (H5)
for the required measurement and go/no-go criterion, and
`docs/trainium_notes.md` for fusion candidates worth checking first
(RMSNorm+residual, activation+projection) once profiler data exists.

`nki/` will hold one file per kernel once any are written, each paired with
an isolated micro-benchmark proving it's faster than the baseline op it
replaces.

`TODO: integrate official AWS Trainium Frontier starter pipeline` — no
kernel work starts until we have the real ops to profile.
