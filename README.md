# Conformal Yukawa MCT

Mode-coupling theory for a **conformally-coupled Yukawa one-component plasma** — a scalar field φ
coupled to plasma density through the interaction strength itself, solved for the intermediate
scattering function F(k,t).

Julia. One day of work, 2026-04-27.

---

## Why this exists

Thirteen days earlier, an [origin note](https://github.com/goodcarp/tricritical-exploration/blob/main/notes/2026-04-14-phi4-free-energy-origin.md)
recorded a free energy that kept reappearing across unrelated problems:

$$f(n, \phi) = n k_B T\left[\ln(n\Lambda^3) - 1\right] - \eta\, n^{4/3}\left(1 + \xi\phi^2\right) + \tfrac{1}{2}m^2\phi^2 + \tfrac{\lambda}{4}\phi^4$$

This repository is that equation made to run. The coupling is promoted from $(1 + \xi\phi^2)$ to a
squared conformal factor:

$$f(n,\phi) = f_0(n) - \eta\, n^{4/3} A^2(\phi) + \tfrac{1}{2}m^2\phi^2 + \tfrac{\lambda}{4}\phi^4,
\qquad A(\phi) = 1 + \tfrac{1}{2}\xi\phi^2$$

which reorganizes into a density-dependent Landau form:

$$f = f_0(n) + \tfrac{1}{2} r(n)\,\phi^2 + \tfrac{1}{4} u(n)\,\phi^4,
\qquad r(n) = m^2 - 2\eta\xi n^{4/3},
\qquad u(n) = \lambda - \eta\xi^2 n^{4/3}$$

Two consequences fall straight out, and they are the reason the tricritical program exists:

| Quantity | Result |
|---|---|
| **Critical density** | $n_c = \left(m^2 / 2\eta\xi\right)^{3/4}$ |
| **Character of the transition at $n_c$** | second-order if $\lambda > \tfrac{1}{2}\xi m^2$, **first-order otherwise** |

The quartic coefficient $u(n)$ changes sign independently of $r(n)$. Where both vanish together is a
**tricritical point** — which is what the rest of the research program went on to chase.

## Layout

```
src/ConformalYukawaMCT.jl     custom memory kernel plugging into ModeCouplingTheory.jl
src/PlasmaMCTKernel.jl        standalone regularized Yukawa-OCP kernel (no solver dependency)
structure-factor/             HNC / WLP-regularized S(k) inputs at Γ=50, κ=1
runs/                         the sweeps: task2 baseline, run A (ξ), run B (density)
examples/ test/ data/         worked example, unit tests, HNC input-table spec
plots/                        six result figures
```

### Two kernels, deliberately

- **`ConformalYukawaMCT.jl`** is the research kernel — the φ⁴-coupled version, built against
  [ModeCouplingTheory.jl](https://github.com/IlianPihlajamaa/ModeCouplingTheory.jl)
  (Pihlajamaa et al., JOSS 2023, [arXiv:2305.01365](https://arxiv.org/abs/2305.01365)).
- **`PlasmaMCTKernel.jl`** is the clean-room version: the regularized 3D Yukawa-OCP memory kernel
  with the full isotropic angular reduction, no field coupling, no solver dependency, unit-tested.
  It exists to check the research kernel's plasma sector in isolation.

The memory kernel, with the angular integral reduced:

$$K(k,t) = \frac{n}{8\pi^2}\int\! dq\, q^2 \int_{-1}^{1}\!\! d\mu\;
\Big[q\mu\, c_s(q) + (k - q\mu)\, c_s(p)\Big]^2 F(q,t) F(p,t),
\qquad p = \sqrt{k^2 + q^2 - 2kq\mu}$$

evaluated on 64 Gauss-Legendre angular nodes. The vertex uses **only** the short-range
Wertheim–Lebowitz–Percus regularized direct correlation function — the HNC direct correlation with
the bare long-range Yukawa Fourier component removed. Using the unregularized $c(q)$ here is the
standard way to get a divergent vertex in a charged system, so it is enforced rather than assumed.

## The runs

| Run | Sweep | Figures |
|---|---|---|
| **task2** | ξ = 0 baseline — no conformal coupling, pure Yukawa OCP | `task2_F_raw.png`, `task2_F_normalized.png` |
| **run A** | ξ ∈ {0, 0.05, 0.10, 0.15} at fixed n, m²=10, λ=2 | `runA_F_kpeak_xi.png`, `runA_F_kmin_xi.png` |
| **run B** | density sweep across $n_c$ | `runB_F_kpeak_n.png`, `runB_F_kmin_n.png` |

All at Γ = 50, κ = 1. F(k,t) is sampled at the structure-factor peak and at the smallest k on the
grid — the two places where a coupling-driven change in relaxation should show up first.

## Running it

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. runs/run_task2.jl
julia --project=. runs/run_runA.jl
julia --project=. runs/run_runB.jl
```

Tests for the standalone kernel:

```bash
julia --project=. test/runtests.jl
```

`Project.toml` / `Manifest.toml` pin the environment as it was on 2026-04-27. Run A and Run B need
`ModeCouplingTheory` and `Plots`; `PlasmaMCTKernel.jl` needs only `LinearAlgebra`.

To use real HNC data rather than the built-in approximation, drop a `k,S,c_short` table into `data/`
per [`data/README.md`](data/README.md) — `c_short` must be WLP-regularized, not the full HNC $c(k)$.

## Status

Exploratory. The sweeps ran and produced figures; nothing here has been validated against an
independent MCT implementation or against simulation, and the HNC inputs are approximated rather than
tabulated. Treat the transition characterization as analytic (it follows from the Landau form) and
the dynamics as suggestive.

---

*Private repository. Provenance for the tricritical research program.*
