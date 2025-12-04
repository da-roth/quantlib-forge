# QuantLib-Forge

Forge integration layer for QuantLib – enabling **fast JIT‑compiled kernels** that you can use either for **pure forward re‑evaluation** of pricing engines with different inputs, or for **forward re‑evaluation plus AAD**.

## What this project does

This repo provides:

1. **Integration Layer** – headers and adapters so existing QuantLib pricing engines can run as Forge JIT‑compiled kernels (forward‑only or forward+AAD) without rewriting models.
2. **ABool Patches** – small QuantLib patches to support conditional recording via Forge’s `ABool` type (coupon pricers, normal CDF, barrier engine).
3. **Benchmark Harness** – XVA swap and barrier benchmarks comparing Forge Forward‑Only, Forge AAD, finite differences, and XAD.

The integration supports:

- **Forward‑only kernels**: record once with `markForgeInput()`, JIT‑compile, then **re‑evaluate very quickly** for many different input scenarios (no gradient buffers allocated).
- **Forward + AAD kernels**: record with `markForgeInputAndDiff()`, JIT‑compile, then compute gradients in a single backward pass.
- **Instruction sets**: scalar **SSE2** by default, with optional **AVX2** packed variants for higher throughput.

This is most useful when **the same pricing logic is evaluated many times with different inputs** – e.g. XVA engines re‑pricing portfolios across scenarios, Monte Carlo simulations running many paths through one pricer, or regulatory stress tests. You pay a one‑time kernel compilation cost, then re‑use that kernel cheaply across scenarios.

## Example / benchmark focus

The main performance example is an **XVA‑style vanilla swap benchmark** with up to 100 risk factors and multiple time steps. It compares:

- **Bump‑Reval (QuantLib)** – direct QuantLib evaluation with finite‑difference sensitivities.
- **Forge Forward‑Only** – JIT‑compiled kernel using `markForgeInput()`; sensitivities still via bump‑reval, but much faster evaluations.
- **Forge AAD** – JIT‑compiled kernel using `markForgeInputAndDiff()` with a backward pass for gradients.
- **XAD / QuantLib‑Risks‑Cpp** – an external AAD engine used as a baseline.

We also provide **AVX2 variants** of the Forge kernels that process scenarios 4‑wide via SIMD, so you can compare:

- Scalar SSE2 vs packed AVX2.
- Stability‑only vs fully optimized compiler configurations.

## Performance

XVA benchmark: vanilla swap pricing with 100 risk factors, 3 time steps. The trade‑off: Forge has a one‑time kernel creation cost, but much faster evaluation once the kernel exists:

```
XAD vs Forge-AAD
═════════════════════════════════════════════════════════════════════════════════════
  Scenarios       XAD               Forge-AAD                      Forge vs XAD
                  (0.12 ms/eval)    (1.54 ms kernel creation)      
───────────────────────────────────────────────────────────────────────────────
          1        0.12 ms           1.58 ms                       0.08×
         10        1.30 ms           1.60 ms                       0.81×
        100       13.00 ms           1.90 ms                       6.8×
      1,000      129.94 ms           4.82 ms                       26.9×
═════════════════════════════════════════════════════════════════════════════════════
```

**Interpretation:**
- **Single-scenario regime** – XAD is competitive or faster for tiny workloads because Forge pays a kernel creation cost.
- **Many-scenarios regime** – once you reach 100–1,000 scenarios, Forge‑AAD amortizes kernel creation and becomes much faster per scenario than both bump‑reval and XAD.

The tables below use timings from a local run on an **Intel Core i9‑13900K**; absolute numbers will vary by CPU, but the relative behaviour (kernel creation vs per‑scenario cost) is representative. The full raw benchmark outputs for all methods and configurations are available under `docs/benchmarkResults/`.

Full benchmark results ([baseline](docs/benchmarkResults/benchmark-baseline.txt) | [forge](docs/benchmarkResults/benchmark-forge.txt) | [xad](docs/benchmarkResults/benchmark-xad.txt)):

```
Total Execution Time
═════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Test Case          Scenarios      QuantLib     Forge-Bump        Forge-AAD            XAD
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1 (10 RF, 1 step)          1        0.28 ms        0.30 ms          0.38 ms        0.04 ms
                                        (1×)         (0.9×)           (0.7×)            (7×)

  2 (100 RF, 1 step)         1        2.52 ms        0.48 ms          0.56 ms        0.04 ms
                                        (1×)         (5.3×)           (4.5×)           (63×)

  3 (100 RF, 3 steps)        1        5.86 ms        1.26 ms          1.58 ms        0.12 ms
                                        (1×)         (4.7×)           (3.7×)           (49×)

  3 (100 RF, 3 steps)       10       57.60 ms        2.02 ms          1.60 ms        1.30 ms
                                        (1×)          (28×)            (36×)           (44×)

  3 (100 RF, 3 steps)      100      575.20 ms        9.58 ms          1.90 ms       13.00 ms
                                        (1×)          (60×)           (303×)           (44×)

  3 (100 RF, 3 steps)    1,000   5,894.60 ms       80.14 ms          4.82 ms      129.94 ms
                                        (1×)          (74×)         (1,223×)           (45×)
═════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

In the Forge results, different **compiler configurations** are used for the bump‑reval, forward‑only and AAD variants (stability‑only vs all‑optimizations, scalar SSE2 vs AVX2). For small numbers of scenarios it can be best to skip expensive graph optimizations and keep kernel creation cheap, while for large scenario counts it can be worth paying more for graph optimization up‑front to get faster per‑scenario execution. The GitHub Actions workflows (`benchmark-baseline.yml`, `benchmark-xad.yml`, `benchmark-forge.yml`, `benchmark-forge-all.yml`) run these benchmarks for plain QuantLib, QuantLib+XAD, and QuantLib‑Forge across the different scalar/AVX and optimization settings.

Per-Scenario Timing Breakdown (ms)

| Test Case               | Scenarios | QuantLib (eval) | Fwd-Kernel (create)  | Bump (eval) | AAD-Kernel (create)  | AAD (eval) | XAD (eval) |
|-------------------------|-----------|-----------------|----------------------|-------------|----------------------|------------|------------|
| 1 (10 RF, 1 step)       | 1         | 0.27            | 0.30                 | 0.01        | 0.38                 | 0.001      | 0.04       |
| 2 (100 RF, 1 step)      | 1         | 2.52            | 0.45                 | 0.03        | 0.56                 | 0.001      | 0.04       |
| 3 (100 RF, 3 steps)     | 1         | 1.96            | 1.17                 | 0.03        | 1.55                 | 0.001      | 0.03       |
| 3 (100 RF, 3 steps)     | 10        | 1.92            | 1.21                 | 0.03        | 1.54                 | 0.001      | 0.04       |
| 3 (100 RF, 3 steps)     | 100       | 1.92            | 1.36                 | 0.03        | 1.59                 | 0.001      | 0.04       |
| 3 (100 RF, 3 steps)     | 1,000     | 1.96            | 1.58                 | 0.03        | 1.70                 | 0.001      | 0.04       |

## QuantLib `Real`, branching, and `ABool`

The XVA swap benchmark above is deliberately **branch‑simple**: once you fix the trade, the pricing path is mostly linear, so **we don’t need any special branch tracking** to compare Forge vs. bump‑reval vs. XAD.

In general QuantLib code, numerical work is written in terms of a `Real` alias (typically `double`) and templated numeric algorithms. That makes it easy to change the numeric type, but **branching is still driven by plain C++ `bool`**, not by a “templated bool” or an AAD‑aware condition type.

Forge introduces **`ABool`**, a boolean‑like type that lives on the AAD tape and can be used for **branch tracking** (`ABool::If`, etc.). To take advantage of this in QuantLib, we add small patches that:

- Introduce Forge’s `ABool` in selected QuantLib components (e.g., coupon pricers, normal CDF, barrier engine).
- Replace some `if/else` branches with `ABool`‑based constructs so that Forge can both **reuse kernels** and still respect changing branch conditions across scenarios.

This becomes critical for **barrier‑style payoffs** and other path‑dependent structures. Our **barrier re‑evaluation benchmarks** compare an *unpatched* build (no branch tracking in the CDF / barrier engine) with a *patched* build where the normal CDF and barrier engine record every branch via `ABool`. The patched version shows that Forge can:

- keep using the **same JIT‑compiled kernel** across scenarios, and
- still respond correctly when barrier crossings and other branch outcomes change.

### Barrier kernel‑reuse example

We compile **one Forge barrier kernel** using an input where the strike is **above** the barrier, then **re‑use** that kernel for several different scenarios:

```text
Input Set 1: strike=100, barrier=100,  u=90,  r=0.10, v=0.10   (strike >= barrier)
Input Set 5: strike=80,  barrier=95,   u=100, r=0.05, v=0.20  (strike <  barrier)
```

- **Unpatched build (no `ABool` in CDF / barrier engine)**  
  - Kernel is recorded and compiled on **Input Set 1**.  
  - Re‑evaluating with **Input Set 5** reuses the *same* kernel, but **branch conditions are baked in** from Set 1.  
  - Result for Input Set 5:
    - Expected price: **10.08**, Forge price: **24.59** → **~144% error**.  
    - Greeks also diverge strongly (e.g. \(d/dB\): expected \(-1.70\) vs Forge \(0.00\)).  
  - **Outcome:** Input Sets 1–4 pass, **Input Set 5 fails** – kernel reuse is *not* safe when branches can flip.

- **Patched build (with `ABool` in ErrorFunction, CND, AnalyticBarrierEngine)**  
  - Same kernel‑creation setup: compile once on **Input Set 1**, then re‑evaluate on all 5 input sets.  
  - `ABool::If` records all relevant branches (normal CDF regions, asymptotics, strike/barrier conditions).  
  - Result for Input Set 5:
    - Expected price: **22.73**, Forge price: **22.73** (0% error).  
    - Greeks match bump‑reval to within numerical noise for all inputs.  
  - **Outcome:** **All 5 input sets pass**, showing that Forge can reuse a single barrier kernel even when barrier crossings change, as long as branch logic is expressed with `ABool`.

## Dependencies

- **[Forge](https://github.com/da-roth/forge)** - JIT compiler and AAD engine (zlib license)
- **[QuantLib](https://github.com/lballabio/QuantLib)** - Financial library (permissive license)
- **Boost** - Required by QuantLib

## Building

```bash
# Clone workspace
git clone <this-repo> quantlib-forge
cd quantlib-forge

# Build Forge package first (see forge/tools/packaging)
# Then configure and build:

cmake -B build -S . \
    -DForge_DIR=<path-to-forge-install>/lib/cmake/Forge \
    -DQuantLib_DIR=<path-to-quantlib-install>/lib/cmake/QuantLib

cmake --build build
```

## Project Structure

```
quantlib-forge/
  LICENSE              # AGPL-3.0-or-later
  NOTICE.md            # Third-party attributions
  CMakeLists.txt       # Main build configuration
  docs/                # Documentation
  patches/             # QuantLib ABool patch
  integration/         # Forge <-> QuantLib adapter code
    ql/                # QuantLib-specific headers
  harness/             # Benchmark harness (engine-agnostic)
  benchmarks/          # Benchmark executables
  examples/            # Usage examples
  cmake/               # CMake modules and presets
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for details.

Some code is derived from [QuantLib-Risks-Cpp](https://github.com/auto-differentiation/QuantLib-Risks-Cpp) (Xcelerit Computing Ltd., AGPL-3.0).
