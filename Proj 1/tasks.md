Things I need to learn how to do:
- Can learn how to do the experiments later, these are the tools and experimental knobs (and tips)

* GCC with optimization enabled (e.g., high-level auto-vectorization)
- gcc -03 enables auto-vectorization (and optimization)
- gcc -03 -march=native optimizes further for my specific CPU

* Disassembly or compiler vectorization reports to confirm SIMD (e.g., opt reports; check for vector instructions in the binary).
- -fopt-info-vec* lets you see what is being vectorized
- 2> file.txt at the end of the gcc command saves to file

* System timers and statistical scripts to measure runtime, compute GFLOP/s and cycles per element (CPE), and generate plots.
- gpt has coded this in performance_measuring.c

* OS performance counters (e.g., perf) to report instructions retired, vector instruction count, and memory traffic

* aligned arrays vs deliberately misaligned

* sizes that are multiples of the vector width vs sizes with remainders (tail handling) (?)

* unit-stride (contiguous) vs strided (e.g., 2, 4, 8) or gather-like index patterns where applicable (?)

* Working-set size: within L1, within L2, within LLC, and DRAM-resident (increase N across these regimes) (?, what is N)

* Compiler / ISA flags: scalar-only (vectorization disabled), default auto-vectorized, and a build targeted to your CPU’s widest available vectors. Record any fast-math, FMA, FTZ/DAZ settings

* Fix CPU frequency (performance governor) and pin to a core to reduce run-to-run variance; document SMT state (?, SMT)

* Warm up data to populate caches for “hot” runs; also test cold runs where relevant and state which you report. (?)
- have warming function, dont know where cold runs are relevant

* Avoid denormals and constant-zero paths; initialize with non-trivial data. Consider FTZ/DAZ if supported (and document). (?)

* Check vector width available on your machine (e.g., 128-/256-/512-bit) and target that ISA with your compiler flags. 

* Verify alignment of arrays; test deliberately misaligned variants to see the cost. (?)

* Measure more than time: compute GFLOP/s, CPE, and memory traffic estimates; use multiple repetitions and report variability. 

* Record everything: compiler versions/flags, environment, thermal state; randomize run order to mitigate drift.




Tools I need:
* SAXPY / AXPY: y ← a x + y (streaming FMA) Kernel
* Dot product / reduction: s ← Σ x_i y_i (reduction) Kernel
* Elementwise multiply: z_i ← x_i · y_i (no reduction) Kernel
* 1D 3-point stencil: y_i ← a x_{i-1} + b x_i + c x_{i+1} (neighbor access) Kernel


