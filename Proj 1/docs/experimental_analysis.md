# Experimental Analysis: SIMD Performance

## Known Limitations & Anomalies

### 1. Minimal SIMD Speedup
**Observation:** Very small speedup when transitioning from scalar to SIMD implementations.

**Hypothesis:** It appears my processor was not designed with efficient SIMD operations in mind, as evidenced by the lack of integrated graphics (which often provide additional SIMD capabilities). 

### 2. High Measurement Error
**Observation:** Generally high variance in performance measurements.

**Hypothesis:** My suboptimal cooling infrastructure likely caused thermal throttling, resulting in process slowdowns as experiments progressed and components heated up.

### 3. Performance Spike at Smaller Array Sizes
**Observation:** A notable increase in speedup occurs at array sizes of approximately 10^3 elements.

**Hypothesis:** The small array size allows the arrays to fit in the L1 cache. Since L1 is the smallest, fastest cach, the operation likely avoided memory bottlenecks, allowing SIMD to demonstrate its potential over scalar implementations.

## SIMD Gain Compression Analysis

**Observation:** No significant compression of SIMD gains was observed in these experiments.

**Explanation:** The limited SIMD capabilities of the test CPU resulted in generally poor SIMD performance overall. The maximum speedup observed was approximately 1.5x, which occurred under specific ideal conditions, when array sizes aligned optimally with the L2 cache capacity.

**Additional Finding:** At small array sizes (10³ elements), 32-bit data showed modest speedup while 64-bit SIMD instructions actually caused performance degradation due to excessive overhead relative to the small problem size.

## Compute vs. Memory-Bound Conclusions

The Elementwise Multiply kernel is confirmed to be memory-bound, as its data points reside on the memory-bound segment of the roofline. This is an expected result, as this kernel has the lowest arithmetic intensity (FLOPs/byte) of the three, performing only a single operation per element while still requiring three memory accesses. Consequently, its performance is constrained by the available memory bandwidth of the system.<br>
In contrast, the AXPY and Dot Product kernels are shown to be compute-bound, with their data points occuring along the compute-bound segment of the roofline. This indicates that their higher arithmetic intensity (twice that of the Elementwise Multiply kernel) allows them to fully utilize the available computational resources of the processor. Their performance is therefore limited by the throughput of the CPU rather than the memory subsystem.

## Expected vs. Actual SIMD Gains

**Theoretical Expectations:**
- 32-bit instructions: 4× ideal speedup
- 64-bit instructions: 8× ideal speedup
- Real-world expectation: ~2× speedup (accounting for various overheads)

**Experimental Results:** The observed speedup was negligible across most test conditions, falling significantly below expected values. This anomalous behavior is addressed in the Limitations section above, primarily attributed to hardware constraints rather than methodological issues.