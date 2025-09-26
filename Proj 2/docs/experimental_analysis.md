# Experimental Analysis: Memory Hierarchy Performance

## Known Limitations & Anomalies

### 1. Windows Environment Constraints
**Observation:** The Linux perf tool was unavailable due to the Windows operating system environment.

**Hypothesis:** Windows performance counters (via windows.h, pdh.h, and pdhmsg.h) were utilized as functional equivalents to Linux's perf subsystem, providing comparable low-level performance monitoring capabilities.

### 2. 16 MB Random Access Cache Miss Impact
**Observation:** The 16MB random access data exhibits near-linear performance characteristics across varying stride lengths.

**Hypothesis:** At this working set size (exceeding typical L3 cache capacities), all accesses incur DRAM penalties regardless of stride pattern. The random access pattern dominates performance, making stride variations negligible compared to the fundamental memory latency overhead.

### 3. Random/Sequential Ratio Positive Working-set Size Sweep
**Observation:** In the working-set size sweep, certain configurations demonstrated better performance with random access patterns compared to sequential access.

**Hypothesis:** This counterintuitive result occurs with large strides where sequential access exhibits no spatial locality, each access targets a new cache line. Random access may occasionally benefit from accidental locality or avoid prefetcher-induced cache pollution, creating scenarios where random patterns outperform predictable but non-local sequential patterns.


## Pattern & Granularity Sweep Analysis

**Key Findings:** 
- Stride Impact: Increasing stride length reduces spatial locality but increases apparent bandwidth due to more cache lines being accessed

- Access Pattern: 100% read/write operations achieved highest bandwidth by eliminating read-write dependencies

- Sequential vs. Strided: Strided access patterns outperformed purely sequential access due to prefetcher effectiveness

**Explanation:** The memory subsystem demonstrates optimal performance when access patterns align with hardware prefetching capabilities. Sequential access with large strides defeats spatial locality without providing predictable patterns for prefetching, while moderate strides allow prefetchers to operate effectively. Pure read or write streams eliminate synchronization overhead between memory operations.


## Read/Write Mix Sweep Conclusions

**Performance Characteristics:**
- 100% R/W Operations: Showed marginal performance advantages over mixed workloads by eliminating read-after-write dependencies and reducing pipeline stalls

- Stride Optimization: 1KB strides demonstrated superior performance due to efficient page utilization—traversing 4KB pages in four operations maximizes TLB coverage while maintaining access efficiency

- Memory Controller Efficiency: Large strides reduce bank conflict opportunities and enable more efficient command scheduling in the memory controller

## Loaded Latency Intensity Sweep Analysis

**Knee Point Identification:**
The transition from latency-bound to bandwidth-bound operation occurs between 64B and 256B strides. This is evidenced by:

- Stride-64B: ~4.1ns latency, ~14.8 GB/s throughput (L3 cache bound)

- Stride-256B/1024B: ~7.0ns latency, ~8.8 GB/s throughput (DRAM bound)

The knee point represents the working set size where cache capacity is exceeded and memory latency becomes the dominant performance factor.

**Bandwidth Characterization:** 
- Observed Peak Bandwidth: ~15 GB/s (single-core)
- Theoretical Maximum: ~28.8 GB/s (single-channel DDR4-3600)

The achieved bandwidth represents approximately 52% of theoretical single-channel peak, indicating realistic memory controller efficiency for single-threaded workloads. The performance core demonstrates effective memory parallelism despite operating within single-core constraints. 

**Little's Law Validation:** Applying Little's Law (L = λ × W) to the memory subsystem:

- Cache-bound regime: (14.8 GB/s ÷ 64 B/line) × 4.1 ns ≈ 0.95 concurrent requests

- Memory-bound regime: (8.8 GB/s ÷ 64 B/line) × 7.0 ns ≈ 0.96 concurrent requests

The consistent concurrency level (~0.95) across regimes confirms Little's Law and reveals the core's inherent memory parallelism limitation. This explains why additional threads on the same core cannot increase aggregate bandwidth—the memory-level parallelism ceiling is determined by core architecture rather than thread count.

## Cache Miss Impact Analysis

### Strong Correlation Evidence: 
The data demonstrates exponential performance degradation as working set sizes exceed cache hierarchy boundaries:

**Performance Ratios:**

- L1 cache hits: ~2 billion operations/second

- L3 cache hits: ~600 million operations/second

- DRAM access: ~65 million operations/second

## Average Memory Access Time (AMAT) Framework:
The results validate the AMAT model:
**AMAT = Hit Time + Miss Rate × Miss Penalty**

As stride increases and working set sizes grow, miss rates escalate dramatically, causing AMAT to be dominated by miss penalties. The 1000x performance variation between best-case (cache-resident) and worst-case (DRAM-bound) scenarios underscores the critical importance of cache optimization.

## TLB Miss Impact Conclusions

### Clear Performance Correlation:
The experimental data shows strong correlation between TLB reach and performance:

**Key Observations:**

- Larger page sizes increase DTLB coverage, reducing miss rates

- The relationship follows expected patterns: DTLB Reach = DTLB Entries × Page Size

- Performance plateaus occur when working set sizes exceed TLB coverage capacity

**Architectural Implications:**
The results demonstrate that TLB efficiency is as critical as cache efficiency for memory-intensive workloads. Optimal performance requires balancing page size selection with application access patterns to maximize TLB hit rates while maintaining efficient memory utilization.

