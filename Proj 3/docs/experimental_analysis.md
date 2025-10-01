# Experimental Analysis: Memory Hierarchy Performance

## Known Limitations & Anomalies

### 1. 256B Block Size Performance Anomaly

**Observation:** The 256B block size configuration exhibited unexpectedly lower bandwidth compared to adjacent block sizes, breaking the expected performance correlation.

**Hypothesis:** This anomaly likely results from cache thrashing effects in the processor's memory hierarchy. Modern CPU caches typically feature 4-way or 8-way associativity. When performing 256B accesses, the workload may be mapping to only 4 cache lines simultaneously, creating contention and reducing effective bandwidth utilization. The memory controller's prefetching algorithms may also be less effective at this specific block size, failing to optimize data movement patterns.

### 2. IOPS Saturation at High Queue Depths

**Observation:** The read/write mix analysis showed significant read latency variance only in the 100% read configuration, contrary to expected patterns.

**Hypothesis:** This saturation represents the fundamental throughput limit of the storage controller. Beyond this point, the device has reached its maximum command processing capability, and additional queued commands only increase latency without improving throughput. This behavior is characteristic of Amdahl's Law in storage systems, where parallelization benefits diminish as system bottlenecks become dominant.

### 3. Read Latency Variance in Mixed Workloads

**Observation:** Read in read/write sweep had only one non-negligable latency error when 100% read

**Hypothesis:** This counterintuitive result may stem from the storage controller's quality-of-service (QoS) algorithms and background operations. During pure read workloads, the controller may opportunistically execute background maintenance tasks (garbage collection, wear leveling) that temporarily impact read latency. In mixed workloads, the controller's scheduling algorithms may prioritize maintaining consistent latency profiles across operation types.





## Block Size & Pattern Sweep Analysis

### Key Performance Characteristics

**Random Access Pattern:**
- Demonstrated strong negative correlation between IOPS and block size (R² = 0.94)
- Performance dominated by command processing overhead at smaller block sizes
- Exponential IOPS degradation as block size increases
**Sequential Access Pattern:**
- Exhibited positive correlation between bandwidth and block size (R² = 0.89)

- Performance limited by interface bandwidth and controller efficiency at larger block sizes

- Linear bandwidth scaling until physical interface limits
### Architectural Crossover Point

**Critical Finding:** Both access patterns transitioned from IOPS-dominated to bandwidth-dominated regimes at the 64KB block size threshold.

**Technical Explanation:**
- IOPS-dominated regime (≤64KB): Performance is constrained by the storage controller's command processing capability and protocol overhead. Smaller transactions maximize the number of operations per second but underutilize available bandwidth.
- Bandwidth-dominated regime (≥64KB): Performance becomes limited by the physical interface bandwidth (SATA/NVMe bandwidth) and NAND flash array throughput. Larger transactions minimize protocol overhead but reduce operational density.

**Fundamental Difference Between Patterns:** Sequential access optimizes for bandwidth efficiency by enabling prefetching and read-ahead algorithms, while random access maximizes IOPS by leveraging the controller's parallel command processing capabilities. This distinction arises from how each pattern interacts with the storage media's physical characteristics and the controller's scheduling algorithms.


## Read/Write Mix Sweep Analysis
### Performance Asymmetry Characteristics
**Throughput Analysis:**
- Read operations demonstrated 8-12% higher IOPS compared to write operations at equivalent queue depths
- Optimal throughput balance achieved at 70% read / 30% write mix
- Maximum aggregate IOPS observed at the 70/30 ratio (142K IOPS vs 135K at 100% read)

**Latency Characteristics:**
- Write operations exhibited 2.3× higher latency variance compared to reads
- Read latency remained relatively stable across mix ratios (σ = 18.2μs)
- Write latency showed significant sensitivity to workload composition (σ = 42.7μs)

### Architectural Explanations

**Write Amplification Impact:** Write operations inherently suffer from write amplification in flash-based storage, where physical writes exceed logical writes due to garbage collection and wear leveling. This fundamental characteristic explains both the performance penalty and increased latency variance.

**Controller Buffer Dynamics:** The performance optimum at 70/30 mix suggests the controller's buffer management algorithms achieve optimal efficiency when balancing read and write command streams. This ratio likely maximizes concurrent execution opportunities while minimizing resource contention.

**Error Variance Explanation:** The higher write latency variance stems from the non-deterministic nature of flash program/erase cycles and background maintenance operations. Read operations, being non-destructive, benefit from more predictable access patterns and controller optimizations.

## Queue Depth & Parallelism Analysis
### Scalability Characterization

**Knee Point Identification:**
Through systematic analysis, the performance knee point was identified at approximately 120 KIOPS, corresponding to a queue depth of 32.

**Little's Law Validation:**
At the identified knee point:

-Throughput (λ) = 122.1 KIOPS

-Latency (W) = 176.8μs

-Concurrency (L) = λ × W = 21.6

This calculated concurrency of 21.66 is close to the actual queue depth of 32 in this region, validating Little's Law application to storage subsystem analysis. If a queue depth between 16 and 32 was added to the simulation, there is a strong chance the knee point would be found there instead.


### Diminishing Returns Analysis

**Performance Regions:**
1. **Linear Scaling (QD 1-16):** Near-linear throughput improvement with minimal latency impact

2. **Transition Zone (QD 16-32):** Diminishing throughput gains with moderate latency increases

3. **Saturation Region (QD >32):** Negligible throughput improvement with exponential latency growth

**Tail Latency Implications:**

The p95 and p99 latency percentiles exhibited significant degradation at the knee point:

- p95 latency increased ~3× from QD8 to QD32

- p99 latency also increased ~3x over the same range

This demonstrates that while average throughput may appear stable, user-experienced latency can degrade substantially in saturated conditions.

