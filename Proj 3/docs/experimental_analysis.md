# Experimental Analysis: Memory Hierarchy Performance

## Known Limitations & Anomalies

### 1. 256B Sequential Block Size Performance Anomaly

**Observation:** The 256B sequential block size configuration showed unexpectedly lower bandwidth compared to adjacent block sizes, breaking the expected positive correlation.

**Hypothesis:** This likely results from cache thrashing effects in the processor's memory hierarchy. Modern CPU caches typically feature 4-way or 8-way associativity. When performing 256B accesses, the workload may be mapping to only 4 cache lines simultaneously, creating contention and reducing effective bandwidth utilization. This hints towards my CPU's caches using 4-way associativity.

### 2. IOPS Saturation at High Queue Depths

**Observation:** The queue depth sweep revealed a distinct IOPS saturation point where additional queue depth failed to significantly improve throughput.

**Hypothesis:** This saturation is the throughput limit of my storage controllers. Beyond this point, my device has reached its maximum processing capability, and additional queued commands increase only latency without improving IOPS. 

### 3. Read Latency Variance in Mixed Workloads

**Observation:** The read/write mix analysis showed significant read latency error only in the 100% read configuration, contrary to expected patterns.

**Hypothesis:** This counterintuitive result may stem from the storage controller's algorithms and background operations. During pure read workloads, the controller may opportunistically execute background maintenance tasks that temporarily impact read latency, such as garbage collection. In mixed workloads, the controller's scheduling algorithms may prioritize maintaining consistent latency profiles for read operations.


## Block Size & Pattern Sweep Analysis

### Key Performance Characteristics

**Random Access Pattern:**
- Negative correlation between IOPS and block size 
- Performance dominated at smaller block sizes
- Increased latency as block size increases

**Sequential Access Pattern:**
- Positive correlation between bandwidth and block size
- Performance limited by bandwidth at larger block sizes
- Linear bandwidth scaling until physical interface limits
- Increased latency as block size increases

### Architectural Crossover Point

**Critical Finding:** Both access patterns transitioned from IOPS-dominated to bandwidth-dominated regimes around 64KB block size threshold.

**Technical Explanation:**
- IOPS-dominated regime (≤64KB): Performance is constrained by the storage controller's processing capability and latency. Smaller operations maximize IOPS but underutilize available bandwidth.
- Bandwidth-dominated regime (≥64KB): Performance becomes limited by bandwidth. Larger operations maximize bandwidth usage and become limited by latency.

**Fundamental Difference Between Patterns:** Sequential access optimizes for bandwidth efficiency by enabling prefetching and read-ahead algorithms, while random access maximizes IOPS by leveraging the controller's parallel command processing capabilities. This distinction arises from how each pattern interacts with the storage media's physical characteristics and the controller's scheduling algorithms.


## Read/Write Mix Sweep Analysis
### Performance Asymmetry Characteristics
**Throughput Analysis:**
- Read operations demonstrated ~10% higher IOPS compared to write operations
- Maximum IOPS achieved at 70/30 read write mix

**Latency Characteristics:**
- Write operations had comparable latency to reads at 50/50 mix, but much higher relative latency at 70/30 read write mix
- Write operations also had very high latency error
- Read latency appearred parabolic, with 50/50 and 100% read having higher latency than 70/30 read write mix

### Architectural Explanations

**Write Latency Differences:** Write operations suffer from increased latency as the number of write operations increases, where physical writes exceed digital writes. This explains both the performance penalty and increased latency error.

**Performance Maximum:** The performance maximum at 70/30 read write mix hints that the memory controller's algorithms achieve maximum efficiency when balancing read and write commands. This ratio most likely optimizes parallel executions.



## Queue Depth & Parallelism Analysis
### Scalability Characterization

**Knee Point Identification:**
The knee point was identified to be at ~120 KIOPS, at a queue depth(QD) of 32.

**Little's Law Validation:**
At the identified knee point:

-Throughput (λ) = 122.1 KIOPS

-Latency (W) = 176.8μs

-Concurrency (L) = λ × W = 21.6

This calculated concurrency of 21.6 is close to the actual QD of 32 in this region. If a QD between 16 and 32 was added to the simulation, there is a strong possibility the knee point would be found there instead.


### Diminishing Returns Analysis

**Performance Regions:**
1. **Linear Scaling (QD 1-16):** Almost exponential IOPS improvement with almost no latency impact

2. **Transition Zone (QD 16-32):** Diminishing IOPS gains with larger latency increases

3. **Saturation Region (QD >32):** Minimal throughput improvement with massive latency growth

**Tail Latency Implications:**

The p95 and p99 latency percentiles showed a large latency increase around the knee point:

- p95 latency increased ~3× from QD8 to QD32

- p99 latency also increased ~3x over the same range

This shows that while average throughput may appear stable, latency can increase substantially in conditions with high queue depths.

