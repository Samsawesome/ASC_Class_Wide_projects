# SSD Performance Characterization Experiment

## Project Overview
This project focuses on comprehensive performance analysis of Solid State Drives (SSDs) through systematic testing methodologies. The experiment characterizes SSD behavior under various workload conditions to understand performance characteristics across different usage scenarios.

## Experimental Methodology

### Test Configuration
- **Experiment Duration**: 30 seconds per individual test run
- **Cooldown Time**: 10 seconds to allow system stabilization before additional measurements
- **Iterations**: 3 repetitions per test configuration for statistical significance

### Variable Parameters

#### Queue Depth Analysis
The experiment employs a comprehensive queue depth sweep to evaluate how command queuing affects performance:
- **Range**: 1 to 256 concurrent commands
- **Specific Test Points**: [1, 2, 4, 8, 16, 32, 64, 128, 256]
- **Mid-point Reference**: 8 (typical for many applications)

#### Block Size Variation
Multiple block sizes are tested to understand I/O efficiency:
- Small blocks: "4k", "16k", "32k" (typical for random I/O)
- Medium blocks: "64k", "128k" (balanced workloads)
- Large blocks: "256k", "512k", "1m" (sequential/throughput scenarios)

#### Read/Write Mix Patterns
Four distinct workload profiles simulate real-world usage:
- **100r0w**: Pure read workload (data retrieval scenarios)
- **70r30w**: Read-dominated mixed workload (typical database operations)
- **50r50w**: Balanced read/write workload (general purpose computing)
- **0r100w**: Pure write workload (data ingestion/backup scenarios)

## System Under Test
- **SSD Model**: Samsung 970 EVO Plus
- **Interface**: PCIe 3.0 x4
- **Capacity**: 1TB
- **Processor**: Intel Core i7-10700K
- **Operating System**: Windows 10
- **Filesystem**: ext4

## Measurement Approach
1. **Baseline Establishment**: Ramp time ensures consistent starting conditions
2. **Multi-iteration Testing**: Three repetitions minimize measurement variance
3. **Comprehensive Parameter Space**: Full factorial testing across all parameter combinations
4. **Real-world Simulation**: Workload mixes represent actual usage patterns

## Expected Outcomes
- Performance curves showing IOPS, latency, and throughput across different queue depths
- Characterization of optimal block sizes for different workload types
- Identification of performance saturation points and bottlenecks
- Comparative analysis of read vs. write performance characteristics