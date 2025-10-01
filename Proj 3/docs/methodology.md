# SSD Performance Characterization Experiment

## Project Overview
This project focuses on comprehensive performance analysis of Solid State Drives (SSDs) through systematic testing methodologies. The experiment characterizes SSD behavior under various workload conditions to understand performance characteristics across different usage scenarios.


## Experimental Setup

### Hardware Configuration
- **CPU**: Intel i5-12600KF, No ISA Support
- **CPU Frequency**: 3.69 GHz
- **RAM**: 32 GB (2 x 16 GB) DDR4-3600 CL16
- **OS**: Windows 10, Version 22H2, OS Build 19045.6332
- **Memory**: 2TB Samsung 980 SSD
- **Interface**: PCIe 4.0 x16 slot

### Software Configuration
- **Benchmarking Tool**: FIO (Flexible I/O Tester)
- **I/O Engine**: Windows Native Asynchronous I/O
- **Test File**: 10GB dedicated test file on target drive (D)
- **File System**: NTFS with default allocation

## Measurement Methodology

### Test Execution Protocol
1. **Preconditioning**: Drive preconditioning before each test series (unless skipped)
2. **Ramp Time**: 5 seconds for system stabilization
3. **Measurement Duration**: 30 seconds per individual test run
4. **Cooldown Period**: 10 seconds between test iterations
5. **Statistical Reliabilit**y: 3 repetitions per configuration

### Data Collection
- **Performance Metrics**: IOPS, bandwidth (MB/s), latency (μs)
- **Latency Distribution**: p50, p95, p99 percentiles
- **Throughput Analysis**: Read and write operations separately tracked
- **Error Analysis**: Standard deviation across multiple runs

## Workloads Tested

### Access Pattern Analysis
- **Sequential Access**: Optimized for bandwidth utilization
- **Random Access**: Optimized for command processing efficiency

### Block Size Granularity
- **Small Blocks**: "8", "16", "32"
- **Medium Blocks**: "64", "128"
- **Large Blocks**: "256", "512", "1024"

### Read/Write Mix Profiles
- **100r0w**: Pure read workload
- **70r30w**: Read-dominated mixed workload
- **50r50w**: Balanced read/write workload
- **0r100w**: Pure write workload

## Experimental Variables

### Queue Depth Scalability
- **Range**: 1 to 64 concurrent commands
- **Test Points**: [1, 2, 4, 8, 16, 32, 64]

## Analysis Methods
### Performance Metrics
- **IOPS**: Input/Output Operations Per Second
- **Bandwidth**: Megabytes per second (MB/s)
- **Latency**: Microseconds per operation (μs)
- **Throughput**: Effective data transfer rate
- **Tail Latency**: p50, p95, and p99 latency percentiles

### Statistical Processing
- **Mean Values**: Average performance across 3 repetitions
- **Standard Deviatio**n: Measurement variability indicator
- **Error Bars**: ±1 standard deviation in graphical representations
- **Correlation Analysis**: Performance relationships across parameters

## Limitations
### Known Constraints
- **System Cache Effects**: Despite direct I/O, some cache interactions possible
- **Background Processes**: OS and antivirus may introduce minor variability
- **Thermal Throttling**: Potential performance impact during extended runs
- **Drive Conditioning**: Freshness of NAND blocks may affect write performance

### Mitigation Strategies
- **Multiple Iterations**: Average across 3 runs to minimize outliers
- **System Preparation**: Close unnecessary applications before testing
- **Consistent Conditions**: Maintain identical test environment
- **Statistical Validation**: Error bars indicate measurement reliability
- **Preconditioning**: Drive returned to consistent state between test series

### Experimental Boundaries
- **Duration Limitations**: 30-second tests may not capture long-term behavior
- **Workload Simplification**: Synthetic workloads vs. real-world patterns
- **Queue Depth Maximum**: 256 may not saturate all high-end SSDs
- **Block Size Range**: Limited to common enterprise and consumer sizes