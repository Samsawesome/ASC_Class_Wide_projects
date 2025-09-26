# Cache & Memory Performance Methodology

## Experimental Setup

### Hardware Configuration
- **CPU**: Intel i5-12600KF, No ISA Support
- **CPU Frequency**: 3.69 GHz
- **Memory**: 32 GB (2 x 16 GB) DDR4-3600 CL16
- **OS**: Windows 10, Version 22H2, OS Build 19045.6332

### Software Configuration
- **Compiler**: Clang 19.1.5
- **Compiler Flags**: `-O2 -lpdh -olatency.exe -lAdvapi32`


### Measurement Methodology
1. **Warm-up**: 10 iterations to populate caches
2. **Measurement**: 100 timed iterations
3. **Repetitions**: 3 runs per configuration for statistical reliability
4. **Data Initialization**: Non-trivial patterns to avoid compiler optimizations

### Kernels Tested
1. **AXPY**: y ← a * x + y
2. **Dot Product**: s ← Σ x_i * y_i
3. **Elementwise Multiply**: z_i ← x_i * y_i

### Experimental Variables
- **Access pattern/granularity**: sequential vs. random; strides ≈64B / ≈256B / ≈1024B.
- **Read/write ratio**: 100%R, 100%W, 70/30, 50/50, 20/80.
- **Concurrency**: MLC’s loaded-latency mode.


## Analysis Methods

### Performance Metrics
- **Time**: Execution time in seconds
- **Bandwidth**: Memory bandwidth in MB/s
- **Performance**: Operations / second
- **Throughput**: MB / second
- **Latency**: ns / operation, or clock cycles

### SMT Control
- **Default**: Enabled

### CPU Affinity and Priority
- **Process**: Pinned to specific core
- **Thread Priority**: Priority set to `HIGHEST`
- **Process Priority**: Priority set to `HIGH`

### Data Initialization Strategy
- **Features**: Non-zero values, Pattern variation, Modulo operations, Avoiding denormals

### Repetition Scheme
- **Multiple Runs**: 3 repetitions of each experiment
- **Varied patterns**: Different initialization patterns for each run
- **Comprehensive testing**: Multiple array sizes, data types, and memory alignments

### Statistical Processing
- Mean and standard deviation across multiple runs
- Error bars in plots represent one standard deviation

## Limitations

### Known Issues
1. **Windows Performance Counters**: Limited access to detailed hardware counters
2. **Frequency Scaling**: CPU may throttle during intensive computations
3. **Thermal Effects**: Performance may decrease during extended runs
4. **Background Processes**: Other applications may affect measurements

### Mitigation Strategies
1. **Multiple Runs**: Average across multiple measurements
2. **System Preparation**: Close unnecessary applications before testing
3. **Constant Frequency**: Disable CPU frequency scaling in BIOS if possible
4. **Power Settings**: Use high-performance power plan
5. **Cooling**: Ensure consistent thermal conditions with adequate cooling