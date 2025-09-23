# NEEDS TO BE EDITTED

## Experimental Setup

### Hardware Configuration
- **CPU**: Intel i5-12600KF, No ISA Support
- **CPU Frequency**: 3.69 GHz
- **Memory**: 32 GB (2 x 16 GB) DDR4-3600 CL16
- **OS**: Windows 10, Version 22H2, OS Build 19045.6332

### Software Configuration
- **Compiler**: Clang 19.1.5
- **Compiler Flags**:
  - Scalar: `-O2 -fno-tree-vectorize -mno-sse -mno-avx`
  - Vectorized: `-O3 -march=native -ffast-math`
  - AVX2: `-O3 -mavx2 -mfma -ffast-math`

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
- **Data Types**: float32, float64
- **Alignment**: Aligned (64-byte) vs unaligned
- **Stride**: 1, 2, 4, 8, 16
- **Array Sizes**: 1K to 16M elements (covering L1, L2, L3, DRAM)
- **Compiler Optimizations**: Scalar, auto-vectorized, AVX2-targeted

## Analysis Methods

### Performance Metrics
- **Time**: Execution time in seconds
- **GFLOP/s**: Billions of floating-point operations per second
- **CPE**: Cycles per element
- **Bandwidth**: Memory bandwidth in GB/s
- **Speedup**: Scalar time / Vectorized time

### SMT Control
- **Default**: Enabled
- **Controlled**: Disabled via command line argument `nosmt`

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

### Roofline Model
- **Arithmetic Intensity**: FLOPs per byte accessed
- **Peak Performance**: Theoretical maximum based on CPU specifications
- **Performance Characterization**: Compute-bound vs memory-bound

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

## Verification

### Vectorization Verification'
- Compiler reports (using `-Rpass=vector, -Rpass-missed=vectorize, -Rpass-analysis=vectorize` flags)
- Assembly inspection for vector instructions
- Performance patterns consistent with SIMD acceleration

### Correctness Verification
- Comparison against reference implementations
- Validation of numerical results
- Consistency across multiple runs