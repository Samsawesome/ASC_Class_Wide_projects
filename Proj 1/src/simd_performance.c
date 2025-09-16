#include "simd_performance.h"
#include <intrin.h>

// Global results collector
results_collector_t global_collector;

// ==================== UTILITY FUNCTIONS ====================
double get_time() {
    LARGE_INTEGER frequency, time;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / frequency.QuadPart;
}

void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        fprintf(stderr, "Aligned allocation failed!\n");
        exit(1);
    }
    return ptr;
}

void aligned_free(void* ptr) {
    _aligned_free(ptr);
}

double detect_cpu_frequency() {
    HKEY hKey;
    DWORD frequency = 0;
    DWORD dataSize = sizeof(frequency);
    
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, 
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        if (RegQueryValueExA(hKey, "~MHz", NULL, NULL, 
                            (LPBYTE)&frequency, &dataSize) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return frequency / 1000.0;
        }
        RegCloseKey(hKey);
    }
    
    // Fallback: use CPUID to get brand string and estimate frequency
    int cpuInfo[4] = {0};
    char cpuBrandString[0x40] = {0};
    
    __cpuid(cpuInfo, 0x80000000);
    if ((unsigned)cpuInfo[0] >= 0x80000004) {
        __cpuid(cpuInfo, 0x80000002);
        memcpy(cpuBrandString, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000003);
        memcpy(cpuBrandString + 16, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000004);
        memcpy(cpuBrandString + 32, cpuInfo, sizeof(cpuInfo));
        
        // Look for GHz in brand string
        char* ghzPos = strstr(cpuBrandString, "GHz");
        if (ghzPos) {
            // Walk backward to find the frequency number
            char* numStart = ghzPos;
            while (numStart > cpuBrandString && 
                  (*(numStart-1) == '.' || isdigit(*(numStart-1)))) {
                numStart--;
            }
            
            char freqStr[16] = {0};
            strncpy(freqStr, numStart, ghzPos - numStart);
            return atof(freqStr);
        }
    }
    
    return 2.5; // Default fallback
}

void get_cpu_model(char* buffer, size_t size) {
    int cpuInfo[4] = {0};
    char cpuBrandString[0x40] = {0};
    
    __cpuid(cpuInfo, 0x80000000);
    if ((unsigned)cpuInfo[0] >= 0x80000004) {
        __cpuid(cpuInfo, 0x80000002);
        memcpy(cpuBrandString, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000003);
        memcpy(cpuBrandString + 16, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000004);
        memcpy(cpuBrandString + 32, cpuInfo, sizeof(cpuInfo));
        
        // Clean up the brand string
        for (int i = 0; i < sizeof(cpuBrandString); i++) {
            if (cpuBrandString[i] == 0) break;
            if (cpuBrandString[i] < 32 || cpuBrandString[i] > 126) {
                cpuBrandString[i] = ' ';
            }
        }
        
        // Remove extra spaces
        char* dst = cpuBrandString;
        char* src = cpuBrandString;
        int prevSpace = 0;
        
        while (*src) {
            if (*src == ' ') {
                if (!prevSpace) {
                    *dst++ = ' ';
                    prevSpace = 1;
                }
            } else {
                *dst++ = *src;
                prevSpace = 0;
            }
            src++;
        }
        *dst = 0;
        
        strncpy(buffer, cpuBrandString, size - 1);
        buffer[size - 1] = 0;
    } else {
        strncpy(buffer, "Unknown CPU", size - 1);
    }
}

void get_compiler_version(char* buffer, size_t size) {
    #ifdef __clang__
    snprintf(buffer, size, "Clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
    #elif __GNUC__
    snprintf(buffer, size, "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #elif _MSC_VER
    snprintf(buffer, size, "MSVC %d", _MSC_VER);
    #else
    strncpy(buffer, "Unknown compiler", size - 1);
    #endif
}

void initialize_data(void* data, size_t n, data_type_t type, int pattern) {
    switch (type) {
        case TYPE_F32: {
            f32* fdata = (f32*)data;
            for (size_t i = 0; i < n; i++) {
                // Use different patterns to avoid trivial optimizations
                switch (pattern % 3) {
                    case 0: fdata[i] = 1.0f + 0.001f * (i % 100); break;
                    case 1: fdata[i] = 0.5f + 0.0001f * (i % 73); break;
                    case 2: fdata[i] = 2.0f + 0.00001f * (i % 51); break;
                }
            }
            break;
        }
        case TYPE_F64: {
            f64* fdata = (f64*)data;
            for (size_t i = 0; i < n; i++) {
                switch (pattern % 3) {
                    case 0: fdata[i] = 1.0 + 0.001 * (i % 100); break;
                    case 1: fdata[i] = 0.5 + 0.0001 * (i % 73); break;
                    case 2: fdata[i] = 2.0 + 0.00001 * (i % 51); break;
                }
            }
            break;
        }
        case TYPE_I32: {
            i32* idata = (i32*)data;
            for (size_t i = 0; i < n; i++) {
                switch (pattern % 3) {
                    case 0: idata[i] = 1 + (i % 100); break;
                    case 1: idata[i] = 1000 + (i % 73); break;
                    case 2: idata[i] = 50000 + (i % 51); break;
                }
            }
            break;
        }
    }
}

void perf_init(perf_metrics_t* metrics, uint64_t operations, size_t elements, double cpu_freq_ghz) {
    memset(metrics, 0, sizeof(perf_metrics_t));
    metrics->operations = operations;
    metrics->elements = elements;
    metrics->cpu_freq_ghz = cpu_freq_ghz;
}

// ==================== KERNEL IMPLEMENTATIONS ====================

// AXPY kernels
void axpy_scalar_f32(f32 a, f32* x, f32* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_scalar_f64(f64 a, f64* x, f64* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_vectorized_f32(f32 a, f32* x, f32* y, size_t n) {
    // Let compiler auto-vectorize
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_vectorized_f64(f64 a, f64* x, f64* y, size_t n) {
    // Let compiler auto-vectorize
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Dot product kernels
void dot_product_scalar_f32(f32* x, f32* y, size_t n, f32* result) {
    f32 sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

void dot_product_scalar_f64(f64* x, f64* y, size_t n, f64* result) {
    f64 sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

void dot_product_vectorized_f32(f32* x, f32* y, size_t n, f32* result) {
    f32 sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

void dot_product_vectorized_f64(f64* x, f64* y, size_t n, f64* result) {
    f64 sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

// Elementwise multiply kernels
void elementwise_multiply_scalar_f32(f32* x, f32* y, f32* z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

void elementwise_multiply_scalar_f64(f64* x, f64* y, f64* z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

void elementwise_multiply_vectorized_f32(f32* x, f32* y, f32* z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

void elementwise_multiply_vectorized_f64(f64* x, f64* y, f64* z, size_t n) {
    for (size_t i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

// ==================== RESULTS COLLECTION ====================
void results_init(results_collector_t* collector, double cpu_freq_ghz, const char* compiler_flags) {
    collector->count = 0;
    collector->cpu_freq_ghz = cpu_freq_ghz;
    collector->buffer_overflow = 0;
    memset(collector->results, 0, sizeof(collector->results));
    
    get_cpu_model(collector->cpu_model, sizeof(collector->cpu_model));
    get_compiler_version(collector->compiler_version, sizeof(collector->compiler_version));
}

void results_add(results_collector_t* collector, result_t result) {
    if (collector->count < MAX_RESULTS) {
        collector->results[collector->count++] = result;
    } else {
        if (!collector->buffer_overflow) {
            fprintf(stderr, "Warning: Results buffer full! Some results will be lost.\n");
            collector->buffer_overflow = 1;
        }
        // Optionally, you could implement a circular buffer or
        // write to disk immediately when buffer is full
    }
}

void results_save_csv(const results_collector_t* collector, const char* filename) {
    FILE* file;
    fopen_s(&file, filename, "w");
    if (!file) {
        perror("Failed to open results file");
        return;
    }
    
    // Write metadata
    fprintf(file, "# CPU: %s\n", collector->cpu_model);
    fprintf(file, "# CPU Frequency: %.2f GHz\n", collector->cpu_freq_ghz);
    fprintf(file, "# Compiler: %s\n", collector->compiler_version);
    fprintf(file, "# \n");
    
    // Write CSV header
    fprintf(file, "kernel,implementation,data_type,array_size,aligned,stride,time_seconds,gflops,cpe,bandwidth_gbs,speedup,run_id,compiler_flags\n");
    
    // Write data
    for (int i = 0; i < collector->count; i++) {
        const result_t* r = &collector->results[i];
        
        const char* kernel_str;
        switch (r->kernel) {
            case KERNEL_AXPY: kernel_str = "axpy"; break;
            case KERNEL_DOT_PRODUCT: kernel_str = "dot_product"; break;
            case KERNEL_ELEMENTWISE_MULTIPLY: kernel_str = "elementwise_multiply"; break;
            default: kernel_str = "unknown";
        }
        
        const char* impl_str = (r->implementation == IMPL_SCALAR) ? "scalar" : "vectorized";
        const char* type_str;
        switch (r->data_type) {
            case TYPE_F32: type_str = "f32"; break;
            case TYPE_F64: type_str = "f64"; break;
            case TYPE_I32: type_str = "i32"; break;
            default: type_str = "unknown";
        }
        
        fprintf(file, "%s,%s,%s,%zu,%d,%d,%.9f,%.3f,%.3f,%.3f,%.3f,%d,%s\n",
                kernel_str, impl_str, type_str, r->array_size, r->aligned, r->stride,
                r->time_seconds, r->gflops, r->cpe, r->bandwidth_gbs, r->speedup, 
                r->run_id, r->compiler_flags);
    }
    
    fclose(file);
    printf("Results saved to %s (%d records)\n", filename, collector->count);
}

void results_print_summary(const results_collector_t* collector) {
    printf("\n=== RESULTS SUMMARY ===\n");
    printf("CPU: %s\n", collector->cpu_model);
    printf("CPU Frequency: %.2f GHz\n", collector->cpu_freq_ghz);
    printf("Compiler: %s\n", collector->compiler_version);
    printf("Total runs: %d\n", collector->count);
    
    // Group by kernel and implementation
    for (int kernel = KERNEL_AXPY; kernel <= KERNEL_ELEMENTWISE_MULTIPLY; kernel++) {
        for (int impl = IMPL_SCALAR; impl <= IMPL_VECTORIZED; impl++) {
            double total_time = 0.0;
            double total_gflops = 0.0;
            double total_bandwidth = 0.0;
            int count = 0;
            
            for (int i = 0; i < collector->count; i++) {
                const result_t* r = &collector->results[i];
                if (r->kernel == kernel && r->implementation == impl) {
                    total_time += r->time_seconds;
                    total_gflops += r->gflops;
                    total_bandwidth += r->bandwidth_gbs;
                    count++;
                }
            }
            
            if (count > 0) {
                const char* kernel_str;
                switch (kernel) {
                    case KERNEL_AXPY: kernel_str = "AXPY"; break;
                    case KERNEL_DOT_PRODUCT: kernel_str = "Dot Product"; break;
                    case KERNEL_ELEMENTWISE_MULTIPLY: kernel_str = "Elementwise Multiply"; break;
                    //case KERNEL_STENCIL_3POINT: kernel_str = "3-Point Stencil"; break;
                    //case KERNEL_MEMORY_BANDWIDTH: kernel_str = "Memory Bandwidth"; break;
                    default: kernel_str = "Unknown";
                }
                
                const char* impl_str = (impl == IMPL_SCALAR) ? "Scalar" : "Vectorized";
                
                printf("%s (%s): Avg Time=%.6fs, Avg GFLOP/s=%.3f, Avg BW=%.3f GB/s (n=%d)\n",
                       kernel_str, impl_str, total_time/count, total_gflops/count, 
                       total_bandwidth/count, count);
            }
        }
    }
}

void calculate_speedups(results_collector_t* collector) {
    // For each configuration, find scalar and vectorized versions and calculate speedup
    for (int i = 0; i < collector->count; i++) {
        result_t* current = &collector->results[i];
        if (current->implementation == IMPL_VECTORIZED) {
            // Find corresponding scalar result
            for (int j = 0; j < collector->count; j++) {
                result_t* scalar = &collector->results[j];
                if (scalar->implementation == IMPL_SCALAR &&
                    scalar->kernel == current->kernel &&
                    scalar->data_type == current->data_type &&
                    scalar->array_size == current->array_size &&
                    scalar->aligned == current->aligned &&
                    scalar->stride == current->stride &&
                    strcmp(scalar->compiler_flags, current->compiler_flags) == 0) {
                    current->speedup = scalar->time_seconds / current->time_seconds;
                    break;
                }
            }
        }
    }
}

// ==================== EXPERIMENT RUNNERS ====================

typedef struct {
    size_t n;
    data_type_t data_type;
    int aligned;
    int stride;
    void* x;
    void* y;
    void* z;
    void* result;
    kernel_type_t kernel;
    implementation_t implementation;
    int run_id;
    const char* compiler_flags;
} experiment_data_t;

void run_kernel(experiment_data_t* data, perf_metrics_t* metrics) {
    uint64_t operations = 0;
    size_t bytes_accessed = 0;
    
    // Calculate operations and bytes accessed based on kernel type
    switch (data->kernel) {
        case KERNEL_AXPY:
            operations = data->n * 2; // 2 FLOPs per element
            bytes_accessed = data->n * 3 * ((data->data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64));
            break;
        case KERNEL_DOT_PRODUCT:
            operations = data->n * 2; // 2 FLOPs per element
            bytes_accessed = data->n * 2 * ((data->data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64));
            break;
        case KERNEL_ELEMENTWISE_MULTIPLY:
            operations = data->n * 1; // 1 FLOP per element
            bytes_accessed = data->n * 3 * ((data->data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64));
            break;
    }
    
    perf_init(metrics, operations * ITERATIONS, data->n, global_collector.cpu_freq_ghz);
    
    // Warm-up
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        switch (data->kernel) {
            case KERNEL_AXPY:
                if (data->data_type == TYPE_F32) {
                    if (data->implementation == IMPL_SCALAR)
                        axpy_scalar_f32(2.0f, data->x, data->y, data->n);
                    else
                        axpy_vectorized_f32(2.0f, data->x, data->y, data->n);
                } else {
                    if (data->implementation == IMPL_SCALAR)
                        axpy_scalar_f64(2.0, data->x, data->y, data->n);
                    else
                        axpy_vectorized_f64(2.0, data->x, data->y, data->n);
                }
                break;
                
            case KERNEL_DOT_PRODUCT:
                if (data->data_type == TYPE_F32) {
                    if (data->implementation == IMPL_SCALAR)
                        dot_product_scalar_f32(data->x, data->y, data->n, data->result);
                    else
                        dot_product_vectorized_f32(data->x, data->y, data->n, data->result);
                } else {
                    if (data->implementation == IMPL_SCALAR)
                        dot_product_scalar_f64(data->x, data->y, data->n, data->result);
                    else
                        dot_product_vectorized_f64(data->x, data->y, data->n, data->result);
                }
                break;
                
            case KERNEL_ELEMENTWISE_MULTIPLY:
                if (data->data_type == TYPE_F32) {
                    if (data->implementation == IMPL_SCALAR)
                        elementwise_multiply_scalar_f32(data->x, data->y, data->z, data->n);
                    else
                        elementwise_multiply_vectorized_f32(data->x, data->y, data->z, data->n);
                } else {
                    if (data->implementation == IMPL_SCALAR)
                        elementwise_multiply_scalar_f64(data->x, data->y, data->z, data->n);
                    else
                        elementwise_multiply_vectorized_f64(data->x, data->y, data->z, data->n);
                }
                break;
        }
    }
    
    // Actual measurement
    double start = get_time();
    for (int i = 0; i < ITERATIONS; i++) {
        switch (data->kernel) {
            case KERNEL_AXPY:
                if (data->data_type == TYPE_F32) {
                    if (data->implementation == IMPL_SCALAR)
                        axpy_scalar_f32(2.0f, data->x, data->y, data->n);
                    else
                        axpy_vectorized_f32(2.0f, data->x, data->y, data->n);
                } else {
                    if (data->implementation == IMPL_SCALAR)
                        axpy_scalar_f64(2.0, data->x, data->y, data->n);
                    else
                        axpy_vectorized_f64(2.0, data->x, data->y, data->n);
                }
                break;
                
            case KERNEL_DOT_PRODUCT:
                if (data->data_type == TYPE_F32) {
                    if (data->implementation == IMPL_SCALAR)
                        dot_product_scalar_f32(data->x, data->y, data->n, data->result);
                    else
                        dot_product_vectorized_f32(data->x, data->y, data->n, data->result);
                } else {
                    if (data->implementation == IMPL_SCALAR)
                        dot_product_scalar_f64(data->x, data->y, data->n, data->result);
                    else
                        dot_product_vectorized_f64(data->x, data->y, data->n, data->result);
                }
                break;
                
            case KERNEL_ELEMENTWISE_MULTIPLY:
                if (data->data_type == TYPE_F32) {
                    if (data->implementation == IMPL_SCALAR)
                        elementwise_multiply_scalar_f32(data->x, data->y, data->z, data->n);
                    else
                        elementwise_multiply_vectorized_f32(data->x, data->y, data->z, data->n);
                } else {
                    if (data->implementation == IMPL_SCALAR)
                        elementwise_multiply_scalar_f64(data->x, data->y, data->z, data->n);
                    else
                        elementwise_multiply_vectorized_f64(data->x, data->y, data->z, data->n);
                }
                break;
        }
    }
    double end = get_time();
    
    metrics->time_seconds = (end - start) / ITERATIONS;
    metrics->gflops = (operations * data->n / 1e9) / metrics->time_seconds;
    metrics->cpe = (metrics->time_seconds * global_collector.cpu_freq_ghz * 1e9) / data->n;
    metrics->bandwidth_gbs = (bytes_accessed / 1e9) / metrics->time_seconds;
}

void run_experiment(experiment_data_t* data) {
    perf_metrics_t metrics;
    
    run_kernel(data, &metrics);
    
    // Store result
    result_t result = {
        .kernel = data->kernel,
        .implementation = data->implementation,
        .data_type = data->data_type,
        .array_size = data->n,
        .aligned = data->aligned,
        .stride = data->stride,
        .time_seconds = metrics.time_seconds,
        .gflops = metrics.gflops,
        .cpe = metrics.cpe,
        .bandwidth_gbs = metrics.bandwidth_gbs,
        .speedup = 0.0,
        .run_id = data->run_id
    };
    
    strncpy(result.compiler_flags, data->compiler_flags, sizeof(result.compiler_flags) - 1);
    result.compiler_flags[sizeof(result.compiler_flags) - 1] = 0;
    
    results_add(&global_collector, result);
    
    const char* kernel_str;
    switch (data->kernel) {
        case KERNEL_AXPY: kernel_str = "AXPY"; break;
        case KERNEL_DOT_PRODUCT: kernel_str = "Dot Product"; break;
        case KERNEL_ELEMENTWISE_MULTIPLY: kernel_str = "Elementwise Multiply"; break;
        //case KERNEL_STENCIL_3POINT: kernel_str = "3-Point Stencil"; break;
        //case KERNEL_MEMORY_BANDWIDTH: kernel_str = "Memory Bandwidth"; break;
        default: kernel_str = "Unknown";
    }
    
    printf("Run %d: %s %s %s size=%zu time=%.6fs GFLOP/s=%.3f BW=%.3f GB/s\n",
           data->run_id,
           (data->implementation == IMPL_SCALAR) ? "Scalar" : "Vectorized",
           kernel_str,
           (data->data_type == TYPE_F32) ? "f32" : "f64",
           data->n, metrics.time_seconds, metrics.gflops, metrics.bandwidth_gbs);
}

void run_axpy_experiment(results_collector_t* collector, size_t n, data_type_t data_type, int aligned, int stride, 
                         implementation_t impl, int run_id, const char* compiler_flags) {
    experiment_data_t data;
    data.n = n;
    data.data_type = data_type;
    data.aligned = aligned;
    data.stride = stride;
    data.kernel = KERNEL_AXPY;
    data.implementation = impl;
    data.run_id = run_id;
    data.compiler_flags = compiler_flags;
    
    // Allocate memory
    size_t elem_size = (data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64);
    size_t alloc_size = n * elem_size;
    
    if (aligned) {
        data.x = aligned_malloc(alloc_size, ALIGNMENT);
        data.y = aligned_malloc(alloc_size, ALIGNMENT);
    } else {
        data.x = malloc(alloc_size);
        data.y = malloc(alloc_size);
    }
    
    // Initialize data
    initialize_data(data.x, n, data_type, run_id);
    initialize_data(data.y, n, data_type, run_id + 1);
    
    run_experiment(&data);
    
    // Cleanup
    if (aligned) {
        aligned_free(data.x);
        aligned_free(data.y);
    } else {
        free(data.x);
        free(data.y);
    }
}

void run_dot_product_experiment(results_collector_t* collector, size_t n, data_type_t data_type, int aligned, int stride,
                               implementation_t impl, int run_id, const char* compiler_flags) {
    experiment_data_t data;
    data.n = n;
    data.data_type = data_type;
    data.aligned = aligned;
    data.stride = stride;
    data.kernel = KERNEL_DOT_PRODUCT;
    data.implementation = impl;
    data.run_id = run_id;
    data.compiler_flags = compiler_flags;
    
    // Allocate memory
    size_t elem_size = (data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64);
    size_t alloc_size = n * elem_size;
    
    if (aligned) {
        data.x = aligned_malloc(alloc_size, ALIGNMENT);
        data.y = aligned_malloc(alloc_size, ALIGNMENT);
        data.result = aligned_malloc(elem_size, ALIGNMENT);
    } else {
        data.x = malloc(alloc_size);
        data.y = malloc(alloc_size);
        data.result = malloc(elem_size);
    }
    
    // Initialize data
    initialize_data(data.x, n, data_type, run_id);
    initialize_data(data.y, n, data_type, run_id + 1);
    
    run_experiment(&data);
    
    // Cleanup
    if (aligned) {
        aligned_free(data.x);
        aligned_free(data.y);
        aligned_free(data.result);
    } else {
        free(data.x);
        free(data.y);
        free(data.result);
    }
}

void run_elementwise_multiply_experiment(results_collector_t* collector, size_t n, data_type_t data_type, int aligned, int stride,
                                        implementation_t impl, int run_id, const char* compiler_flags) {
    experiment_data_t data;
    data.n = n;
    data.data_type = data_type;
    data.aligned = aligned;
    data.stride = stride;
    data.kernel = KERNEL_ELEMENTWISE_MULTIPLY;
    data.implementation = impl;
    data.run_id = run_id;
    data.compiler_flags = compiler_flags;
    
    // Allocate memory
    size_t elem_size = (data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64);
    size_t alloc_size = n * elem_size;
    
    if (aligned) {
        data.x = aligned_malloc(alloc_size, ALIGNMENT);
        data.y = aligned_malloc(alloc_size, ALIGNMENT);
        data.z = aligned_malloc(alloc_size, ALIGNMENT);
    } else {
        data.x = malloc(alloc_size);
        data.y = malloc(alloc_size);
        data.z = malloc(alloc_size);
    }
    
    // Initialize data
    initialize_data(data.x, n, data_type, run_id);
    initialize_data(data.y, n, data_type, run_id + 1);
    
    run_experiment(&data);
    
    // Cleanup
    if (aligned) {
        aligned_free(data.x);
        aligned_free(data.y);
        aligned_free(data.z);
    } else {
        free(data.x);
        free(data.y);
        free(data.z);
    }
}

/*void run_stencil_experiment(size_t n, data_type_t data_type, int aligned, int stride,
                           implementation_t impl, int run_id, const char* compiler_flags) {
    experiment_data_t data;
    data.n = n;
    data.data_type = data_type;
    data.aligned = aligned;
    data.stride = stride;
    data.kernel = KERNEL_STENCIL_3POINT;
    data.implementation = impl;
    data.run_id = run_id;
    data.compiler_flags = compiler_flags;
    
    // Allocate memory
    size_t elem_size = (data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64);
    size_t alloc_size = n * elem_size;
    
    if (aligned) {
        data.x = aligned_malloc(alloc_size, ALIGNMENT);
        data.y = aligned_malloc(alloc_size, ALIGNMENT);
    } else {
        data.x = malloc(alloc_size);
        data.y = malloc(alloc_size);
    }
    
    // Initialize data
    initialize_data(data.x, n, data_type, run_id);
    
    run_experiment(&data);
    
    // Cleanup
    if (aligned) {
        aligned_free(data.x);
        aligned_free(data.y);
    } else {
        free(data.x);
        free(data.y);
    }
}

void run_memory_bandwidth_experiment(size_t n, data_type_t data_type, int aligned, int stride,
                                    implementation_t impl, int run_id, const char* compiler_flags) {
    experiment_data_t data;
    data.n = n;
    data.data_type = data_type;
    data.aligned = aligned;
    data.stride = stride;
    data.kernel = KERNEL_MEMORY_BANDWIDTH;
    data.implementation = impl;
    data.run_id = run_id;
    data.compiler_flags = compiler_flags;
    
    // Allocate memory
    size_t elem_size = (data_type == TYPE_F32) ? sizeof(f32) : sizeof(f64);
    size_t alloc_size = n * elem_size;
    
    if (aligned) {
        data.x = aligned_malloc(alloc_size, ALIGNMENT);
        data.y = aligned_malloc(alloc_size, ALIGNMENT);
    } else {
        data.x = malloc(alloc_size);
        data.y = malloc(alloc_size);
    }
    
    // Initialize data
    initialize_data(data.x, n, data_type, run_id);
    
    run_experiment(&data);
    
    // Cleanup
    if (aligned) {
        aligned_free(data.x);
        aligned_free(data.y);
    } else {
        free(data.x);
        free(data.y);
    }
}*/

// ==================== CPU AFFINITY AND PRIORITY FUNCTIONS ====================

void set_cpu_affinity(int core_id) {
    DWORD_PTR process_affinity_mask = 0;
    DWORD_PTR system_affinity_mask = 0;
    
    // Get current process handle
    HANDLE current_process = GetCurrentProcess();
    
    // Get number of processors
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    int num_processors = sys_info.dwNumberOfProcessors;
    
    printf("System has %d processors\n", num_processors);
    
    // Validate core_id
    if (core_id < 0 || core_id >= num_processors) {
        printf("Warning: Core ID %d is invalid. Using core 0 instead.\n", core_id);
        core_id = 0;
    }
    
    // Create affinity mask for the specified core
    DWORD_PTR affinity_mask = (DWORD_PTR)1 << core_id;
    
    // Set process affinity
    if (SetProcessAffinityMask(current_process, affinity_mask)) {
        printf("Successfully pinned process to core %d\n", core_id);
    } else {
        DWORD error = GetLastError();
        printf("Warning: Failed to set CPU affinity (Error %lu). Continuing without affinity.\n", error);
    }
    
    // Verify the affinity was set
    if (GetProcessAffinityMask(current_process, &process_affinity_mask, &system_affinity_mask)) {
        printf("Process affinity mask: 0x%lx\n", process_affinity_mask);
        printf("System affinity mask: 0x%lx\n", system_affinity_mask);
    }
}

void set_thread_affinity(int core_id) {
    // Get number of processors
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    int num_processors = sys_info.dwNumberOfProcessors;
    
    // Validate core_id
    if (core_id < 0 || core_id >= num_processors) {
        printf("Warning: Core ID %d is invalid. Using core 0 instead.\n", core_id);
        core_id = 0;
    }
    
    // Create affinity mask for the specified core
    DWORD_PTR affinity_mask = (DWORD_PTR)1 << core_id;
    
    // Set thread affinity
    HANDLE current_thread = GetCurrentThread();
    if (SetThreadAffinityMask(current_thread, affinity_mask)) {
        printf("Successfully pinned thread to core %d\n", core_id);
    } else {
        DWORD error = GetLastError();
        printf("Warning: Failed to set thread affinity (Error %lu). Continuing without affinity.\n", error);
    }
}

void set_high_priority() {
    HANDLE current_process = GetCurrentProcess();
    
    // Set high priority class
    if (SetPriorityClass(current_process, HIGH_PRIORITY_CLASS)) {
        printf("Set process priority to HIGH\n");
    } else {
        DWORD error = GetLastError();
        printf("Warning: Failed to set high priority (Error %lu)\n", error);
    }
    
    // Also set thread priority
    HANDLE current_thread = GetCurrentThread();
    if (SetThreadPriority(current_thread, THREAD_PRIORITY_HIGHEST)) {
        printf("Set thread priority to HIGHEST\n");
    } else {
        DWORD error = GetLastError();
        printf("Warning: Failed to set thread priority (Error %lu)\n", error);
    }
}

void print_cpu_info() {
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    
    printf("CPU Information:\n");
    printf("  Number of processors: %lu\n", sys_info.dwNumberOfProcessors);
    printf("  Processor architecture: %d\n", sys_info.wProcessorArchitecture);
    printf("  Page size: %lu bytes\n", sys_info.dwPageSize);
    
    // Get detailed CPU information using CPUID
    int cpuInfo[4] = {0};
    char cpuBrandString[0x40] = {0};
    
    __cpuid(cpuInfo, 0x80000000);
    if ((unsigned)cpuInfo[0] >= 0x80000004) {
        __cpuid(cpuInfo, 0x80000002);
        memcpy(cpuBrandString, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000003);
        memcpy(cpuBrandString + 16, cpuInfo, sizeof(cpuInfo));
        __cpuid(cpuInfo, 0x80000004);
        memcpy(cpuBrandString + 32, cpuInfo, sizeof(cpuInfo));
        
        // Clean up the brand string
        for (int i = 0; i < sizeof(cpuBrandString); i++) {
            if (cpuBrandString[i] == 0) break;
            if (cpuBrandString[i] < 32 || cpuBrandString[i] > 126) {
                cpuBrandString[i] = ' ';
            }
        }
        
        printf("  CPU: %s\n", cpuBrandString);
    }
}

void disable_smt() {
    // This is a best-effort attempt to disable SMT (Simultaneous Multi-Threading)
    // by using only physical cores
    
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    int num_processors = sys_info.dwNumberOfProcessors;
    
    // Typically, physical cores are the first half of the logical processors
    int physical_cores = num_processors / 2;
    
    if (physical_cores > 0) {
        printf("Disabling SMT - using only %d physical cores\n", physical_cores);
        
        DWORD_PTR affinity_mask = 0;
        for (int i = 0; i < physical_cores; i++) {
            affinity_mask |= (DWORD_PTR)1 << i;
        }
        
        HANDLE current_process = GetCurrentProcess();
        if (SetProcessAffinityMask(current_process, affinity_mask)) {
            printf("Using physical cores only (mask: 0x%lx)\n", affinity_mask);
        }
    }
}

void fix_cpu_frequency() {
    // This is a best-effort attempt to fix CPU frequency
    // Note: This requires administrator privileges and may not work on all systems
    
    printf("Attempting to set high performance power plan...\n");
    
    // Try to set high performance power plan
    // This is a simplified approach - in practice, you might want to use powercfg
    SYSTEM_POWER_STATUS power_status;
    if (GetSystemPowerStatus(&power_status)) {
        if (power_status.ACLineStatus == 1) {
            printf("Running on AC power - good for consistent performance\n");
        } else {
            printf("Running on battery - performance may vary\n");
        }
    }
    
    printf("For best results, manually set Windows power plan to 'High Performance'\n");
    printf("and ensure CPU frequency scaling is disabled in BIOS if possible.\n");
}

// ==================== EXPERIMENT SETS ====================

void run_comprehensive_experiments(results_collector_t* collector, const char* compiler_flags) {
    printf("Running comprehensive experiments with flags: %s\n", compiler_flags);
    
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    int run_id = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
            size_t n = sizes[size_idx];
            
            for (int data_type = TYPE_F32; data_type <= TYPE_F64; data_type++) {
                for (int aligned = 0; aligned <= 1; aligned++) {
                    for (int impl = IMPL_SCALAR; impl <= IMPL_VECTORIZED; impl++) {
                        // Test all kernels
                        run_axpy_experiment(collector, n, data_type, aligned, 1, impl, run_id++, compiler_flags);
                        run_dot_product_experiment(collector, n, data_type, aligned, 1, impl, run_id++, compiler_flags);
                        run_elementwise_multiply_experiment(collector, n, data_type, aligned, 1, impl, run_id++, compiler_flags);
                    }
                }
            }
        }
    }
}

void run_locality_sweep(results_collector_t* collector, const char* compiler_flags) {
    printf("Running locality sweep with flags: %s\n", compiler_flags);
    
    // Sweep through sizes that cross cache boundaries
    size_t sizes[] = {
        1024,    // L1 cache
        4096,    // L1 cache
        16384,   // L1/L2 boundary
        32768,   // L2 cache
        65536,   // L2 cache
        131072,  // L2/L3 boundary
        262144,  // L3 cache
        524288,  // L3 cache
        1048576, // L3/DRAM boundary
        2097152, // DRAM
        4194304, // DRAM
        8388608, // DRAM
        16777216 // DRAM
    };
    
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int run_id = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
            size_t n = sizes[size_idx];
            
            // Test with different data types and implementations
            for (int data_type = TYPE_F32; data_type <= TYPE_F64; data_type++) {
                for (int impl = IMPL_SCALAR; impl <= IMPL_VECTORIZED; impl++) {
                    run_axpy_experiment(collector, n, data_type, 1, 1, impl, run_id++, compiler_flags);
                    run_dot_product_experiment(collector, n, data_type, 1, 1, impl, run_id++, compiler_flags);
                }
            }
        }
    }
}

void run_alignment_study(results_collector_t* collector, const char* compiler_flags) {
    printf("Running alignment study with flags: %s\n", compiler_flags);
    
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int run_id = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
            size_t n = sizes[size_idx];
            
            // Test with different alignments
            for (int aligned = 0; aligned <= 1; aligned++) {
                for (int data_type = TYPE_F32; data_type <= TYPE_F64; data_type++) {
                    for (int impl = IMPL_SCALAR; impl <= IMPL_VECTORIZED; impl++) {
                        run_axpy_experiment(collector, n, data_type, aligned, 1, impl, run_id++, compiler_flags);
                        run_dot_product_experiment(collector, n, data_type, aligned, 1, impl, run_id++, compiler_flags);
                    }
                }
            }
        }
    }
}

void run_stride_study(results_collector_t* collector, const char* compiler_flags) {
    printf("Running stride study with flags: %s\n", compiler_flags);
    
    size_t n = 1048576; // Fixed size for stride study
    int strides[] = {1, 2, 4, 8, 16};
    int num_strides = sizeof(strides) / sizeof(strides[0]);
    int run_id = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        for (int stride_idx = 0; stride_idx < num_strides; stride_idx++) {
            int stride = strides[stride_idx];
            
            for (int data_type = TYPE_F32; data_type <= TYPE_F64; data_type++) {
                for (int impl = IMPL_SCALAR; impl <= IMPL_VECTORIZED; impl++) {
                    run_axpy_experiment(collector, n, data_type, 1, stride, impl, run_id++, compiler_flags);
                    run_dot_product_experiment(collector, n, data_type, 1, stride, impl, run_id++, compiler_flags);
                    run_elementwise_multiply_experiment(collector, n, data_type, 1, stride, impl, run_id++, compiler_flags);
                }
            }
        }
    }
}

void run_data_type_study(results_collector_t* collector, const char* compiler_flags) {
    printf("Running data type study with flags: %s\n", compiler_flags);
    
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int run_id = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
            size_t n = sizes[size_idx];
            
            // Test with different data types
            for (int data_type = TYPE_F32; data_type <= TYPE_F64; data_type++) {
                for (int impl = IMPL_SCALAR; impl <= IMPL_VECTORIZED; impl++) {
                    run_axpy_experiment(collector, n, data_type, 1, 1, impl, run_id++, compiler_flags);
                    run_dot_product_experiment(collector, n, data_type, 1, 1, impl, run_id++, compiler_flags);
                    run_elementwise_multiply_experiment(collector, n, data_type, 1, 1, impl, run_id++, compiler_flags);
                }
            }
        }
    }
}

void results_flush(results_collector_t* collector) {
    if (collector->count == 0) return;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "simd_performance_%s.csv", 
             collector->compiler_version);
    
    // Use the updated results_save_csv function
    results_save_csv(collector, filename);
    
    // Reset buffer
    collector->count = 0;
}

void results_finalize(results_collector_t* collector) {
    // Flush any remaining results
    results_flush(collector);
    
    printf("Results collection finalized. Total measurements: %d\n", collector->count);
}

// ==================== MAIN FUNCTION ====================

int main(int argc, char* argv[]) {
    printf("SIMD Performance Analysis Project\n");
    printf("=================================\n\n");
    printf("%s", argv[0]);
    // CPU configuration
    print_cpu_info();
    
    // Set CPU affinity to core 0 (you can make this configurable)
    int target_core = 0;
    if (argc > 2) {
        target_core = atoi(argv[2]);
    }
    set_cpu_affinity(target_core);
    
    // Set high priority
    set_high_priority();
    
    // Optional: Disable SMT for more consistent results
    if (argc > 3 && strcmp(argv[3], "nosmt") == 0) {
        disable_smt();
    }
    
    // Optional: Attempt to fix CPU frequency
    if (argc > 4 && strcmp(argv[4], "fixfreq") == 0) {
        fix_cpu_frequency();
    }
    
    double cpu_freq_ghz = detect_cpu_frequency();
    printf("Detected CPU frequency: %.2f GHz\n", cpu_freq_ghz);
    
    char cpu_model[128];
    get_cpu_model(cpu_model, sizeof(cpu_model));
    printf("CPU Model: %s\n", cpu_model);
    
    char compiler_version[128];
    get_compiler_version(compiler_version, sizeof(compiler_version));
    printf("Compiler: %s\n", compiler_version);
    
    // Check command line arguments
    const char* compiler_flags = "default";
    if (argc > 1) {
        compiler_flags = argv[1];
    }
    
    // Initialize results collector
    //results_collector_t global_collector;
    results_init(&global_collector, cpu_freq_ghz, compiler_flags);
    
    // Run different experiment sets - FIXED: Now passing correct parameters
    run_comprehensive_experiments(&global_collector, compiler_flags);
    run_locality_sweep(&global_collector, compiler_flags);
    run_alignment_study(&global_collector, compiler_flags);
    run_stride_study(&global_collector, compiler_flags);
    run_data_type_study(&global_collector, compiler_flags);
    
    // Calculate speedups
    calculate_speedups(&global_collector);
    
   // Save final combined CSV
    char final_filename[256];
    snprintf(final_filename, sizeof(final_filename), "simd_performance_%s.csv", compiler_flags);
    results_save_csv(&global_collector, final_filename);
    
    // Finalize and save any remaining results
    results_finalize(&global_collector);

    printf("\nExperiment completed successfully!\n");
    
    // Restore normal priority (optional)
    HANDLE current_process = GetCurrentProcess();
    SetPriorityClass(current_process, NORMAL_PRIORITY_CLASS);
    
    return 0;
}


