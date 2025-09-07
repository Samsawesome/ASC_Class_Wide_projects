#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>

#define N 1000000
#define ITERATIONS 100

// ==================== WINDOWS HIGH-RESOLUTION TIMING ====================
double get_time() {
    LARGE_INTEGER frequency, time;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / frequency.QuadPart;
}

// ==================== PERFORMANCE METRICS STRUCTURE ====================
typedef struct {
    double time_seconds;
    double gflops;
    double cpe; // Cycles per element
    uint64_t operations;
    size_t elements;
    double cpu_freq_ghz;
    
    // Performance counters
    uint64_t instructions_retired;
    uint64_t vector_instructions;
    uint64_t cache_misses;
    uint64_t branch_misses;
    uint64_t mem_loads;
    uint64_t mem_stores;
    double memory_bandwidth_gbs;
    double memory_traffic_gb;
    int hw_counters_available;
} perf_metrics_t;

// ==================== WINDOWS PERFORMANCE COUNTER COLLECTION ====================
int run_with_windows_perf_counters(const char *program_path, perf_metrics_t *metrics) {
    // Windows doesn't have perf like Linux, so we'll use different approaches
    
    printf("Windows performance monitoring - using analytical estimates and basic timing\n");
    
    // For Windows, we'll use analytical estimates since hardware counters are complex to access
    metrics->hw_counters_available = 0;
    
    // Analytical estimates based on common CPU characteristics
    metrics->instructions_retired = metrics->operations * 8; // ~8 instructions per FLOP
    metrics->vector_instructions = metrics->instructions_retired * 0.6; // Assume 60% vectorization
    
    // Cache miss estimation based on working set size
    if (metrics->elements * sizeof(double) <= 32 * 1024) { // L1 cache
        metrics->cache_misses = metrics->elements * 0.01; // 1% miss rate
    } else if (metrics->elements * sizeof(double) <= 256 * 1024) { // L2 cache
        metrics->cache_misses = metrics->elements * 0.05; // 5% miss rate
    } else if (metrics->elements * sizeof(double) <= 8 * 1024 * 1024) { // L3 cache
        metrics->cache_misses = metrics->elements * 0.1; // 10% miss rate
    } else { // RAM
        metrics->cache_misses = metrics->elements * 0.2; // 20% miss rate
    }
    
    metrics->branch_misses = metrics->instructions_retired * 0.02; // 2% branch misprediction
    
    // Memory traffic estimation
    size_t bytes_transferred = 3 * metrics->elements * sizeof(double) * ITERATIONS;
    metrics->memory_traffic_gb = bytes_transferred / 1e9;
    metrics->memory_bandwidth_gbs = metrics->memory_traffic_gb / metrics->time_seconds;
    
    return 0;
}

// ==================== ALTERNATIVE: USE WINDOWS PERFORMANCE API ====================
int get_cpu_usage_counters(perf_metrics_t *metrics) {
    // This is a simplified approach - Windows performance counters are complex
    static PDH_HQUERY cpu_query;
    static PDH_HCOUNTER cpu_counter;
    static int initialized = 0;
    
    if (!initialized) {
        PdhOpenQuery(NULL, 0, &cpu_query);
        PdhAddEnglishCounter(cpu_query, "\\Processor(_Total)\\% Processor Time", 0, &cpu_counter);
        PdhCollectQueryData(cpu_query);
        initialized = 1;
    }
    
    PDH_FMT_COUNTERVALUE counter_value;
    PdhCollectQueryData(cpu_query);
    PdhGetFormattedCounterValue(cpu_counter, PDH_FMT_DOUBLE, NULL, &counter_value);
    
    // This gives CPU usage %, not detailed hardware counters
    printf("CPU Usage: %.1f%%\n", counter_value.doubleValue);
    
    return 0;
}

// ==================== PERFORMANCE MEASUREMENT SYSTEM ====================
void perf_init(perf_metrics_t *metrics, uint64_t operations, size_t elements, double cpu_freq_ghz) {
    metrics->operations = operations;
    metrics->elements = elements;
    metrics->cpu_freq_ghz = cpu_freq_ghz;
    metrics->time_seconds = 0.0;
    metrics->gflops = 0.0;
    metrics->cpe = 0.0;
    metrics->instructions_retired = 0;
    metrics->vector_instructions = 0;
    metrics->cache_misses = 0;
    metrics->branch_misses = 0;
    metrics->mem_loads = 0;
    metrics->mem_stores = 0;
    metrics->memory_bandwidth_gbs = 0.0;
    metrics->memory_traffic_gb = 0.0;
    metrics->hw_counters_available = 0;
}

void perf_measure(perf_metrics_t *metrics, void (*func)(void*), void *data) {
    double start = get_time();
    func(data);
    double end = get_time();
    
    metrics->time_seconds = end - start;
    metrics->gflops = (metrics->operations / 1e9) / metrics->time_seconds;
    metrics->cpe = (metrics->time_seconds * metrics->cpu_freq_ghz * 1e9) / metrics->elements;
}

void perf_print(const perf_metrics_t *metrics, const char *label) {
    printf("=== %s ===\n", label);
    printf("Time:      %.6f seconds\n", metrics->time_seconds);
    printf("GFLOP/s:   %.3f\n", metrics->gflops);
    printf("CPE:       %.3f cycles/element\n", metrics->cpe);
    printf("Elements:  %zu\n", metrics->elements);
    printf("Operations: %lu\n", metrics->operations);
    
    printf("\n--- PERFORMANCE ANALYSIS (Windows) ---\n");
    
    if (metrics->hw_counters_available) {
        printf("Hardware counters: AVAILABLE\n");
        printf("Instructions retired:    %'lu\n", metrics->instructions_retired);
        printf("IPC:                     %.2f instructions/cycle\n",
               metrics->instructions_retired / (metrics->time_seconds * metrics->cpu_freq_ghz * 1e9));
        
        if (metrics->vector_instructions > 0) {
            printf("Vector instructions:     %'lu (%.1f%% of total)\n", 
                   metrics->vector_instructions,
                   (metrics->vector_instructions * 100.0) / metrics->instructions_retired);
        }
        
        printf("Cache misses:            %'lu\n", metrics->cache_misses);
        printf("Branch misses:           %'lu\n", metrics->branch_misses);
    } else {
        printf("Hardware counters: Using analytical estimates\n");
        printf("Instructions (estimated): %'lu\n", metrics->instructions_retired);
        printf("Vector instr (estimated): %'lu (%.1f%% of total)\n", 
               metrics->vector_instructions,
               (metrics->vector_instructions * 100.0) / metrics->instructions_retired);
        printf("IPC (estimated):         %.2f instructions/cycle\n",
               metrics->instructions_retired / (metrics->time_seconds * metrics->cpu_freq_ghz * 1e9));
        printf("Cache misses (est):      %'lu\n", metrics->cache_misses);
        printf("Branch misses (est):     %'lu\n", metrics->branch_misses);
    }
    
    printf("Memory traffic:          %.3f GB\n", metrics->memory_traffic_gb);
    printf("Memory bandwidth:        %.2f GB/s\n", metrics->memory_bandwidth_gbs);
    
    // Calculate arithmetic intensity
    double ai = (double)metrics->operations / (metrics->memory_traffic_gb * 1e9);
    printf("Arithmetic intensity:    %.2f FLOPs/byte\n", ai);
    
    // Performance characterization
    if (ai < 1.0) {
        printf("Performance:            MEMORY-BOUND\n");
    } else if (ai < 4.0) {
        printf("Performance:            BALANCED\n");
    } else {
        printf("Performance:            COMPUTE-BOUND\n");
    }
    
    printf("\n");
}

// ==================== VECTOR OPERATIONS EXAMPLE ====================
typedef struct {
    double a;
    double *x;
    double *y;
    size_t n;
    int iterations;
} axpy_data_t;

void axpy_operation(double a, double *x, double *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_wrapper(void *data) {
    axpy_data_t *d = (axpy_data_t*)data;
    for (int iter = 0; iter < d->iterations; iter++) {
        axpy_operation(d->a, d->x, d->y, d->n);
    }
}

// ==================== DETECT CPU INFORMATION ====================
double detect_cpu_frequency() {
    // Try to get CPU frequency from Windows registry
    HKEY hKey;
    DWORD frequency = 0;
    DWORD dataSize = sizeof(frequency);
    
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        if (RegQueryValueExA(hKey, "~MHz", NULL, NULL, (LPBYTE)&frequency, &dataSize) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return frequency / 1000.0; // Convert MHz to GHz
        }
        RegCloseKey(hKey);
    }
    
    // Fallback: use a reasonable default
    return 2.5; // 2.5 GHz
}

// ==================== MAIN FUNCTION ====================
int main() {
    // Detect CPU frequency
    double cpu_freq_ghz = detect_cpu_frequency();
    printf("Detected CPU frequency: %.2f GHz\n", cpu_freq_ghz);
    
    // Allocate memory
    double *x = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize data
    for (size_t i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
    }
    
    // Setup performance measurement
    axpy_data_t data = {2.0, x, y, N, ITERATIONS};
    uint64_t operations_per_iteration = 2 * N; // 2 FLOPs per element
    uint64_t total_operations = operations_per_iteration * ITERATIONS;
    
    perf_metrics_t metrics;
    perf_init(&metrics, total_operations, N, cpu_freq_ghz);
    
    // Warm-up run
    printf("Performing warm-up run...\n");
    axpy_wrapper(&data);
    
    // Measure performance
    printf("Measuring performance...\n");
    perf_measure(&metrics, axpy_wrapper, &data);
    
    // Use Windows performance monitoring
    printf("Collecting performance data...\n");
    run_with_windows_perf_counters(NULL, &metrics);
    
    // Optional: Get CPU usage counters
    get_cpu_usage_counters(&metrics);
    
    // Print results
    perf_print(&metrics, "AXPY Performance Analysis (Windows)");
    
    free(x);
    free(y);
    return 0;
}