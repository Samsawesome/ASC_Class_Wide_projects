#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>

// Configuration
#define MIN_SIZE 1024
#define MAX_SIZE (64 * 1024 * 1024)
#define ITERATIONS 100
#define WARMUP_ITERATIONS 10
#define ALIGNMENT 64
#define MAX_RESULTS 5000
#define NUM_RUNS 3  // For statistical reliability

// Data types
typedef float f32;
typedef double f64;
typedef int32_t i32;

// Kernel types
typedef enum {
    KERNEL_AXPY,
    KERNEL_DOT_PRODUCT,
    KERNEL_ELEMENTWISE_MULTIPLY/*,
    KERNEL_STENCIL_3POINT,
    KERNEL_MEMORY_BANDWIDTH*/
} kernel_type_t;

// Implementation types
typedef enum {
    IMPL_SCALAR,
    IMPL_VECTORIZED
} implementation_t;

// Data types
typedef enum {
    TYPE_F32,
    TYPE_F64,
    TYPE_I32
} data_type_t;

// Performance metrics
typedef struct {
    double time_seconds;
    double gflops;
    double cpe;
    double bandwidth_gbs;
    uint64_t operations;
    size_t elements;
    double cpu_freq_ghz;
} perf_metrics_t;

// Result structure
typedef struct {
    kernel_type_t kernel;
    implementation_t implementation;
    data_type_t data_type;
    size_t array_size;
    int aligned;
    int stride;
    double time_seconds;
    double gflops;
    double cpe;
    double bandwidth_gbs;
    double speedup;
    int run_id;
    char compiler_flags[256];
} result_t;

// Global results collection
typedef struct {
    result_t results[MAX_RESULTS];
    int count;
    int buffer_overflow;
    double cpu_freq_ghz;
    char cpu_model[128];
    char compiler_version[128];
} results_collector_t;

// Utility functions
double get_time();
void perf_init(perf_metrics_t* metrics, uint64_t operations, size_t elements, double cpu_freq_ghz);
double detect_cpu_frequency();
void get_cpu_model(char* buffer, size_t size);
void get_compiler_version(char* buffer, size_t size);
void* aligned_malloc(size_t size, size_t alignment);
void aligned_free(void* ptr);
void initialize_data(void* data, size_t n, data_type_t type, int pattern);

// Results collection functions
void results_init(results_collector_t* collector, double cpu_freq_ghz, const char* compiler_flags);
void results_add(results_collector_t* collector, result_t result);
void results_save_csv(const results_collector_t* collector, const char* filename);
void results_print_summary(const results_collector_t* collector);
void calculate_speedups(results_collector_t* collector);

// Kernel functions
void run_axpy_experiment(size_t n, data_type_t data_type, int aligned, int stride, 
                         implementation_t impl, int run_id, const char* compiler_flags);
void run_dot_product_experiment(size_t n, data_type_t data_type, int aligned, int stride,
                               implementation_t impl, int run_id, const char* compiler_flags);
void run_elementwise_multiply_experiment(size_t n, data_type_t data_type, int aligned, int stride,
                                        implementation_t impl, int run_id, const char* compiler_flags);
/*void run_stencil_experiment(size_t n, data_type_t data_type, int aligned, int stride,
                           implementation_t impl, int run_id, const char* compiler_flags);
void run_memory_bandwidth_experiment(size_t n, data_type_t data_type, int aligned, int stride,
                                    implementation_t impl, int run_id, const char* compiler_flags);*/

// Experiment runners
void run_comprehensive_experiments(const char* compiler_flags);
void run_locality_sweep(const char* compiler_flags);
void run_alignment_study(const char* compiler_flags);
void run_stride_study(const char* compiler_flags);
void run_data_type_study(const char* compiler_flags);

// Core pinning
void set_cpu_affinity(int core_id);
void print_cpu_info();
void set_high_priority();