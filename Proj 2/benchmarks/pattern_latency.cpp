#include "../include/memory_utils.h"
#include <windows.h> 
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <malloc.h>  // Add for _aligned_malloc
#include <emmintrin.h>  // Add for _mm_mfence
#include <functional>

// Set process affinity to core 0 and highest priority
void set_high_priority_affinity() {
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    
    SetPriorityClass(process, HIGH_PRIORITY_CLASS);
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
    
    DWORD_PTR affinityMask = 1;
    SetThreadAffinityMask(thread, affinityMask);
    SetThreadIdealProcessor(thread, 0);
    
    PROCESS_POWER_THROTTLING_STATE powerThrottling = {};
    powerThrottling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    powerThrottling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    powerThrottling.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    
    SetProcessInformation(process, ProcessPowerThrottling, 
                         &powerThrottling, sizeof(powerThrottling));
}

class HighResTimer {
private:
    LARGE_INTEGER frequency;
    LARGE_INTEGER startTime;
    
public:
    HighResTimer() {
        QueryPerformanceFrequency(&frequency);
    }
    
    void start() {
        QueryPerformanceCounter(&startTime);
    }
    
    double stop() {
        LARGE_INTEGER endTime;
        QueryPerformanceCounter(&endTime);
        return (endTime.QuadPart - startTime.QuadPart) * 1000000000.0 / frequency.QuadPart; // ns
    }
};

// Windows-compatible aligned_alloc and aligned_free
void* windows_aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

void windows_aligned_free(void* ptr) {
    _aligned_free(ptr);
}

constexpr int ITERATIONS = 1000000;
constexpr int WARMUP = 1000;

// Generate access patterns
std::vector<size_t> generate_sequential(size_t size, int stride) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < size; i += stride) {
        indices.push_back(i);
    }
    return indices;
}

std::vector<size_t> generate_random(size_t size, int stride) {
    auto indices = generate_sequential(size, stride);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

std::vector<size_t> generate_strided(size_t size, int stride, int pattern_stride) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < size; i += stride) {
        indices.push_back((i * pattern_stride) % size);
    }
    return indices;
}

std::vector<size_t> generate_cluster(size_t size, int stride, int cluster_size) {
    std::vector<size_t> indices;
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<size_t> dist(0, size - cluster_size * stride);
    
    for (int cluster = 0; cluster < size / (cluster_size * stride); ++cluster) {
        size_t base = dist(g);
        for (int i = 0; i < cluster_size; ++i) {
            indices.push_back((base + i * stride) % size);
        }
    }
    return indices;
}

std::vector<double> measure_pattern_latency_single_run(char* memory, const std::vector<size_t>& pattern) {
    HighResTimer timer;
    volatile char result = 0;
    
    // Warm-up
    for (int i = 0; i < WARMUP && i < (int)pattern.size(); ++i) {
        result += memory[pattern[i]];
    }
    
    flush_cache();
    
    timer.start();
    for (int i = 0; i < ITERATIONS; ++i) {
        size_t index = pattern[i % pattern.size()];
        result += memory[index];
        _mm_mfence();
    }
    double total_ns = timer.stop();
    
    if (result == 0) {
        std::cout << "Never happens" << std::endl;
    }
    
    return {total_ns / ITERATIONS};
}

void test_access_pattern(const std::string& pattern_name, const std::vector<size_t>& pattern, 
                        char* memory, size_t size) {
    const int NUM_RUNS = 3;
    std::vector<double> latencies;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        auto run_latencies = measure_pattern_latency_single_run(memory, pattern);
        latencies.insert(latencies.end(), run_latencies.begin(), run_latencies.end());
        flush_cache();
    }
    
    // Output all three measurements
    std::cout << "Pattern: " << std::setw(12) << pattern_name 
              << ", Size: " << std::setw(8) << size << " bytes, Latencies: ";
    for (size_t i = 0; i < latencies.size(); ++i) {
        std::cout << latencies[i] << " ns";
        if (i < latencies.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

int main() {
    set_high_priority_affinity();
    
    const size_t TEST_SIZE = 64 * 1024 * 1024; // 64MB
    char* memory = (char*)windows_aligned_alloc(CACHE_LINE_SIZE, TEST_SIZE);
    
    if (!memory) {
        std::cerr << "Failed to allocate memory!" << std::endl;
        return 1;
    }
    
    // Initialize memory
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        memory[i] = (char)(i % 256);
    }
    
    // Define test configurations
    struct PatternTest {
        std::string name;
        std::function<std::vector<size_t>(size_t, int)> generator;
        int param;
    };
    
    std::vector<PatternTest> tests = {
        {"Sequential", generate_sequential, 64},
        {"Random", generate_random, 64},
        {"Stride-2", [](size_t s, int st) { return generate_strided(s, st, 2); }, 64},
        {"Stride-4", [](size_t s, int st) { return generate_strided(s, st, 4); }, 64},
        {"Stride-8", [](size_t s, int st) { return generate_strided(s, st, 8); }, 64},
        {"Cluster-4", [](size_t s, int st) { return generate_cluster(s, st, 4); }, 64},
        {"Cluster-16", [](size_t s, int st) { return generate_cluster(s, st, 16); }, 64}
    };
    
    // Randomize test order
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(tests.begin(), tests.end(), g);
    
    std::cout << "=== Memory Access Pattern Latency Tests ===" << std::endl;
    
    for (const auto& test : tests) {
        auto pattern = test.generator(TEST_SIZE, test.param);
        test_access_pattern(test.name, pattern, memory, TEST_SIZE);
    }
    
    windows_aligned_free(memory);
    return 0;
}