#include "../include/memory_utils.h"
#include <windows.h>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <regex>
#include <filesystem>


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


// Use Windows high-performance counters for better resolution
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

constexpr int ITERATIONS = 1000000;  // Reduced for Windows timer limitations
constexpr int WARMUP = 1000;

// Force memory operations to not be optimized away
#pragma optimize("", off)
void memory_access(volatile char* ptr) {
    *ptr = *ptr + 1;
}
#pragma optimize("", on)

double measure_latency_improved(char* memory, size_t size, int access_stride) {
    HighResTimer timer;
    volatile char result = 0;
    
    std::vector<size_t> indices;
    for (size_t i = 0; i < size; i += access_stride) {
        indices.push_back(i % size);
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Warm-up phase with more iterations
    for (int i = 0; i < WARMUP * 10 && i < (int)indices.size(); ++i) {
        result += memory[indices[i]];
    }
    
    const int NUM_RUNS = 3;
    double total_latency = 0.0;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        flush_cache();
        
        timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            size_t index = indices[i % indices.size()];
            result += memory[index];
            _mm_mfence();
        }
        total_latency += timer.stop() / ITERATIONS;
    }
    
    if (result == 0) {
        std::cout << "Never happens" << std::endl;
    }
    
    return total_latency / NUM_RUNS;
}

void test_memory_level_improved(size_t size, const char* level_name) {
    const size_t buffer_size = size * 2; // Ensure larger than cache
    char* memory = (char*)aligned_alloc(64, buffer_size);
    
    if (!memory) {
        std::cerr << "Allocation failed for " << level_name << std::endl;
        return;
    }
    
    // Initialize memory
    for (size_t i = 0; i < buffer_size; ++i) {
        memory[i] = (char)(i % 256);
    }
    
    // Clear cache more effectively
    flush_cache();
    
    // Use appropriate stride for each level
    int stride = 64; // Cache line size
    if (size > 8 * 1024 * 1024) { // For DRAM, use larger stride
        stride = 4096;
    }
    
    double latency = measure_latency_improved(memory, buffer_size, stride);
    
    std::cout << "Level: " << level_name << ", Size: " << size << " bytes, Latency: " 
              << latency << " ns" << std::endl;
    
    aligned_free(memory);
}

struct LatencyResult {
    std::string level;
    double latency_ns;
    size_t size_bytes;
};

std::vector<LatencyResult> parse_mlc_output(const std::string& filename) {
    std::vector<LatencyResult> results;
    std::ifstream file(filename);
    std::string line;
    
    std::regex latency_pattern(R"((\w+)\s+latency\s*=\s*([0-9]+\.[0-9]+)\s*ns)");
    std::regex size_pattern(R"(Cache\s+size\s+:\s+([0-9]+)\s*([KM]B))");
    
    std::string current_level;
    double current_latency = 0.0;
    size_t current_size = 0;
    
    while (std::getline(file, line)) {
        std::smatch match;
        
        // Match latency lines
        if (std::regex_search(line, match, latency_pattern)) {
            current_level = match[1];
            current_latency = std::stod(match[2]);
            
            // Convert level name to size
            if (current_level == "L1") current_size = 32 * 1024;
            else if (current_level == "L2") current_size = 256 * 1024;
            else if (current_level == "L3") current_size = 8 * 1024 * 1024;
            else current_size = 64 * 1024 * 1024; // DRAM
            
            results.push_back({current_level, current_latency, current_size});
        }
    }
    
    return results;
}

std::wstring find_mlc_path() {
    std::vector<std::wstring> possible_paths = {
        L"..\\scripts\\mlc.exe",
        L"scripts\\mlc.exe", 
        L"mlc.exe",
        L"..\\mlc.exe",
        L"..\\..\\scripts\\mlc.exe"
    };
    
    wchar_t currentDir[MAX_PATH];
    GetCurrentDirectoryW(MAX_PATH, currentDir);
    
    for (const auto& rel_path : possible_paths) {
        std::wstring full_path = std::wstring(currentDir) + L"\\" + rel_path;
        if (GetFileAttributesW(full_path.c_str()) != INVALID_FILE_ATTRIBUTES) {
            std::wcout << L"Found MLC at: " << full_path << std::endl;
            return full_path;
        }
    }
    
    return L""; // Not found
}

void run_mlc_measurements() {
    std::wstring mlcPath = find_mlc_path();
    
    if (mlcPath.empty()) {
        std::cerr << "Could not find mlc.exe in any expected location." << std::endl;
        std::cerr << "Please place mlc.exe in the scripts folder or current directory." << std::endl;
        return;
    }
    
    // Convert wide string to UTF-8 for system() command
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, mlcPath.c_str(), (int)mlcPath.length(), nullptr, 0, nullptr, nullptr);
    std::string mlcPath_utf8(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, mlcPath.c_str(), (int)mlcPath.length(), &mlcPath_utf8[0], size_needed, nullptr, nullptr);
    
    std::string command = "\"" + mlcPath_utf8 + "\" --latency_matrix > mlc_output.txt 2>&1";
    std::cout << "Executing: " << command << std::endl;
    
    int result = system(command.c_str());
    
    if (result == 0) {
        std::cout << "MLC executed successfully!" << std::endl;
    } else {
        std::cerr << "MLC execution failed with error code: " << result << std::endl;
    }
}

int main() {
    set_high_priority_affinity();
    // Try our improved method first
    std::cout << "=== Improved Custom Measurements ===" << std::endl;
    test_memory_level_improved(32 * 1024, "L1");
    test_memory_level_improved(256 * 1024, "L2");
    test_memory_level_improved(8 * 1024 * 1024, "L3");
    test_memory_level_improved(64 * 1024 * 1024, "DRAM");
    
    std::cout << "\n";
    
    // Then use Intel MLC for accurate results
    run_mlc_measurements();
    
    return 0;
}