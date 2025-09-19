#pragma once
#include <windows.h>
#include <vector>
#include <random>

#define CACHE_LINE_SIZE 64

void flush_cache() {
    const size_t size = 64 * 1024 * 1024; // 64MB flush buffer
    volatile char* buffer = new char[size];
    for (size_t i = 0; i < size; i += CACHE_LINE_SIZE) {
        buffer[i] = i;
    }
    delete[] buffer;
}

void* aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

void aligned_free(void* ptr) {
    _aligned_free(ptr);
}