/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <benchmark/benchmark.h>
#include "kernel/memory/allocator/scalable_allocator.h"

using namespace gert;

namespace {
	constexpr MemSize smallSize  = 1_KB;
	constexpr MemSize normalSize = 10_MB;
	constexpr MemSize largeSize  = 512_MB;

	constexpr size_t  batchCount = 10;

	constexpr size_t  allocOnlyTimes = 10000;
	void* reservedPtr = nullptr;
	void* TRY_ALLOC_PTR[batchCount] = {nullptr};
}

static void scale_alloc_and_free_small(benchmark::State &state) {
    ScalableAllocator allocator{0};
    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(smallSize);
    	((PageSpan*)reservedPtr)->Free();
    }
}

BENCHMARK(scale_alloc_and_free_small);

static void scale_alloc_and_free_normal(benchmark::State &state) {
	ScalableAllocator allocator{0};
    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(normalSize);
    	((PageSpan*)reservedPtr)->Free();
    }
}

BENCHMARK(scale_alloc_and_free_normal);

static void scale_alloc_and_free_large(benchmark::State &state) {
	ScalableAllocator allocator{0};
    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(largeSize);
    	((PageSpan*)reservedPtr)->Free();
    }
}

BENCHMARK(scale_alloc_and_free_large);

static void scale_alloc_and_free_in_size(benchmark::State &state) {
	ScalableAllocator allocator{0};
	reservedPtr = allocator.Alloc(normalSize);
	((PageSpan*)reservedPtr)->Free();

    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(normalSize);
    	((PageSpan*)reservedPtr)->Free();
    }
}

BENCHMARK(scale_alloc_and_free_in_size);

static void scale_alloc_and_free_fit_size(benchmark::State &state) {
	ScalableAllocator allocator{0};

	auto block1 = allocator.Alloc(normalSize);
	auto block2 = allocator.Alloc(normalSize);
	block1->Free();
	block2->Free();

    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(normalSize);
    	((PageSpan*)reservedPtr)->Free();
    }
}

BENCHMARK(scale_alloc_and_free_fit_size);

static void scale_alloc_and_free_split_and_merge_size(benchmark::State &state) {
	ScalableAllocator allocator{0};

	auto block1 = allocator.Alloc(1_MB);
	block1->Free();
	auto block2 = allocator.Alloc(128_KB);

    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(32_KB);
    	((PageSpan*)reservedPtr)->Free();
    }

	block2->Free();
}

BENCHMARK(scale_alloc_and_free_split_and_merge_size);

static void scale_alloc_and_free_multiple(benchmark::State &state) {
	ScalableAllocator allocator{0};
    for (auto _ : state) {
        for (auto& ptr : TRY_ALLOC_PTR) {
        	ptr = allocator.Alloc(normalSize);
        }
        for (auto ptr : TRY_ALLOC_PTR) {
        	((PageSpan*)ptr)->Free();
        }
    }
}

BENCHMARK(scale_alloc_and_free_multiple);

static void scale_alloc_only(benchmark::State &state) {
	ScalableAllocator allocator{0};
    for (auto _ : state) {
    	reservedPtr = allocator.Alloc(smallSize);
    }
}

BENCHMARK(scale_alloc_only)->Iterations(allocOnlyTimes);
