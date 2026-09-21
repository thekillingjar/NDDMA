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
#include "schedule/queue/lock_free/spsc_queue.h"
#include "schedule/queue/lock_free/mpmc_queue.h"
#include "schedule/queue/task_queue_factory.h"
#include "schedule/queue/task_queue.h"
#include "schedule/task/exec_task.h"
#include <iostream>
#include <atomic>
#include <string>
#include <chrono>
#include <thread>

using namespace gert;

//////////////////////////////////////////////////////////////

static constexpr uint32_t SIZE_LOG2 = 4;
static constexpr uint32_t QUEUE_LENGTH = 1 << SIZE_LOG2;

int items[QUEUE_LENGTH] = {0};

//////////////////////////////////////////////////////////////

static void spsc_queue_push_pop(benchmark::State &state) {
  SpscQueue<int *> q{SIZE_LOG2};
  bool found = true;
  int *item = 0;
  for (auto _ : state) {
    q.Push(&items[0]);
    found = q.Pop(item);
  }
  assert(found);
  assert(*item == 0);
}

BENCHMARK(spsc_queue_push_pop);

//////////////////////////////////////////////////////////////

static void mpmc_queue_push_pop(benchmark::State &state) {
  MpmcQueue<int *> q{SIZE_LOG2};
  bool found = true;
  int *item = 0;
  for (auto _ : state) {
    q.Push(&items[0]);
    found = q.Pop(item);
  }
  assert(found);
  assert(*item == 0);
}

BENCHMARK(mpmc_queue_push_pop);

//////////////////////////////////////////////////////////////

namespace {
ExecTask *FakeTaskPtr = (ExecTask *)(0xdeadbeaf);
ExecTask *resultTask = nullptr;
}  // namespace

static void spsc_queue_push_pop_by_factory(benchmark::State &state) {
  auto q = TaskQueueFactory::GetInstance().Create(TaskQueueType::SPSC, SIZE_LOG2);

  for (auto _ : state) {
    q->Push(*FakeTaskPtr);
    resultTask = q->Pop();
  }

  assert(resultTask == FakeTaskPtr);
}

BENCHMARK(spsc_queue_push_pop_by_factory);

//////////////////////////////////////////////////////////////

static void mpmc_queue_push_pop_by_factory(benchmark::State &state) {
  auto q = TaskQueueFactory::GetInstance().Create(TaskQueueType::MPMC, SIZE_LOG2);

  for (auto _ : state) {
    q->Push(*FakeTaskPtr);
    resultTask = q->Pop();
  }

  assert(resultTask == FakeTaskPtr);
}

BENCHMARK(mpmc_queue_push_pop_by_factory);
