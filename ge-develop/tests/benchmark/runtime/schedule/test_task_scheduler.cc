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
#include "schedule/scheduler/task_scheduler.h"
#include "schedule/config/task_worker_config.h"
#include "schedule/worker/task_worker_factory.h"
#include "schedule/producer/task_producer.h"
#include "schedule/task/exec_task.h"
#include <atomic>
#include <thread>

using namespace gert;

//////////////////////////////////////////////////////////////

namespace {
struct FakeExecTask : ExecTask {
  void Execute() override {
    completed = true;
  }

  bool completed{false};
};

struct FakeTaskProducer : TaskProducer {
  FakeTaskProducer() {
    tasks.push_back(task);
  }

  ge::Status Prepare(const void *executeData) override {
    return ge::SUCCESS;
  }

  ge::Status StartUp() override {
    isFinished = false;
    return ge::SUCCESS;
  }

  TaskPackage Produce() override {
    if (isFinished) {
      return TaskPackage();
    }
    isFinished = true;
    return std::move(tasks);
  }

  void Recycle(TaskPackage &tasks) override {
    tasks.clear();
    tasks.push_back(task);
  }

  void Dump() const override {}

  TaskPackage tasks;
  FakeExecTask task;
  bool isFinished{false};
};

uint32_t dummy = 0;
constexpr void *execution_data = &dummy;
}  // namespace

//////////////////////////////////////////////////////////////

static void scheduler_used_one_spsc_worker(benchmark::State &state) {
  auto producer = new FakeTaskProducer;
  TaskScheduler scheduler(*producer);

  TaskWorkerConfig cfg;
  cfg.thread_count = 1;
  cfg.thread_mode = TaskThreadMode::MODERATE;
  cfg.bind_task_type = ExecTaskType::NORMAL;
  auto worker = TaskWorkerFactory::GetInstance().Create(cfg);
  scheduler.AddWorker(*worker, ExecTaskType::NORMAL);

  scheduler.Prepare(TaskScheduler::ScheduleData{execution_data}, -1, nullptr);

  for (auto _ : state) {
    producer->StartUp();
    scheduler.Schedule();
  }

  assert(producer->task.completed);
}

BENCHMARK(scheduler_used_one_spsc_worker);

//////////////////////////////////////////////////////////////

static void scheduler_used_one_mpmc_worker(benchmark::State &state) {
  auto producer = new FakeTaskProducer;
  TaskScheduler scheduler(*producer);

  TaskWorkerConfig cfg;
  cfg.thread_count = 2;
  cfg.thread_mode = TaskThreadMode::MODERATE;
  cfg.bind_task_type = ExecTaskType::NORMAL;
  auto worker = TaskWorkerFactory::GetInstance().Create(cfg);
  scheduler.AddWorker(*worker, ExecTaskType::NORMAL);

  scheduler.Prepare(TaskScheduler::ScheduleData{execution_data}, -1, nullptr);

  assert(!producer->task.completed);

  for (auto _ : state) {
    producer->StartUp();
    scheduler.Schedule();
  }

  scheduler.Dump();

  assert(producer->task.completed);
}

BENCHMARK(scheduler_used_one_mpmc_worker);
