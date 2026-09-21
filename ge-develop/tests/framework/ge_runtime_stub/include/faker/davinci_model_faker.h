/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_DAVINCI_MODEL_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_DAVINCI_MODEL_FAKER_H_
#include "graph/load/model_manager/davinci_model.h"
#include "stub/task_info_registry_stub.h"
namespace ge {
class DavinciModelFaker {
 public:
  constexpr static uint64_t kFmDeviceAddrBase = 0x10000000;
  constexpr static uint64_t kFmLength = 0x90000000;
  // model io 地址段紧跟在fm后面
  constexpr static uint64_t kModelIoDevAddrBase = 0xa0000000;

  static uint64_t GetFmDevBase(uint64_t seed) {
    return kFmDeviceAddrBase + (seed << 32);
  }

  static uint64_t GetModelIoDevBase(uint64_t seed) {
    return kModelIoDevAddrBase + (seed << 32);
  }

 public:
  DavinciModelFaker &GeModel(GeModelPtr model) {
    model_ = std::move(model);
    return *this;
  }
  DavinciModelFaker &ModelFmLengths(std::vector<int64_t> fm_lengths) {
    fms_lengths_ = std::move(fm_lengths);
    return *this;
  }
  DavinciModelFaker &ModelFusionLengths(std::vector<int64_t> fusion_lengths) {
    fusion_lengths_ = std::move(fusion_lengths);
    return *this;
  }
  DavinciModelFaker &ModelInputLengths(std::vector<int64_t> input_lengths) {
    input_lengths_ = std::move(input_lengths);
    return *this;
  }
  DavinciModelFaker &ModelOutputLengths(std::vector<int64_t> output_lengths) {
    output_lengths_ = std::move(output_lengths);
    return *this;
  }

  DavinciModelFaker &GenerateSymbolForTaskInfoFaker(TaskInfoRegistryStub *stub) {
    task_info_registry_stub_ = stub;
    return *this;
  }

  DavinciModelFaker &SetFmRefreshable(bool is_refreshable) {
    is_feature_map_refreshable_ = is_refreshable;
    return *this;
  }

  std::unique_ptr<DavinciModel> Build();

 private:
  void FakeAllocationTable(DavinciModel *dm);
  void SetSymbolToStubIfNeed();
  void SetAdviseSymbolAddr(const DavinciModel &dm, uint64_t &fm_offset);
  void SetAdviseWsAddr(uint64_t &fm_offset);

 private:
  std::vector<int64_t> fusion_lengths_;
  std::vector<int64_t> fms_lengths_;
  std::vector<int64_t> input_lengths_;
  std::vector<int64_t> output_lengths_;
  GeModelPtr model_;
  bool is_feature_map_refreshable_{false};
  TaskInfoRegistryStub *task_info_registry_stub_{nullptr};
};
}  // namespace ge

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_DAVINCI_MODEL_FAKER_H_
