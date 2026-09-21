/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTCASE_DUMP_UTILS_DUMP_CONFIG_BUILDER_H
#define TESTCASE_DUMP_UTILS_DUMP_CONFIG_BUILDER_H

#include <gtest/gtest.h>
#include "depends/runtime/src/runtime_stub.h"
#include "framework/executor/ge_executor.h"

namespace ge {
class DumpConfigBuilder {
public:
  DumpConfigBuilder() {
    config_.dump_path = "./";
    config_.dump_status = "off";
    config_.dump_debug = "off";
    config_.dump_exception = "0";
    unsetenv("NPU_COLLECT_PATH");
    unsetenv("NPU_COLLECT_PATH_EXE");
  }
  virtual ~DumpConfigBuilder() = default;

  void Commit() {
    GeExecutor tmp;
    ASSERT_EQ(tmp.SetDump(this->Build()), SUCCESS);
  }

protected:
  virtual DumpConfig Build() = 0;
  DumpConfig config_;
};

class DataDumpConfigBuilder : public DumpConfigBuilder {
protected:
  DumpConfig Build() override {
    config_.dump_status = "on";
    config_.dump_mode = mode_;
    config_.dump_op_switch = op_switch_;
    config_.dump_step = step_;
    config_.dump_list = list_;
    config_.dump_data = data_;
    return config_;
  }

public:
  DataDumpConfigBuilder &Mode(std::string mode) { mode_ = std::move(mode); return *this; }
  DataDumpConfigBuilder &OpSwitch(std::string op_switch) { op_switch_ = std::move(op_switch); return *this; }
  DataDumpConfigBuilder &Step(std::string step) { step_ = std::move(step); return *this; }
  DataDumpConfigBuilder &ModelConfig(std::vector<std::string> layers,
                                     std::vector<std::string> watcher_nodes = {}, std::string model_name = "") {
    list_.emplace_back(ModelDumpConfig{std::move(model_name), std::move(layers), std::move(watcher_nodes)});
    return *this;
  }
  DataDumpConfigBuilder &Data(std::string data) { data_ = std::move(data); return *this; }
private:
  std::string mode_ = "all";
  std::string op_switch_ = "on";
  std::string step_;
  std::vector<ModelDumpConfig> list_;
  std::string data_ = "tensor";
};

class OverflowDumpConfigBuilder : public DumpConfigBuilder {
protected:
  DumpConfig Build() override {
    config_.dump_debug = "on";
    config_.dump_step = step_;
    return config_;
  }

public:
  OverflowDumpConfigBuilder &Step(std::string step) { step_ = std::move(step); return *this; }
private:
  std::string step_;
  EnvGuard env_guard_{"SYNCSTREAM_OVERFLOW_RET", "aicore"};
};
} // namespace ge

#endif
