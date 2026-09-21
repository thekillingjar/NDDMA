/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A28F1D6C52554BE98567C4D4DFD69089_H
#define A28F1D6C52554BE98567C4D4DFD69089_H

#include "graph/utils/graph_utils.h"
#include "framework/common/helper/model_helper.h"
#include "task_def_faker.h"

namespace gert {
struct GeModelBuilder{
  struct ModelResult {
    ge::GeModelPtr model;
    std::unordered_map<size_t, int64_t> task_indexes_to_node_id;
  };
  struct TbeConfig {
    TbeConfig() = default;
    TbeConfig(const char *name_or_type) : node_name_or_type(name_or_type), need_atomic(false) {}
    TbeConfig(const char *name_or_type, bool atomic) : node_name_or_type(name_or_type), need_atomic(atomic) {}
    TbeConfig(const TbeConfig &) = default;
    TbeConfig(TbeConfig &&) = default;
    TbeConfig &operator=(const TbeConfig &) = default;
    TbeConfig &operator=(TbeConfig &&) = default;
    std::string node_name_or_type;
    bool need_atomic;
  };
  GeModelBuilder(ge::ComputeGraphPtr compute_graph);
  GeModelBuilder &ConstWeight(const std::string &node_name);
  GeModelBuilder &AddTaskDef(const std::string &node_type_or_name, const TaskDefFaker& task_def);
  GeModelBuilder &AppendTaskDef(const TaskDefFaker &task_def);
  GeModelBuilder &FakeTbeBin(const std::vector<TbeConfig> &node_types_or_names);
  GeModelBuilder &AddTaskDefForAll(const TaskDefFaker& task_def);
  GeModelBuilder &AddWeight();
  GeModelBuilder &AddDefaultWeights();
  GeModelBuilder &AddDefaultTasks();
  GeModelBuilder &SetRootModelStreamNum(int64_t root_stream_num) {
    root_model_stream_num_ = root_stream_num;
    return *this;
  }
  GeModelBuilder &SetRootModelEventNum(int64_t root_event_num) {
    root_model_event_num_ = root_event_num;
    return *this;
  }
  GeModelBuilder &EnableSubMemInfo(const bool set_sub_mem_infos) {
    set_sub_mem_infos_ = set_sub_mem_infos;
    return *this;
  };
  ge::GeModelPtr Build();
  ge::GeRootModelPtr BuildGeRootModel();
  ModelResult BuildDetail();
  void AddCustAicpuKernelStore(const std::string &name);
  void AddTbeKernelStore();

 private:
  void BuildCommon();
  void SetAttrs();
  void SetTaskDefs();
  void SetTaskDef(TaskDefFaker& task_def, int64_t node_id);
  void FakeTbeBinToNodes();
  ge::TBEKernelStore BuildKernelStoreFromNodes() const;
  ge::CustAICPUKernelStore BuildCustAicpuKernelStoreFromNodes() const;
  void AddTbeKernelStoreFromNodes();

 private:
  ge::GeModelPtr ge_model;
  ge::ComputeGraphPtr compute_graph_;
  uint64_t op_index_;
  std::shared_ptr<domi::ModelTaskDef> model_task_def;
  std::unordered_map<size_t, int64_t> task_indexes_to_node_id_;
  std::vector<uint8_t> weight_buffer_;
  std::map<std::string, std::unique_ptr<TaskDefFaker>> node_name_or_types_to_task_def_faker_;
  std::map<std::string, TbeConfig> node_names_or_types_to_tbe_bin_fake_cfg_;
  std::unique_ptr<TaskDefFaker> all_task_def_faker_;
  std::vector<std::unique_ptr<TaskDefFaker>> append_task_def_fakers_;
  bool use_default_task_;
  bool set_sub_mem_infos_{false};
  int64_t root_model_stream_num_{1};
  int64_t root_model_event_num_{0};
};

}  // namespace gert

#endif
