/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_MODE_AUTO_AUTO_THREAD_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_MODE_AUTO_AUTO_THREAD_TASK_BUILDER_H_
#include "task_builder/mode/thread_task_builder.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace ffts {
class AutoTheadTaskBuilder : public TheadTaskBuilder {
 public:
  AutoTheadTaskBuilder();
  ~AutoTheadTaskBuilder() override;

  Status Initialize() override;

  Status GenFftsPlusContextId(ge::ComputeGraph &sgt_graph, std::vector<ge::NodePtr> &sub_graph_nodes,
                              uint64_t &ready_context_num, uint64_t &total_context_number,
                              std::vector<ge::NodePtr> &memset_nodes) override;

  Status GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes, std::vector<ge::NodePtr> &sub_graph_nodes,
                            domi::TaskDef &task_def) override;

 private:
  void SetCtxIdList(ge::NodePtr &node, uint32_t &context_id, const uint32_t &window_size) const;

  void GetStartFlag(const ge::NodePtr &node, bool &conn_start) const;

  void GetEndFlag(const ge::NodePtr &node, bool &conn_end) const;

  void SetAllAttrInFirstNode(ge::ComputeGraph &sgt_graph, const vector<uint32_t> &at_start_ctx_id_list,
                             const uint32_t &out_label_ctx_id) const;

  void SetAttrExceptCtxIdList(ge::ComputeGraph &sgt_graph, const vector<uint32_t> &at_start_ctx_id_list,
                              const vector<uint32_t> &at_end_ctx_id_list, int &count_node_conn_end,
                              const uint32_t &out_label_ctx_id, std::vector<ge::NodePtr> &sub_graph_nodes,
                              uint64_t &total_context_number) const;

  Status GenInLabelAtStartCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) const;

  Status GenOutLabelAtEndCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) const;

  Status AddSuccListInCtx(domi::FftsPlusTaskDef *ffts_plus_task_def, const FFTSPlusTaskBuilderPtr &task_builder,
                          const vector<uint32_t> &context_id_list, const vector<uint32_t> &output_context_id_list,
                          const ge::NodePtr &out_node) const;

  Status FillSerialDependency(const ge::NodePtr &sub_node, domi::FftsPlusTaskDef *ffts_plus_task_def,
                              const FFTSPlusTaskBuilderPtr &task_builder,
                              const vector<uint32_t> &context_id_list) const;

  Status UpdateAtomicSuccList(const ge::NodePtr &node, const vector<uint32_t> &context_id_list,
                              const FFTSPlusTaskBuilderPtr &task_builder,
                              domi::FftsPlusTaskDef *ffts_plus_task_def) const ;
  Status FillContextSuccList(const ge::NodePtr &sub_node, const FFTSPlusTaskBuilderPtr &task_builder,
                             const vector<uint32_t> &context_id_list,
                             const vector<uint32_t> &at_end_ctx_id_list, bool &netoutput_flag) const;
  Status GenerateAtomicCtx(std::vector<ge::NodePtr> &atomic_nodes, domi::FftsPlusTaskDef *ffts_plus_task_def) const;

  bool AddAtEndToWriteBackSuccList(const vector<uint32_t> &at_end_ctx_id_list,
                                   const vector<uint32_t> &context_id_list) const;

  domi::FftsPlusTaskDef *ffts_plus_task_def_{nullptr};
};
}  // namespace ffts
#endif  // FFTS_ENGINE_TASK_BUILDER_MODE_AUTO_AUTO_THREAD_TASK_BUILDER_H_
