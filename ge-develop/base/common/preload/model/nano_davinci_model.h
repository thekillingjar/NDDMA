/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_
#define GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_
#include "common/preload/model/pre_davinci_model.h"

namespace ge {
class NanoDavinciModel : public PreDavinciModel {
 public:
  NanoDavinciModel() = default;
  virtual ~NanoDavinciModel() = default;
  Status Init() override;
  Status DoPartitionProcess() override;
  Status InitNodes(const ComputeGraphPtr &compute_graph) override;

 private:
  Status InitTaskId();
  Status MatchIndexToTaskIndex(const uint32_t label_idx, uint32_t &task_index) const;
  Status NanoAddSwitchKernel(const OpDescPtr &op_desc);
  Status GetTaskKernelOffset(const std::string &kernel_name, uint32_t &offset) const;
  Status NanoSetWeightData(OpDescPtr &op_desc) const;
  Status NanoAddSwitchConstNode(const std::vector<uint64_t> &cond_task_id_list, const ge::NodePtr &sw_node,
                                size_t &weight_offset, ComputeGraphPtr &graph) const;
  Status NanoSwitchWeightDataInit(ComputeGraphPtr &compute_graph, const ComputeGraph::Vistor<NodePtr> &all_nodes);
  Status InitSwitchWeightData(ComputeGraphPtr &compute_graph);
  Status InitSwitchNodes(const ComputeGraphPtr &compute_graph);
  Status InitFifoWindowCacheOffset(const ge::NodePtr &node);
  Status UpdateFifoWindowCacheRefOffset(const ge::NodePtr &node) const;
  Status SetAnchorsOffset(const ge::NodePtr &node, const uint32_t index, const bool is_input,
                          const uint32_t offset) const;
  Status SetPeerInDataOffset(const OutDataAnchorPtr out_anchor, const uint32_t offset) const;
  uint32_t search_id_{0U};
  std::map<uint32_t, int32_t> task_list_;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_NANO_DAVINCI_MODEL_H_