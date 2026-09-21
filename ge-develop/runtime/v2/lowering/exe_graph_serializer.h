/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_EXE_GRAPH_SERIALIZER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_EXE_GRAPH_SERIALIZER_H_
#include "exe_graph/lowering/graph_frame.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "lowering/model_converter.h"
#include "graph/fast_graph/fast_node.h"
namespace gert {
enum class ExeGraphPhase {
  kWholeGraph,
  kModelDescSerialized,
  kOfflineOptimized,
  kOnlineOptimized,
  kTextAligned,
  kEnd
};
// 不完全实现，当前仅对ComputeNodeInfo、KernelExtendInfo、ModelDesc做了序列化，以后会对整张图都做序列化
class ExeGraphSerializer {
 public:
  explicit ExeGraphSerializer(const bg::GraphFrame &frame) : frame_(frame) {}
  ge::graphStatus SaveSerialization(const std::vector<ge::FastNode *> &all_exe_nodes) const;

  ExeGraphSerializer &SetComputeGraph(ge::ComputeGraphPtr compute_graph);
  ExeGraphSerializer &SetModelDescHolder(ModelDescHolder *model_desc_holder);
  ge::graphStatus SerializeModelDesc(bg::BufferPool &model_desc_pool) const;
  ExeGraphSerializer &SetExecuteGraph(ge::ExecuteGraph *const execute_graph);

 private:
  ge::graphStatus SerializeComputeNodeInfo(const std::vector<ge::FastNode *> &all_exe_nodes,
                                           bg::BufferPool &compute_node_infos, bg::BufferPool &buffer_pool) const;
  ge::graphStatus SerializeKernelExtendInfo(const std::vector<ge::FastNode *> &all_exe_nodes,
                                            bg::BufferPool &kernel_extend_infos, bg::BufferPool &buffer_pool) const;
  ge::graphStatus SerializeDfxExtendInfo(bg::BufferPool &dfx_extend_infos, bg::BufferPool &buffer_pool) const;

 private:
  const bg::GraphFrame &frame_;
  ge::ComputeGraphPtr compute_graph_;
  ge::ExecuteGraph *execute_graph_ = nullptr;
  ModelDescHolder *model_desc_holder_ = nullptr;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_EXE_GRAPH_SERIALIZER_H_
