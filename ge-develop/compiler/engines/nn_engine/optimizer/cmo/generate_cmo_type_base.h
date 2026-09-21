/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_BASE_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_BASE_H_

#include "graph/node.h"
#include "utils/graph_utils.h"
#include "common/aicore_util_types.h"

namespace fe {
struct StreamCtrl {
  int64_t src_stream_id;
  int64_t dst_stream_id;
  ge::NodePtr src_ctrl_node;
  ge::NodePtr dst_ctrl_node;
  StreamCtrl()
      : src_stream_id(-1), dst_stream_id(-1), src_ctrl_node(nullptr), dst_ctrl_node(nullptr) {}
};

using StreamCtrlMap = std::unordered_map<int64_t, StreamCtrl>;

class GenerateCMOTypeBase {
 public:
  GenerateCMOTypeBase();

  virtual ~GenerateCMOTypeBase(){};

  virtual void GenerateType(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls,
                            std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                            std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) {
    (void)node;
    (void)stream_ctrls;
    (void)prefetch_cache_map;
    (void)stream_node_map;
  };
 protected:
  const int64_t kPrefetchMin = 4096;  // 4k
  const int64_t kPrefetchMax = 1048576;  // 1M

  void AddToNodeCmoAttr(const ge::OpDescPtr &op_desc, const std::string &cmo_type,
                        const std::vector<CmoAttr> &attr_vec) const;

  uint64_t GetCacheSize() const;

  int64_t GetInputTensorSize(const ge::OpDescPtr &op_desc) const;

  int64_t GetOutputTensorSize(const ge::OpDescPtr &op_desc) const;

  int64_t GetWorkSpaceSize(const ge::OpDescPtr &op_desc) const;

  int64_t GetWeightSize(const ge::NodePtr &node) const;

  bool CheckParentOpIsAiCore(const ge::InDataAnchorPtr &in_anchor) const;

  bool ReadIsLifeCycleEnd(const ge::NodePtr &node, const ge::InDataAnchorPtr &in_anchor) const;

  bool IsPeerOutNoTaskNode(const ge::InDataAnchorPtr &in_anchor) const;

  bool CheckAndGetWeightSize(const ge::GeTensorDescPtr &weight, int64_t &tensor_size) const;
 private:
  void CalcTotalTensorSize(const ge::GeTensorDescPtr &tensor_desc, int64_t &total_tensor_size) const;
}; // class GenerateCMOTypeBase
}  // namespace fe

#endif
