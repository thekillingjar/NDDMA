/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_INVALID_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_INVALID_H_

#include "generate_cmo_type_base.h"

namespace fe {
class GenerateCMOTypeInvalid : public GenerateCMOTypeBase {
public:
  GenerateCMOTypeInvalid();

  ~GenerateCMOTypeInvalid() override {};

  void GenerateType(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls,
                    std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                    std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) override;

private:
  bool CheckReadDistance(const ge::OpDescPtr &op_desc, const ge::InDataAnchorPtr &in_anchor) const;

  bool IsLastPeerOutNode(const ge::InDataAnchorPtr &inanchor, const ge::NodePtr &node) const;

  void GenerateWorkSpace(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls) const;

  void GenerateInput(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls) const;

  void CheckReuseDistanceAndLabeled(const ge::NodePtr &node, const ge::NodePtr &pre_node,
                                    const CmoTypeObject cmo_type_obj, const int32_t index,
                                    const StreamCtrlMap &stream_ctrls) const;
  void LabeledInvalidOrBarrier(const ge::NodePtr &src_node, const CmoAttr &attr, const std::string &cmo_type) const;

  bool CheckReuseDistance(const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                          const ge::NodePtr &last_use_node, const StreamCtrlMap &stream_ctrls) const;
  bool CheckDiffStreamReuseDistance(const int64_t &src_stream_id, const int64_t &dst_stream_id,
                                   const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                                   const StreamCtrlMap &stream_ctrls) const;
  bool CheckInputNeedGenerate(const ge::NodePtr &node, const ge::InDataAnchorPtr &in_anchor) const;

  bool IsNeedGenerate(const ge::NodePtr &node) const;
};
} // namespace fe
#endif
