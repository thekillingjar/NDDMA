/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_PREFETCH_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_PREFETCH_H_

#include "generate_cmo_type_base.h"
namespace fe {
class GenerateCMOTypePrefetch : public GenerateCMOTypeBase {
public:
  GenerateCMOTypePrefetch();

  ~GenerateCMOTypePrefetch() override {};

  void GenerateType(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls,
                    std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                    std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) override;

private:
  void LabeledPrefetch(const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                       std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map) const;

  ge::NodePtr GetLastPreNode(const ge::NodePtr &node,
                             std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) const;

  bool CheckSizeIsAvailable(const ge::NodePtr &src_node, const ge::NodePtr &dst_node) const;

  bool CheckNeedPretch(const ge::NodePtr &src_node, const ge::NodePtr &dst_node) const;
};
} // namespace fe
#endif
