/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_ASC_BACKEND_TORCH_FUSION_DECIDER_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_ASC_BACKEND_TORCH_FUSION_DECIDER_H_
#include "asc_backend_fusion_decider.h"
#include "utils/autofuse_utils.h"

namespace ge {
struct AxisMapInfo {
  std::unordered_map<int64_t, std::vector<int64_t>> node1_map;
  std::unordered_map<int64_t, std::vector<int64_t>> node2_map;
  std::vector<ge::Expression> base;
  std::vector<AxisPtr> node1_axis;
  std::vector<AxisPtr> node2_axis;
  std::vector<AxisPtr> base_axis;

  std::string ToString() const {
    std::ostringstream oss;
    auto print_node_map = [&oss](const std::string &name, const auto &node_map) {
      oss << name << ":\n";
      for (const auto &pair : node_map) {
        oss << "  Key: " << pair.first << " -> [";
        for (size_t i = 0U; i < pair.second.size(); ++i) {
          oss << pair.second[i];
          if (i != pair.second.size() - 1U) {
            oss << ", ";
          }
        }
        oss << "]\n";
      }
    };
    print_node_map("\nnode1_map", node1_map);
    print_node_map("\nnode2_map", node2_map);
    oss << "base:\n" << AutofuseUtils::VectorToStr(base);
    return oss.str();
  }
};

// torch下的子图融合策略，canfuse条件是是否能循环合并
class AscBackendTorchFusionDecider : public AscBackendFusionDecider {
 public:
  AscBackendTorchFusionDecider() = default;
  ~AscBackendTorchFusionDecider() override = default;

  AscBackendTorchFusionDecider(const AscBackendTorchFusionDecider &) = delete;
  AscBackendTorchFusionDecider &operator=(const AscBackendTorchFusionDecider &) = delete;

  // 检查两个节点是否可以垂直融合
  bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) override {
    // 检查两个节点是否满足融合条件
    return CanFuse(node1, node2);
  }

  // 检查两个节点是否可以水平融合
  bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) override {
    return CanFuse(node1, node2);
  }

  // 融合两个节点
  NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) override;

  // 获取融合对的优先级
  FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) override {
    (void)node1;
    (void)node2;
    return FusionPriority::HIGH;
  }

 private:
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) const override;
  Status UnifySubgraphAxis(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info,
                           AxisMapInfo &map_info) const;
  bool CheckFusionStrategy(const NodePtr &node1, const NodePtr &node2) const;
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_ASC_BACKEND_TORCH_FUSION_DECIDER_H_
