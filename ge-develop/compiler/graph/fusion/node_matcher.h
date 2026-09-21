/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_NODE_MATCHER_H
#define CANN_GRAPH_ENGINE_NODE_MATCHER_H

#include "graph/node.h"
#include "framework/common/types.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/utils/constant_utils.h"
namespace ge {
namespace fusion {
class NodeMatcher {
 public:
  /**
   * 返回t_node是否与p_node匹配
   * @param p_node 匹配图pattern graph中的node
   * @param t_node 待匹配图target graph中的node
   * @return
   */
  virtual bool IsMatch(const NodePtr &p_node, const NodePtr &t_node) const = 0;
  virtual ~NodeMatcher() = default;
};

class DataMatcher : public NodeMatcher {
 public:
  bool IsMatch(const NodePtr &p_node, const NodePtr &t_node) const override {
    (void)p_node;
    (void)t_node;
    return true;
  };
};
/**
 * 常量节点的matcher，通过enable_value_match_开关配置支持值匹配
 * 当enable_value_match_为false:若t_node类型为const/constant，则认为匹配成功
 * 当enable_value_match_为true: 支持Scaler或一维向量的值匹配，(这两种场景在图融合中最常用）
 *                             若超出matcher支持范围，返回不匹配及miss原因，引导用户转向自定义校验逻辑
 */
class ConstantMatcher : public NodeMatcher {
 public:
  explicit ConstantMatcher(bool enable_value_match, bool enable_match_cross_subgraph)
      : enable_value_match_(enable_value_match), enable_match_cross_subgraph_(enable_match_cross_subgraph) {}
  bool IsMatch(const NodePtr &p_node, const NodePtr &t_node) const override;
 private:
  bool enable_value_match_ = false;
  bool enable_match_cross_subgraph_ = false;
};

/**
 * 普通计算节点的matcher，通过enable_ir_attr_match_开关配置支持ir属性的值匹配
 * 当enable_ir_attr_match_为false:若t_node类型与p_node一致，则认为匹配成功
 * 当enable_ir_attr_match_为true:支持对p_node上表达的IR属性进行匹配，包含属性个数，属性值。
 */
class NormalNodeMatcher : public NodeMatcher {
 public:
  explicit NormalNodeMatcher(bool enable_ir_attr_match) : enable_ir_attr_match_(enable_ir_attr_match) {}
  bool IsMatch(const NodePtr &p_node, const NodePtr &t_node) const override;
 private:
  bool enable_ir_attr_match_ = false;
};
} // namespace fusion
} // namespace ge
#endif  // CANN_GRAPH_ENGINE_NODE_MATCHER_H
