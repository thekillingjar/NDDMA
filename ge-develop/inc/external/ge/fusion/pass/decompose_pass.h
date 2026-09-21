/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GRAPH_FUSION_DECOMPOSE_PASS_H
#define INC_EXTERNAL_GE_GRAPH_FUSION_DECOMPOSE_PASS_H
#include "graph/graph.h"
#include "ge_common/ge_api_types.h"
#include "fusion_base_pass.h"
#include "fusion_pass_reg.h"
namespace ge{
namespace fusion {
using GraphUniqPtr = std::unique_ptr<Graph>;
class DecomposePass : public FusionBasePass {
 public:
  explicit DecomposePass(const std::vector<AscendString> &op_types);
  /**
   * 使用注册时声明的op type，找到图中匹配的node
   * 对匹配到的node, 依次调用子类定义的MeetRequirements函数，判断是否可被替换
   * 获取到子类定义的replacement，对图中匹配到节点依次进行替换
   *
   * 注意: Run函数只处理当前图，若需要处理子图，由Pass调用者来负责
   * @param graph
   * @return
   */
  Status Run(GraphPtr &graph, CustomPassContext &pass_context) override;

  virtual ~DecomposePass() = default;

 protected:
  /**
   * 通过对匹配到的node进行条件判断，符合条件返回true
   * 该函数为匹配节点的过滤器
   * @param matched_node
   * @return
   */
  virtual bool MeetRequirements(const GNode &matched_node);
  /**
   * 定义替换结构
   * @param matched_node 目标图中匹配到的真实节点，携带shape/format等信息
   * @return
   */
  virtual GraphUniqPtr Replacement(const GNode &matched_node) = 0;

 private:
  std::vector<AscendString> op_types_;
};

#define REG_DECOMPOSE_PASS(pass_class, decompose_op_types) \
  REG_DECOMPOSE_PASS_UNIQ_HELPER(__COUNTER__, #pass_class, pass_class, decompose_op_types)
#define REG_DECOMPOSE_PASS_UNIQ_HELPER(ctr, pass_name, pass_class, decompose_op_types) \
  REG_DECOMPOSE_PASS_UNIQ(ctr, pass_name, pass_class, decompose_op_types)
#define REG_DECOMPOSE_PASS_UNIQ(ctr, pass_name, pass_class, decompose_op_types)                                   \
  static ::ge::fusion::PassRegistrar register_pass##ctr __attribute__((unused)) =                                 \
      ::ge::fusion::FusionPassRegistrationData((pass_name)).CreatePassFn([]() -> ::ge::fusion::FusionBasePass * { \
        return new (std::nothrow) pass_class(decompose_op_types);                                                 \
      })
} // namespace fusion
} // namespace ge
#endif  // INC_EXTERNAL_GE_GRAPH_FUSION_DECOMPOSE_PASS_H
