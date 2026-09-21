/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "my_IdentityN_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"

#ifdef __cplusplus
extern "C" {
#endif
EsIdentityNOutput MyEsIdentityN(EsCTensorHolder **x, int64_t x_num) {
  // 新增自定义逻辑
  int64_t y_num = x_num;

  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(x_num));
  EsCTensorHolder *non_null_in = nullptr;
  ES_ASSERT_NOTNULL(x);
  if (non_null_in == nullptr) {
    for (int64_t i = 0; i < x_num; ++i) {
      if (x[i] != nullptr) {
        non_null_in = x[i];
        break;
      }
    }
  }
  ES_ASSERT_NOTNULL(non_null_in);

  // 1. 根据入参查找GraphBuilder图构建器
  auto &builder = non_null_in->GetOwnerBuilder();
  auto ge_graph = builder.GetGraph();

  // 2. 根据算子原型构建合法Node实例并设置IR信息到节点上
  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("IdentityN")
      .Name( builder.GenerateNodeName("IdentityN").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .IrDefAttrs({
      })
      .InstanceDynamicInputNum("x", static_cast<int32_t>(x_num))
      .InstanceDynamicOutputNum("y", static_cast<int32_t>(y_num))
      .Build();

  // 3. 完成图上入参和Node实例的连边
  if ((x != nullptr) && (x_num > 0)) {
    for (int64_t i = 0; i < x_num; ++i) {
      auto one_x = x[i];
      ES_ASSERT_NOTNULL(one_x);
      ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_x->GetProducer(), one_x->GetOutIndex(), node, 0 + i));
    }
  }

  // 4. 通过Node实例构建TensorHolder对象返回
  auto y_holders = builder.CreateDynamicTensorHolderFromNode(node, 0, y_num);
  return EsIdentityNOutput{
      y_holders->data(),
      static_cast<int64_t>(y_holders->size()),
  };
}
#ifdef __cplusplus
}
#endif
