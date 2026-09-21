/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "my_Conv2D_c.h"
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"

#ifdef __cplusplus
extern "C" {
#endif
// 基于原来Gen函数新增x_format && filter_format入参, 用于设置输出Format
EsCTensorHolder *MyEsConv2D(EsCTensorHolder *x, ge::Format x_format,
                            EsCTensorHolder *filter, ge::Format filter_format,
                            EsCTensorHolder *bias, EsCTensorHolder *offset_w,
                            const int64_t *strides, int64_t strides_num,
                            const int64_t *pads, int64_t pads_num,
                            const int64_t *dilations, int64_t dilations_num,
                            int64_t groups, const char *data_format,
                            int64_t offset_x, const char *my_attr) {
  ES_ASSERT_NOTNULL(x);
  ES_ASSERT_NOTNULL(filter);

  // 1. 根据入参查找GraphBuilder图构建器
  auto &builder = x->GetOwnerBuilder();
  auto ge_graph = builder.GetGraph();

  // 2. 根据算子原型构建合法Node实例并设置IR信息到节点上
  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType("Conv2D")
      .Name( builder.GenerateNodeName("Conv2D").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"filter", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
          {"offset_w", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "strides",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "ListInt",
              ge::es::CreateFrom(std::vector<int64_t>(strides, strides + strides_num))
          },
          {
              "pads",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "ListInt",
              ge::es::CreateFrom(std::vector<int64_t>(pads, pads + pads_num))
          },
          {
              "dilations",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "ListInt",
              ge::es::CreateFrom(std::vector<int64_t>(dilations, dilations + dilations_num))
          },
          {
              "groups",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFrom(static_cast<int64_t>(groups))
          },
          {
              "data_format",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "String",
              ge::es::CreateFrom(ge::AscendString(data_format))
          },
          {
              "offset_x",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Int",
              ge::es::CreateFrom(static_cast<int64_t>(offset_x))
          },
      })
      .Build();
      
  // 3. 完成图上入参和Node实例的连边
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, x->GetProducer(), x->GetOutIndex(), node, 0));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, filter->GetProducer(), filter->GetOutIndex(), node, 1));
  if (bias != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, bias->GetProducer(), bias->GetOutIndex(), node, 2));
  }
  if (offset_w != nullptr) {
    ES_ASSERT_GRAPH_SUCCESS(
        ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, offset_w->GetProducer(), offset_w->GetOutIndex(), node, 3));
  }
  // 新增自定义逻辑
  ge::TensorDesc td;

  // 设置输出x的format
  ES_ASSERT_GRAPH_SUCCESS(node.GetInputDesc(0, td));
  td.SetFormat(x_format);
  ES_ASSERT_GRAPH_SUCCESS(node.UpdateInputDesc(0, td));

  // 设置输出filter的format
  ES_ASSERT_GRAPH_SUCCESS(node.GetInputDesc(1, td));
  td.SetFormat(filter_format);
  ES_ASSERT_GRAPH_SUCCESS(node.UpdateInputDesc(1, td));

  ge::AscendString ascend_string_attr(my_attr);
  // 新增自定义属性
  node.SetAttr("my_attr", ascend_string_attr);

  // 4. 通过Node实例构建TensorHolder对象返回
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}
#ifdef __cplusplus
}
#endif
