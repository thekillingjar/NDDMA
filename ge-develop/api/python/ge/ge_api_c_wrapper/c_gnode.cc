/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include "graph/gnode.h"
#include "common/checker.h"
#include "graph/utils/node_adapter.h"
#include "graph/attr_value.h"
#include "ge_api_c_wrapper_utils.h"

using namespace ge;
using namespace ge::c_wrapper;

#ifdef __cplusplus
extern "C" {
#endif

void GeApiWrapper_GNode_DestroyGNode(const GNode *node) {
  delete node;
}

void GeApiWrapper_GNode_FreeGNodeArray(GNode **nodes) {
  delete[] nodes;
}

void GeApiWrapper_GNode_FreeIntArray(int32_t *arrs) {
  delete[] arrs;
}

const char *GeApiWrapper_GNode_GetName(const GNode *node) {
  GE_ASSERT_NOTNULL(node);
  AscendString name;
  GE_ASSERT_GRAPH_SUCCESS(node->GetName(name));
  return AscendStringToChar(name);
}

const char *GeApiWrapper_GNode_GetType(const GNode *node) {
  GE_ASSERT_NOTNULL(node);
  AscendString type;
  GE_ASSERT_GRAPH_SUCCESS(node->GetType(type));
  return AscendStringToChar(type);
}

GNode **GeApiWrapper_GNode_GetInControlNodes(const GNode *node, size_t *node_num) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node_num);
  return VecGNodesToArray(node->GetInControlNodes(), node_num);
}

graphStatus GeApiWrapper_GNode_GetInDataNodesAndPortIndexes(const GNode *node, int32_t in_index, GNode **in_node,
                                                            int32_t *index) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(in_node);
  GE_ASSERT_NOTNULL(index);

  auto in_node_and_index = node->GetInDataNodesAndPortIndexs(in_index);
  if (in_node_and_index.first == nullptr) {
    return GRAPH_FAILED;
  }
  *in_node = new (std::nothrow) GNode(*in_node_and_index.first);
  GE_ASSERT_NOTNULL(*in_node);
  *index = in_node_and_index.second;
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_GNode_GetAttr(const GNode *node, const char *key, void *attr_value) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(key);
  GE_ASSERT_NOTNULL(attr_value);
  auto *av = static_cast<AttrValue *>(attr_value);
  AscendString attr_name(key);
  return node->GetAttr(attr_name, *av);
}

graphStatus GeApiWrapper_GNode_SetAttr(GNode *node, const char *key, const void *attr_value) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(key);
  GE_ASSERT_NOTNULL(attr_value);

  auto *av = static_cast<AttrValue *>(const_cast<void *>(attr_value));

  AscendString attr_name(key);
  return node->SetAttr(attr_name, *av);
}

graphStatus GeApiWrapper_GNode_GetInputAttr(const GNode *node, const char *attr_name, uint32_t input_index,
                                            void *attr_value) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(attr_name);
  GE_ASSERT_NOTNULL(attr_value);

  auto *av = static_cast<AttrValue *>(attr_value);

  AscendString name(attr_name);
  return node->GetInputAttr(name, input_index, *av);
}

graphStatus GeApiWrapper_GNode_SetInputAttr(GNode *node, const char *attr_name, uint32_t input_index,
                                            const void *attr_value) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(attr_name);
  GE_ASSERT_NOTNULL(attr_value);

  auto *av = static_cast<const AttrValue *>(attr_value);

  AscendString name(attr_name);
  return node->SetInputAttr(name, input_index, *av);
}

graphStatus GeApiWrapper_GNode_GetOutputAttr(const GNode *node, const char *attr_name, uint32_t output_index,
                                             void *attr_value) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(attr_name);
  GE_ASSERT_NOTNULL(attr_value);

  auto *av = static_cast<AttrValue *>(attr_value);

  AscendString name(attr_name);
  return node->GetOutputAttr(name, output_index, *av);
}

graphStatus GeApiWrapper_GNode_SetOutputAttr(GNode *node, const char *attr_name, uint32_t output_index,
                                             const void *attr_value) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(attr_name);
  GE_ASSERT_NOTNULL(attr_value);

  auto *av = static_cast<const AttrValue *>(attr_value);

  AscendString name(attr_name);
  return node->SetOutputAttr(name, output_index, *av);
}

graphStatus GeApiWrapper_GNode_GetOutDataNodesAndPortIndexes(const GNode *node, int32_t out_index,
GNode **&out_node, int32_t *&index, int* size) {
  GE_ASSERT_NOTNULL(node);
  auto out_nodes_and_indexes = node->GetOutDataNodesAndPortIndexs(out_index);
  *size = out_nodes_and_indexes.size();
  auto c_out_nodes = new (std::nothrow) GNode *[*size];
  auto c_out_indexes = new (std::nothrow) int32_t [*size];
  for(int i = 0; i < *size; ++i) {
    auto node_index_pair = out_nodes_and_indexes.at(i);
    GE_ASSERT_NOTNULL(node_index_pair.first);
    c_out_nodes[i] = new (std::nothrow) GNode(*node_index_pair.first);
    c_out_indexes[i] = node_index_pair.second;
  }
  out_node = c_out_nodes;
  index = c_out_indexes;
  return GRAPH_SUCCESS;
}

GNode **GeApiWrapper_GNode_GetOutControlNodes(const GNode *node, size_t *node_num) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node_num);
  return VecGNodesToArray(node->GetOutControlNodes(), node_num);
}

size_t GeApiWrapper_GNode_GetInputsSize(const GNode *node) {
  GE_ASSERT_NOTNULL(node);
  return node->GetInputsSize();
}

size_t GeApiWrapper_GNode_GetOutputsSize(const GNode *node) {
  GE_ASSERT_NOTNULL(node);
  return node->GetOutputsSize();
}

bool GeApiWrapper_GNode_HasAttr(GNode *node, const char *attr_name) {
  GE_ASSERT_NOTNULL(node);
  AscendString name(attr_name);
  return node->HasAttr(name);
}

#ifdef __cplusplus
}
#endif
