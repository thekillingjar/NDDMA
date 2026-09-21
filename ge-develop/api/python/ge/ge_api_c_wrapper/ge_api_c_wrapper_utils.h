/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_API_C_WRAPPER_UTILS_H
#define GE_API_C_WRAPPER_UTILS_H
#include <cstdlib>
#include <cstring>
#include <vector>
#include "graph/gnode.h"
#include "graph/ascend_string.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"
#include "ge/ge_api.h"
#include "ge/eager_style_graph_builder/c/esb_funcs.h"

namespace ge {
namespace c_wrapper {

// 通用释放函数
template <typename T>
void SafeFree(T *p) {
  if (p != nullptr) {
    std::free(p);
  }
}

// 为 POD 向量分配一块新内存并拷贝
template <typename T>
T *MallocCopy(const std::vector<T> &v) {
  if (v.empty()) {
    return nullptr;
  }
  T *buf = static_cast<T *>(std::malloc(sizeof(T) * v.size()));
  if (buf == nullptr) {
    return nullptr;
  }
  if (ge::GeMemcpy(ge::PtrToPtr<T, uint8_t>(buf), sizeof(T) * v.size(), ge::PtrToPtr<T, const uint8_t>(v.data()),
                   sizeof(T) * v.size()) != ge::SUCCESS) {
    SafeFree(buf);
    return nullptr;
  }
  return buf;
}

// 安全的字符串拷贝函数，使用malloc分配内存
inline char *MallocCopyString(const char *src) {
  if (src == nullptr) {
    return nullptr;
  }
  size_t len = std::strlen(src);
  std::vector<char> v(src, src + len + 1);  // +1 for null terminator
  return MallocCopy(v);
}

// 返回一块连续内存，FreeListString一次释放
inline char **MallocCopyListString(const std::vector<ge::AscendString> &v, int64_t *size) {
  if (v.empty()) {
    *size = 0;
    return nullptr;
  }
  *size = static_cast<int64_t>(v.size());

  // 计算总内存需求：指针数组 + 所有字符串数据
  size_t ptr_bytes = sizeof(char *) * v.size();
  size_t total_data_len = 0;
  for (const auto &s : v) {
    total_data_len += s.GetLength() + 1U;
  }
  size_t total = ptr_bytes + total_data_len;

  char **base = static_cast<char **>(std::malloc(total));
  if (base == nullptr) {
    return nullptr;
  }
  char *data_area = reinterpret_cast<char *>(base) + ptr_bytes;
  char *cursor = data_area;
  for (size_t i = 0; i < static_cast<size_t>(*size); ++i) {
    const std::string s = v[i].GetString();
    const size_t len = s.size() + 1U;
    if (ge::GeMemcpy(ge::PtrToPtr<char, uint8_t>(cursor), len, ge::PtrToPtr<const char, const uint8_t>(s.c_str()),
                     len) != ge::SUCCESS) {
      SafeFree(base);
      return nullptr;
    }
    base[i] = cursor;
    cursor += len;
  }
  return base;
}

inline GNode **VecGNodesToArray(std::vector<GNodePtr> nodes, size_t *node_num) {
  GE_ASSERT_NOTNULL(node_num);
  *node_num = nodes.size();
  auto c_nodes = new (std::nothrow) GNode *[*node_num];
  GE_ASSERT_NOTNULL(c_nodes);
  for (size_t i = 0U; i < *node_num; ++i) {
    c_nodes[i] = new (std::nothrow) GNode(*nodes.at(i));
    GE_ASSERT_NOTNULL(c_nodes[i]);
  }
  return c_nodes;
}

inline Tensor **VecTensorToArray(std::vector<Tensor> tensors, size_t *tensor_num) {
  GE_ASSERT_NOTNULL(tensor_num);
  *tensor_num = tensors.size();
  auto c_tensors = new (std::nothrow) Tensor *[*tensor_num];
  GE_ASSERT_NOTNULL(c_tensors);
  for (size_t i = 0U; i < *tensor_num; ++i) {
    c_tensors[i] = new (std::nothrow) Tensor(tensors.at(i));
    GE_ASSERT_NOTNULL(c_tensors[i]);
  }
  return c_tensors;
}

inline GNode **VecGNodesToArray(std::vector<GNode> nodes, size_t *node_num) {
  GE_ASSERT_NOTNULL(node_num);
  *node_num = nodes.size();
  auto c_nodes = new (std::nothrow) GNode *[*node_num];
  GE_ASSERT_NOTNULL(c_nodes);
  for (size_t i = 0U; i < *node_num; ++i) {
    c_nodes[i] = new (std::nothrow) GNode(nodes.at(i));
    GE_ASSERT_NOTNULL(c_nodes[i]);
  }
  return c_nodes;
}

inline int64_t *VecDimsToArray(std::vector<int64_t> dims, size_t *dims_num) {
  GE_ASSERT_NOTNULL(dims_num);
  *dims_num = dims.size();
  auto c_dims = new (std::nothrow) int64_t[*dims_num];
  GE_ASSERT_NOTNULL(c_dims);
  for (size_t i = 0U; i < *dims_num; ++i) {
    c_dims[i] = dims.at(i);
  }
  return c_dims;
}

inline const char *AscendStringToChar(const AscendString &s) {
  return MallocCopyString(s.GetString());
}
}  // namespace c_wrapper
}  // namespace ge

#ifdef __cplusplus
#ifndef char_t
using char_t = char;
#endif
extern "C" {
#endif
ge::graphStatus GeApiWrapper_AttrValue_SetBool(void *av, bool value);
void GeApiWrapper_FreeListString(char **p);
int64_t *GeApiWrapper_AttrValue_GetListInt(const void *av, int64_t *size);
float *GeApiWrapper_AttrValue_GetListFloat(const void *av, int64_t *size);
void GeApiWrapper_FreeString(char *p);
void GeApiWrapper_FreeListFloat(float *p);
ge::graphStatus GeApiWrapper_AttrValue_SetListDataType(void *av, const ge::DataType *data, int64_t size);
void GeApiWrapper_FreeListDataType(ge::DataType *p);
void GeApiWrapper_AttrValue_Destroy(void *av);
ge::graphStatus GeApiWrapper_AttrValue_SetListString(void *av, const char *const *data, int64_t size);
ge::graphStatus GeApiWrapper_AttrValue_GetInt(const void *av, int64_t *value);
ge::graphStatus GeApiWrapper_AttrValue_SetFloat(void *av, float value);
ge::graphStatus GeApiWrapper_AttrValue_SetInt(void *av, int64_t value);
ge::graphStatus GeApiWrapper_AttrValue_GetFloat(const void *av, float *value);
ge::graphStatus GeApiWrapper_AttrValue_SetString(void *av, const char *value);
ge::graphStatus GeApiWrapper_AttrValue_SetDataType(void *av, ge::DataType value);
char **GeApiWrapper_AttrValue_GetListString(const void *av, int64_t *size);
void GeApiWrapper_FreeListInt(int64_t *p);
void *GeApiWrapper_AttrValue_Create();
ge::graphStatus GeApiWrapper_AttrValue_GetBool(const void *av, bool *value);
bool *GeApiWrapper_AttrValue_GetListBool(const void *av, int64_t *size);
ge::graphStatus GeApiWrapper_AttrValue_SetListBool(void *av, const bool *data, int64_t size);
void GeApiWrapper_FreeListBool(bool *p);
char *GeApiWrapper_AttrValue_GetString(const void *av);
ge::graphStatus GeApiWrapper_AttrValue_SetListInt(void *av, const int64_t *data, int64_t size);
ge::graphStatus GeApiWrapper_AttrValue_GetDataType(const void *av, ge::DataType *value);
int32_t GeApiWrapper_AttrValue_GetValueType(const void *av);
ge::DataType *GeApiWrapper_AttrValue_GetListDataType(const void *av, int64_t *size);
ge::graphStatus GeApiWrapper_AttrValue_SetListFloat(void *av, const float *data, int64_t size);
ge::graphStatus GeApiWrapper_GNode_GetInputAttr(const ge::GNode *node, const char *attr_name, uint32_t input_index,
                                                void *attr_value);
ge::GNode **GeApiWrapper_GNode_GetOutControlNodes(const ge::GNode *node, size_t *node_num);
void GeApiWrapper_GNode_DestroyGNode(const ge::GNode *node);
ge::graphStatus GeApiWrapper_GNode_GetInDataNodesAndPortIndexes(const ge::GNode *node, int32_t in_index, ge::GNode **in_node,
                                                                int32_t *index);
const char *GeApiWrapper_GNode_GetType(const ge::GNode *node);
ge::GNode **GeApiWrapper_GNode_GetInControlNodes(const ge::GNode *node, size_t *node_num);
size_t GeApiWrapper_GNode_GetInputsSize(const ge::GNode *node);
ge::graphStatus GeApiWrapper_GNode_SetAttr(ge::GNode *node, const char *key, const void *attr_value);
ge::graphStatus GeApiWrapper_GNode_GetOutputAttr(const ge::GNode *node, const char *attr_name, uint32_t output_index,
                                                 void *attr_value);
void GeApiWrapper_GNode_FreeGNodeArray(ge::GNode **nodes);
const char *GeApiWrapper_GNode_GetName(const ge::GNode *node);
bool GeApiWrapper_GNode_HasAttr(ge::GNode *node, const char *attr_name);
size_t GeApiWrapper_GNode_GetOutputsSize(const ge::GNode *node);
ge::graphStatus GeApiWrapper_GNode_GetOutDataNodesAndPortIndexes(const ge::GNode *node, int32_t out_index,
                                                                 ge::GNode **&out_node, int32_t *&index, int *size);
ge::graphStatus GeApiWrapper_GNode_SetInputAttr(ge::GNode *node, const char *attr_name, uint32_t input_index,
                                                const void *attr_value);
ge::graphStatus GeApiWrapper_GNode_GetAttr(const ge::GNode *node, const char *key, void *attr_value);
ge::graphStatus GeApiWrapper_GNode_SetOutputAttr(ge::GNode *node, const char *attr_name, uint32_t output_index,
                                                 const void *attr_value);
void GeApiWrapper_GNode_FreeIntArray(int32_t *arrs);
ge::Status GeApiWrapper_GEFinalize();
ge::Status GeApiWrapper_GEInitialize(char **keys, char **values, int size);
ge::graphStatus GeApiWrapper_Graph_LoadFromAir(ge::Graph *graph, const char_t *file_name);
ge::graphStatus GeApiWrapper_Graph_AddControlEdge(ge::Graph *graph, ge::GNode &src_node, ge::GNode &dst_node);
ge::graphStatus GeApiWrapper_Graph_SetAttr(ge::Graph *graph, const char *key, const void *attr_value);
ge::graphStatus GeApiWrapper_Graph_RemoveNode(ge::Graph *graph, ge::GNode &node);
ge::graphStatus GeApiWrapper_Graph_Dump_To_File(const ge::Graph *graph, int32_t format, const char *suffix);
const char *GeApiWrapper_Graph_Dump_To_Stream(const ge::Graph *graph, int32_t format);
ge::graphStatus GeApiWrapper_Graph_FindNodeByName(const ge::Graph *graph, const char *name, ge::GNode **node);
const char *GeApiWrapper_Graph_GetName(const ge::Graph *graph);
ge::Graph *GeApiWrapper_Graph_CreateGraph(const char *name);
ge::graphStatus GeApiWrapper_Graph_SaveToAir(const ge::Graph *graph, const char_t *file_name);
ge::graphStatus GeApiWrapper_Graph_RemoveEdge(ge::Graph *graph, ge::GNode &src_node, const int32_t src_port_index,
                                              ge::GNode &dst_node, const int32_t dst_port_index);
ge::graphStatus GeApiWrapper_Graph_GetAttr(const ge::Graph *graph, const char *key, void *attr_value);
void GeApiWrapper_Graph_DestroyGraph(const ge::Graph *graph);
void GeApiWrapper_Graph_FreeGraphArray(ge::Graph **graphs);
ge::graphStatus GeApiWrapper_Graph_AddDataEdge(ge::Graph *graph, ge::GNode &src_node, const int32_t src_port_index,
                                               ge::GNode &dst_node, const int32_t dst_port_index);
ge::GNode **GeApiWrapper_Graph_GetAllNodes(const ge::Graph *graph, size_t *node_num);
ge::GNode **GeApiWrapper_Graph_GetDirectNode(const ge::Graph *graph, size_t *node_num);
ge::graphStatus GeApiWrapper_Graph_Dump_To_Onnx(ge::Graph *graph, const char *path, const char *suffix);
ge::Graph **GeApiWrapper_Graph_GetAllSubgraphs(const ge::Graph *graph, size_t *subgraph_num);
ge::Graph *GeApiWrapper_Graph_GetSubGraph(const ge::Graph *graph, const char *name);
ge::graphStatus GeApiWrapper_Graph_AddSubGraph(ge::Graph *graph, const ge::Graph *subgraph);
ge::graphStatus GeApiWrapper_Graph_RemoveSubgraph(ge::Graph *graph, const char *name);
ge::Session *GeApiWrapper_Session_CreateSession();
ge::Tensor** GeApiWrapper_Session_RunGraph(ge::Session *session, uint32_t graph_id, void **inputs, int input_count, size_t *tensor_num);
ge::Status GeApiWrapper_Session_AddGraph(ge::Session *session, uint32_t graph_id, ge::Graph *graph);
void GeApiWrapper_Session_DestroySession(const ge::Session *session);
ge::Format GeApiWrapper_Tensor_GetFormat(EsCTensor *tensor);
EsCTensor *GeApiWrapper_Tensor_CreateTensor();
void GeApiWrapper_Tensor_DestroyEsCTensor(EsCTensor *tensor);
ge::graphStatus GeApiWrapper_Tensor_SetFormat(EsCTensor *tensor, const ge::Format &format);
ge::DataType GeApiWrapper_Tensor_GetDataType(EsCTensor *tensor);
ge::graphStatus GeApiWrapper_Tensor_SetDataType(EsCTensor *tensor, const ge::DataType &dtype);
#ifdef __cplusplus
}
#endif
#endif  // GE_API_C_WRAPPER_UTILS_H
