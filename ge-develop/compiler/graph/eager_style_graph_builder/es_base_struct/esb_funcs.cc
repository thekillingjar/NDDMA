/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "esb_funcs.h"
#include "es_c_graph_builder.h"
#include "es_c_tensor_holder.h"
#include "es_graph_builder.h"
#include "compliant_node_builder.h"
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/types.h"
#include "common/checker.h"
#include "mmpa/mmpa_api.h"
#include "graph/utils/type_utils.h"
#include <fstream>
#include <iostream>
#include "base/err_msg.h"

namespace {
const char *kIrIndexAttrName = "index";
const char *kIrIntAttrType = "VT_INT";
const char *kIrOutputNameY = "y";
const char *kIrInputNameX = "x";
const char *kDefaultGraphName = "graph";

#define ES_MAKE_GUARD(var, callback) const ScopeGuard const_guard_##var(callback)
class ScopeGuard {
public:
  // Noncopyable
  ScopeGuard(ScopeGuard const &) = delete;
  ScopeGuard &operator=(ScopeGuard const &) & = delete;

  explicit ScopeGuard(const std::function<void()> &on_exit_scope) : on_exit_scope_(on_exit_scope), dismissed_(false) {}

  ~ScopeGuard() {
    if (!dismissed_) {
      if (on_exit_scope_ != nullptr) {
        try {
          on_exit_scope_();
        } catch (std::bad_function_call &) { }
        catch (...) { }
      }
    }
  }

  void Dismiss() { dismissed_ = true; }

private:
  std::function<void()> on_exit_scope_;
  bool dismissed_ ;
};

template <typename T>
EsCTensor *EsCreateEsCTensorHelper(const void *value, const int64_t *dims, int64_t dim_num, ge::DataType dt,
                                   C_Format format) {
  auto ge_tensor =
      ge::es::CreateTensor<T>(static_cast<const T *>(value), dims, dim_num, dt, static_cast<ge::Format>(format));

  return static_cast<EsCTensor *>(static_cast<void *>(ge_tensor.release()));
}

/**
 * @brief 获取路径的绝对路径
 * @param path 路径
 * @return 绝对路径
 */
std::string GetRealPath(const char *path) {
  GE_ASSERT_NOTNULL(path);
  GE_ASSERT_TRUE(strnlen(path, static_cast<size_t>(MMPA_MAX_PATH)) < static_cast<size_t>(MMPA_MAX_PATH));

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  char resolved_path[MMPA_MAX_PATH] = {};
  GE_ASSERT_TRUE(mmRealPath(path, &(resolved_path[0U]), MMPA_MAX_PATH) == EN_OK);
  std::string res = &(resolved_path[0]);

  return res;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif
EsCGraphBuilder *EsCreateGraphBuilder(const char *name) {
  if (name == nullptr) {
    name = kDefaultGraphName;
  }
  return new EsCGraphBuilder(name);
}

void EsDestroyGraphBuilder(EsCGraphBuilder *graph) {
  delete graph;
}

EsCTensorHolder *EsCreateGraphInputWithDetails(EsCGraphBuilder *graph, int64_t index, const char *name,
                                               const char *type, C_DataType data_type, C_Format format,
                                               const int64_t *shape, int64_t dim_num) {
  return graph->AddGraphInput(index, name, type, data_type, format, shape, dim_num);
}

EsCTensorHolder *EsCreateGraphInput(EsCGraphBuilder *graph, int64_t index) {
  return graph->AddGraphInput(index, nullptr, nullptr);
}
uint32_t EsSetGraphOutput(EsCTensorHolder *tensor, int64_t output_index) {
  GE_ASSERT_NOTNULL(tensor);
  return static_cast<uint32_t>(tensor->GetOwnerBuilder().SetGraphOutput(tensor, output_index));
}
EsCGraph *EsBuildGraphAndReset(EsCGraphBuilder *graph) {
  return static_cast<EsCGraph *>(static_cast<void *>(graph->BuildGraphAndReset().release()));
}
EsCNode *EsGetProducer(EsCTensorHolder *tensor) {
  GE_ASSERT_NOTNULL(tensor);
  return static_cast<EsCNode *>(static_cast<void *>(&(tensor->GetProducer())));
}

EsCGraphBuilder *EsGetOwnerBuilder(EsCTensorHolder *tensor) {
  GE_ASSERT_NOTNULL(tensor);
  return &tensor->GetOwnerBuilder();
}
uint32_t EsAddControlEdge(EsCTensorHolder *dest_ctrl_tensor, EsCTensorHolder **src_ctrl_tensors,
                          int64_t ctrl_tensors_num) {
  GE_ASSERT_NOTNULL(dest_ctrl_tensor);
  GE_ASSERT_NOTNULL(src_ctrl_tensors);
  auto ge_graph = dest_ctrl_tensor->GetOwnerBuilder().GetGraph();
  auto &node = dest_ctrl_tensor->GetProducer();
  for (int64_t i = 0; i < ctrl_tensors_num; i++) {
    auto &ctrl_ins_node = src_ctrl_tensors[i]->GetProducer();
    GE_ASSERT_GRAPH_SUCCESS(ge_graph->AddControlEdge(ctrl_ins_node, node));
  }
  return ge::SUCCESS;
}

EsCTensor *EsCreateEsCTensor(const void *data, const int64_t *dim, int64_t dim_num, C_DataType data_type,
                             C_Format format) {
  GE_ASSERT_NOTNULL(data);

  // 从C_DataType映射到ge::DataType
  switch (data_type) {
    case C_DT_FLOAT:
      return EsCreateEsCTensorHelper<float>(data, dim, dim_num, ge::DT_FLOAT, format);
    case C_DT_FLOAT16:  // Use uint16_t for float16 size
      return EsCreateEsCTensorHelper<uint16_t>(data, dim, dim_num, ge::DT_FLOAT16, format);
    case C_DT_INT8:
      return EsCreateEsCTensorHelper<int8_t>(data, dim, dim_num, ge::DT_INT8, format);
    case C_DT_INT16:
      return EsCreateEsCTensorHelper<int16_t>(data, dim, dim_num, ge::DT_INT16, format);
    case C_DT_INT32:
      return EsCreateEsCTensorHelper<int32_t>(data, dim, dim_num, ge::DT_INT32, format);
    case C_DT_INT64:
      return EsCreateEsCTensorHelper<int64_t>(data, dim, dim_num, ge::DT_INT64, format);
    case C_DT_UINT8:
      return EsCreateEsCTensorHelper<uint8_t>(data, dim, dim_num, ge::DT_UINT8, format);
    case C_DT_UINT16:
      return EsCreateEsCTensorHelper<uint16_t>(data, dim, dim_num, ge::DT_UINT16, format);
    case C_DT_UINT32:
      return EsCreateEsCTensorHelper<uint32_t>(data, dim, dim_num, ge::DT_UINT32, format);
    case C_DT_UINT64:
      return EsCreateEsCTensorHelper<uint64_t>(data, dim, dim_num, ge::DT_UINT64, format);
    case C_DT_BOOL:
      return EsCreateEsCTensorHelper<bool>(data, dim, dim_num, ge::DT_BOOL, format);
    default:
      REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char*>({"parameter", "value", "reason"}),
          std::vector<const char*>({"data_type", ge::TypeUtils::DataTypeToAscendString(static_cast<ge::DataType>(data_type)).GetString(), "Doesn't support current data type."}));
      GELOGE(ge::GRAPH_FAILED, "[Create][EsCTensor] unsupported data type %s", ge::TypeUtils::DataTypeToAscendString(static_cast<ge::DataType>(data_type)).GetString());
      break;
  }

  return nullptr;
}

EsCTensor *EsCreateEsCTensorFromFile(const char *data_file_path, const int64_t *dim, int64_t dim_num,
                                     C_DataType data_type, C_Format format) {
  const std::string real_path = GetRealPath(data_file_path);
  GE_ASSERT_TRUE(!real_path.empty(), "Data file path: %s is invalid.", data_file_path);

  std::ifstream ifs(real_path, std::ifstream::binary);
  GE_ASSERT_TRUE(ifs.is_open(), "Data file cannot be opened, path: %s.", data_file_path);

  (void)ifs.seekg(0, std::ifstream::end);
  const size_t act_file_len = ifs.tellg();
  ifs.clear();
  // 校验文件长度大于0
  GE_ASSERT_TRUE(act_file_len > 0U, "File length should be greater than 0, path: %s.", data_file_path);
  int64_t total_elements = 1;  // dims为空代表标量，默认值使用1
  // 如果dim非空，校验文件长度是否足够
  if (dim != nullptr) {
    for (int64_t i = 0; i < dim_num; ++i) {
      total_elements *= dim[i];
    }
  }

  const size_t data_type_size = ge::GetSizeByDataType(static_cast<ge::DataType>(data_type));
  GE_ASSERT_TRUE(data_type_size > 0, "Unsupported data type: %s.",
                 ge::TypeUtils::DataTypeToAscendString(static_cast<ge::DataType>(data_type)).GetString());
  const size_t expected_size = static_cast<size_t>(total_elements) * data_type_size;
  GE_ASSERT_TRUE(act_file_len >= expected_size,
                 "File length %zu is less than expected size %zu (dim product %ld * data_type_size %zu), path: %s.",
                 act_file_len, expected_size, total_elements, data_type_size, data_file_path);
  auto bin_buff = std::make_unique<char[]>(act_file_len);
  (void)ifs.seekg(0, std::ifstream::beg);
  (void)ifs.read(bin_buff.get(), static_cast<std::streamsize>(act_file_len));
  ES_MAKE_GUARD(close_ifs, [&ifs]() { ifs.close(); });
  GE_ASSERT_TRUE(ifs.good(), "file: %s cannot be read.", data_file_path);

  return EsCreateEsCTensor(bin_buff.get(), dim, dim_num, data_type, format);
}

#define ES_CREATE_CONST(type, cpp_type, ge_type)                                                           \
  EsCTensorHolder *EsCreateConst##type(EsCGraphBuilder *graph, const cpp_type *value, const int64_t *dims, \
                                       int64_t dim_num) {                                                  \
    return ge::es::EsCreateConst<cpp_type>(graph, value, dims, dim_num, ge_type, ge::FORMAT_ND);          \
  }

ES_CREATE_CONST(Int64, int64_t, ge::DT_INT64)
ES_CREATE_CONST(Float, float, ge::DT_FLOAT)
ES_CREATE_CONST(UInt64, uint64_t, ge::DT_UINT64)
ES_CREATE_CONST(Int32, int32_t, ge::DT_INT32)
ES_CREATE_CONST(UInt32, uint32_t, ge::DT_UINT32)

#undef ES_CREATE_CONST
EsCTensorHolder *EsCreateVectorInt64(EsCGraphBuilder *graph, const int64_t *value, int64_t dim) {
  return EsCreateConstInt64(graph, value, &dim, 1);
}
EsCTensorHolder *EsCreateVectorInt32(EsCGraphBuilder *graph, const int32_t *value, int64_t dim) {
  return EsCreateConstInt32(graph, value, &dim, 1);
}
EsCTensorHolder *EsCreateVectorUInt64(EsCGraphBuilder *graph, const uint64_t *value, int64_t dim) {
  return EsCreateConstUInt64(graph, value, &dim, 1);
}
EsCTensorHolder *EsCreateVectorUInt32(EsCGraphBuilder *graph, const uint32_t *value, int64_t dim) {
  return EsCreateConstUInt32(graph, value, &dim, 1);
}
EsCTensorHolder *EsCreateVectorFloat(EsCGraphBuilder *graph, const float *value, int64_t dim) {
  return EsCreateConstFloat(graph, value, &dim, 1);
}

EsCTensorHolder *EsCreateScalarInt64(EsCGraphBuilder *graph, int64_t value) {
  return EsCreateConstInt64(graph, &value, nullptr, 0);
}
EsCTensorHolder *EsCreateScalarInt32(EsCGraphBuilder *graph, int32_t value) {
  return EsCreateConstInt32(graph, &value, nullptr, 0);
}
EsCTensorHolder *EsCreateScalarUInt64(EsCGraphBuilder *graph, uint64_t value) {
  return EsCreateConstUInt64(graph, &value, nullptr, 0);
}
EsCTensorHolder *EsCreateScalarUInt32(EsCGraphBuilder *graph, uint32_t value) {
  return EsCreateConstUInt32(graph, &value, nullptr, 0);
}
EsCTensorHolder *EsCreateScalarFloat(EsCGraphBuilder *graph, float value) {
  return EsCreateConstFloat(graph, &value, nullptr, 0);
}

EsCTensorHolder *EsCreateVariable(EsCGraphBuilder *graph, int32_t index, const char *name) {
  GE_ASSERT_NOTNULL(graph);
  auto av_index = ge::AttrValue();
  auto index_value = static_cast<int64_t>(index);
  (void) av_index.SetAttrValue(index_value);
  GE_ASSERT_NOTNULL(graph->GetGraph(), "Graph has been build and reset, cannot create variable");
  auto node = ge::es::CompliantNodeBuilder(graph->GetGraph())
                  .OpType("Variable")
                  .Name(name)
                  .IrDefInputs({{kIrInputNameX, ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                  .IrDefOutputs({{kIrOutputNameY, ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .IrDefAttrs({{kIrIndexAttrName, ge::es::CompliantNodeBuilder::kEsAttrOptional, kIrIntAttrType, av_index}})
                  .Build();

  return graph->GetTensorHolderFromNode(node, 0);
}

uint32_t EsSetShape(EsCTensorHolder *tensor, const int64_t *shape, int64_t dim_num) {
  GE_ASSERT_NOTNULL(tensor);
  if (shape == nullptr) {
    GE_ASSERT_TRUE(dim_num == 0, "When shape is nullptr, dim_num should be 0(means a scalar).");
  }
  return static_cast<uint32_t>(tensor->SetShape(ge::Shape(std::vector<int64_t>(shape, shape + dim_num))));
}
uint32_t EsSetOriginSymbolShape(EsCTensorHolder *tensor, const char *const *shape, int64_t dim_num) {
  GE_ASSERT_NOTNULL(tensor);
  if (shape == nullptr) {
    GE_ASSERT_TRUE(dim_num == 0, "When shape is nullptr, dim_num should be 0(means a scalar).");
  }
  return static_cast<uint32_t>(tensor->SetOriginSymbolShape(shape, dim_num));
}
uint32_t EsSetDataType(EsCTensorHolder *tensor, C_DataType data_type) {
  GE_ASSERT_NOTNULL(tensor);
  return static_cast<uint32_t>(tensor->SetDataType(static_cast<ge::DataType>(data_type)));
}
uint32_t EsSetFormat(EsCTensorHolder *tensor, C_Format format) {
  GE_ASSERT_NOTNULL(tensor);
  return static_cast<uint32_t>(tensor->SetFormat(static_cast<ge::Format>(format)));
}
#define ES_SET_ATTR_FOR_GRAPH(type, value_type, value_expr)                                             \
  uint32_t EsSet##type##AttrForGraph(EsCGraphBuilder *graph, const char *attr_name, value_type value) { \
    GE_ASSERT_NOTNULL(graph);                                                                           \
    auto ge_graph = graph->GetGraph();                                                                  \
    GE_ASSERT_NOTNULL(ge_graph, "Graph has been build and reset, cannot set attr for graph");           \
    GE_ASSERT_TRUE(ge_graph->IsValid(), "You can not set attr for empty graph, add some node firstly"); \
    ge::AttrValue av;                                                                                   \
    GE_ASSERT_GRAPH_SUCCESS(av.SetAttrValue(value_expr));                                               \
    return static_cast<uint32_t>(ge_graph->SetAttr(attr_name, av));                                     \
  }

#define ES_SET_ATTR_FOR_TENSOR(type, value_type, value_expr)                                              \
  uint32_t EsSet##type##AttrForTensor(EsCTensorHolder *tensor, const char *attr_name, value_type value) { \
    GE_ASSERT_NOTNULL(tensor);                                                                            \
    auto node = tensor->GetProducer();                                                                    \
    ge::AttrValue av;                                                                                     \
    GE_ASSERT_GRAPH_SUCCESS(av.SetAttrValue(value_expr));                                                 \
    return static_cast<uint32_t>(node.SetOutputAttr(attr_name, tensor->GetOutIndex(), av));               \
  }

#define ES_SET_ATTR_FOR_NODE(type, value_type, value_expr)                                              \
  uint32_t EsSet##type##AttrForNode(EsCTensorHolder *tensor, const char *attr_name, value_type value) { \
    GE_ASSERT_NOTNULL(tensor);                                                                          \
    auto node = tensor->GetProducer();                                                                  \
    auto inner_value = value_expr;                                                                      \
    return static_cast<uint32_t>(node.SetAttr(attr_name, inner_value));                                 \
  }

// 生成所有属性设置函数
ES_SET_ATTR_FOR_GRAPH(Int64, int64_t, static_cast<int64_t>(value))
ES_SET_ATTR_FOR_GRAPH(String, const char *, ge::AscendString(value))
ES_SET_ATTR_FOR_GRAPH(Bool, bool, value)

ES_SET_ATTR_FOR_TENSOR(Int64, int64_t, static_cast<int64_t>(value))
ES_SET_ATTR_FOR_TENSOR(String, const char *, ge::AscendString(value))
ES_SET_ATTR_FOR_TENSOR(Bool, bool, value)

ES_SET_ATTR_FOR_NODE(Int64, int64_t, static_cast<int64_t>(value))
ES_SET_ATTR_FOR_NODE(String, const char *, ge::AscendString(value))
ES_SET_ATTR_FOR_NODE(Bool, bool, value)

#undef ES_SET_ATTR_FOR_GRAPH
#undef ES_SET_ATTR_FOR_TENSOR
#undef ES_SET_ATTR_FOR_NODE
#ifdef __cplusplus
}
#endif
