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
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include "graph/graph.h"
#include "graph/utils/tensor_adapter.h"

namespace {
template <typename T, ge::DataType DT>
ge::GeTensor CreateGeTensor(const T *value, const int64_t *dims, int64_t dim_num) {
  int64_t shape_size = 1;
  std::vector<int64_t> dims_vec;
  ge::Shape shape;
  if (dims != nullptr) {
    dims_vec.assign(dims, dims + dim_num);
    shape = ge::Shape{dims_vec};
    shape_size = shape.GetShapeSize();
    // Shape::GetShapeSize在scalar时返回0（期望的是1），这里需要特殓处理
  }
  ge::TensorDesc td{shape, ge::FORMAT_ND, DT};
  td.SetOriginShape(shape);
  ge::Tensor tensor{td};
  tensor.SetData(static_cast<const uint8_t *>(static_cast<const void *>(value)), sizeof(T) * shape_size);
  return ge::TensorAdapter::AsGeTensor(tensor);
}
template <typename T, ge::DataType DT>
EsbTensor *EsCreateConst(EsbGraph *graph, const T *value, const int64_t *dims, int64_t dim_num) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(value);

  auto ge_tensor = CreateGeTensor<T, DT>(value, dims, dim_num);
  auto c = ge::CompliantOpDescBuilder()
      .OpType("Const")
      .Name(("Const" + std::to_string(graph->NextNodeIndex())).c_str())
      .IrDefOutputs({{"y", ge::kIrOutputRequired, ""}})
      .IrDefAttrs({{"value", ge::kAttrOptional, "Tensor", ge::AnyValue::CreateFrom(ge_tensor)}})
      .InstanceOutputShape("y", ge_tensor.GetTensorDesc().GetShape().GetDims())
      .Build();
  GE_ASSERT_NOTNULL(c);
  auto ge_graph = graph->GetComputeGraph();
  return graph->GetEsbTensorFromNode(ge_graph->AddNode(c), 0);
}

template <typename T, ge::DataType DT>
EsbTensor *EsCreateVariable(EsbGraph *graph, int32_t index, const T *value, const int64_t *dims, int64_t dim_num,
                            const char *container, const char *shared_name) {
  (void) container;
  (void) shared_name;
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(value);

  auto ge_tensor = CreateGeTensor<T, DT>(value, dims, dim_num);
  auto c = ge::CompliantOpDescBuilder()
      .OpType("Variable")
      .Name(("Variable" + std::to_string(index)).c_str())
      .IrDefInputs({{"x", ge::kIrInputRequired, ""}})
      .IrDefOutputs({{"y", ge::kIrOutputRequired, ""}})
      .IrDefAttrs({{"index", ge::kAttrOptional, "Int", ge::AnyValue::CreateFrom(static_cast<int64_t>(index))},
                   {"value", ge::kAttrOptional, "Tensor", ge::AnyValue::CreateFrom(ge_tensor)}})
      .InstanceOutputShape("y", ge_tensor.GetTensorDesc().GetShape().GetDims())
      .Build();
  GE_ASSERT_NOTNULL(c);
  auto ge_graph = graph->GetComputeGraph();
  return graph->GetEsbTensorFromNode(ge_graph->AddNode(c), 0);
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif
EsbGraph *EsCreateGraph(const char *name) {
  if (name == nullptr) {
    name = "graph";
  }
  return new EsbGraph(name);
}

void EsDestroyGraph(EsbGraph *graph) {
  delete graph;
}

EsbTensor *EsCreateGraphInputWithDetails(EsbGraph *graph, int index, const char *name, const char *type) {
  return graph->AddGraphInput(index, name, type);
}
int EsSetGraphOutput(EsbTensor *tensor, int output_index) {
  GE_ASSERT_NOTNULL(tensor);
  return static_cast<int>(tensor->GetOwner().SetGraphOutput(tensor, output_index));
}
void *EsBuildGraph(EsbGraph *graph) {
  return graph->BuildGraph().release();
}
EsbGraph *EsGetOwnerGraph(EsbTensor *tensor) {
  GE_ASSERT_NOTNULL(tensor);
  return &tensor->GetOwnerGraph();
}

EsbTensor *EsCreateConstInt64(EsbGraph *graph, const int64_t *value, const int64_t *dims, int64_t dim_num) {
  return EsCreateConst<int64_t, ge::DT_INT64>(graph, value, dims, dim_num);
}
EsbTensor *EsCreateVectorInt64(EsbGraph *graph, const int64_t *value, int64_t dim) {
  return EsCreateConstInt64(graph, value, &dim, 1);
}
EsbTensor *EsCreateScalarInt64(EsbGraph *graph, int64_t value) {
  return EsCreateConstInt64(graph, &value, nullptr, 0);
}
EsbTensor *EsCreateConstInt32(EsbGraph *graph, const int32_t *value, const int64_t *dims, int64_t dim_num) {
  return EsCreateConst<int32_t, ge::DT_INT32>(graph, value, dims, dim_num);
}
EsbTensor *EsCreateScalarInt32(EsbGraph *graph, int32_t value) {
  return EsCreateConstInt32(graph, &value, nullptr, 0);
}

EsbTensor *EsCreateScalarFloat(EsbGraph *graph, float value) {
  return EsCreateConst<float, ge::DT_FLOAT>(graph, &value, nullptr, 0);
}
EsbTensor *EsCreateScalarDouble(EsbGraph *graph, double value) {
  return EsCreateConst<double, ge::DT_DOUBLE>(graph, &value, nullptr, 0);
}

EsbTensor *EsCreateVariableInt32(EsbGraph *graph, int32_t index, const int32_t *value, const int64_t *dims,
                                 int64_t dim_num, const char *container, const char *shared_name) {
  return EsCreateVariable<int32_t, ge::DT_INT32>(graph, index, value, dims, dim_num, container, shared_name);
}

EsbTensor *EsCreateVariableInt64(EsbGraph *graph, int32_t index, const int64_t *value, const int64_t *dims,
                                 int64_t dim_num, const char *container, const char *shared_name) {
  return EsCreateVariable<int64_t, ge::DT_INT64>(graph, index, value, dims, dim_num, container, shared_name);
}

EsbTensor *EsCreateVariableFloat(EsbGraph *graph, int32_t index, const float *value, const int64_t *dims,
                                 int64_t dim_num, const char *container, const char *shared_name) {
  return EsCreateVariable<float, ge::DT_FLOAT>(graph, index, value, dims, dim_num, container, shared_name);
}

int EsSetShape(EsbTensor *tensor, const int64_t *shape, int64_t dim_num) {
  GE_ASSERT_NOTNULL(tensor);
  if (shape == nullptr) {
    GE_ASSERT_TRUE(dim_num == 0, "When shape is nullptr, dim_num should be 0(means a scalar).");
  }
  return static_cast<int>(tensor->SetShape(ge::GeShape(std::vector<int64_t>(shape, shape + dim_num))));
}
int EsSetSymbolShape(EsbTensor *tensor, const char *const *shape, int64_t dim_num) {
  GE_ASSERT_NOTNULL(tensor);
  if (shape == nullptr) {
    GE_ASSERT_TRUE(dim_num == 0, "When shape is nullptr, dim_num should be 0(means a scalar).");
  }
  return static_cast<int>(tensor->SetSymbolShape(shape, dim_num));
}
int EsSetInputSymbolShape(EsbTensor *tensor, const char *const *shape, int64_t dim_num) {
  GE_ASSERT_NOTNULL(tensor);
  if (shape == nullptr) {
    GE_ASSERT_TRUE(dim_num == 0, "When shape is nullptr, dim_num should be 0(means a scalar).");
  }
  return static_cast<int>(tensor->SetInputSymbolShape(shape, dim_num));
}
#ifdef __cplusplus
}
#endif

