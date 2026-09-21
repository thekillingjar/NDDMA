/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "attribute_group/attr_group_shape_env.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "symbolic_infer_util.h"
#include "graph/ir_definitions_recover.h"
#include "graph/utils/graph_utils.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "symbolic_shape_inference.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/symbolizer/guard_dfx_context.h"

namespace ge {
namespace {
constexpr int64_t kMaxSymbolicValueSize = 200;

template <typename T>
ge::Symbol CreateSymbol(const uint8_t *ptr, size_t i) {
  T value;
  GE_ASSERT_EOK(memcpy_s(&value, sizeof(T), ptr + i * sizeof(T), sizeof(T)));
  return ge::Symbol(value);
}

std::map<ge::DataType, std::function<ge::Symbol(const uint8_t *, size_t)>> kComputeSupportTypes = {
    {DataType::DT_INT32, [](const uint8_t *ptr, size_t i) { return CreateSymbol<int32_t>(ptr, i); }},
    {DataType::DT_UINT32, [](const uint8_t *ptr, size_t i) { return CreateSymbol<uint32_t>(ptr, i); }},
    {DataType::DT_INT64, [](const uint8_t *ptr, size_t i) { return CreateSymbol<int64_t>(ptr, i); }},
    {DataType::DT_UINT64, [](const uint8_t *ptr, size_t i) { return CreateSymbol<uint64_t>(ptr, i); }},
    {DataType::DT_FLOAT, [](const uint8_t *ptr, size_t i) { return CreateSymbol<float>(ptr, i); }}};

std::string DebugInOutSymbolInfo(const OpDescPtr &op_desc, const std::string &stage) {
  std::stringstream ss;
  ss << stage << " op_desc[" << op_desc->GetName().c_str() << "] Input symbolic info: {";
  size_t i = 0;
  for (const auto& input_desc: op_desc->GetAllInputsDescPtr()) {
    auto attr = input_desc->GetAttrsGroup<SymbolicDescAttr>();
    if (attr == nullptr) {
      continue;
    }
    ss << "IN[" << i++ << "] {";
    ss << "origin:" << SymbolicInferUtil::DumpSymbolTensor(attr->symbolic_tensor).c_str();
    ss << "}. ";
  }
  ss << "} Output symbolic info: {";
  i = 0;
  for (const auto& output_desc: op_desc->GetAllOutputsDescPtr()) {
    auto attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
    if (attr == nullptr) {
      continue;
    }
    ss << "OUT[" << i++ << "] {";
    ss << "origin:" << SymbolicInferUtil::DumpSymbolTensor(attr->symbolic_tensor).c_str();
    ss << "}. ";
  }
  ss << "}";
  return ss.str();
}

bool IsInputDescValid(const ge::GeTensorDesc &input_desc, size_t &invalid_index_num) {
  if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
    invalid_index_num++;
    return false;
  }
  return true;
}

bool IsSupportInfer(const ge::OpDescPtr &op_desc) {
  GE_ASSERT_NOTNULL(op_desc);
  size_t idx = 0;
  if (!op_desc->GetSubgraphInstanceNames().empty()) {
    GELOGW("Subgraph is not supported Currently.");
    return false;
  }
  for (const auto &td : op_desc->GetAllInputsDesc()) {
    idx += 1;
    if (td.GetOriginFormat() != td.GetFormat()) {
      GELOGI("symbolic infer shape unsupported: name %s[%s], idx = %u, origin format [%d] is not equal to format[%d].",
             op_desc->GetNamePtr(), op_desc->GetTypePtr(), idx, td.GetOriginFormat(), td.GetFormat());
      return false;
    }
  }
  return true;
}

graphStatus CreateExpression(const ge::DataType dtype, const uint8_t *ptr, size_t size,
                             std::vector<Expression> &const_symbol) {
  if (kComputeSupportTypes.find(dtype) == kComputeSupportTypes.end()) {
    GELOGW("symbolic value generalize and compute not support data type %s",
           TypeUtils::DataTypeToSerialString(dtype).c_str());
    return UNSUPPORTED;
  }
  const_symbol.reserve(size);
  for (size_t i = 0; i < size; i++) {
    GELOGD("CreateExpression for dtype %s", TypeUtils::DataTypeToSerialString(dtype).c_str());
    const_symbol.emplace_back(kComputeSupportTypes[dtype](ptr, i));
  }
  return ge::GRAPH_SUCCESS;
}

bool SupportSymbolizeValue(const char *op_type, const GeTensorDescPtr &tensor_desc) {
  GE_ASSERT_NOTNULL(op_type);
  GE_ASSERT_NOTNULL(tensor_desc);
  GELOGI("Max symbolic value size %lld is supported", kMaxSymbolicValueSize);
  auto attr = tensor_desc->GetAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto symbol_shape_size = attr->symbolic_tensor.GetOriginSymbolShape().GetSymbolShapeSize();
  int64_t const_shape_size = -1;
  if (symbol_shape_size.GetExprType() == ExprType::kExprConstantInteger &&
      symbol_shape_size.GetConstValue(const_shape_size) && const_shape_size > kMaxSymbolicValueSize) {
    GELOGW("symbolic value generalize and compute only support shape size <= %lld, but current shape size is %lld",
        kMaxSymbolicValueSize, const_shape_size);
    return false;
  }
  return true;
}

std::unique_ptr<gert::SymbolTensor> GetInputSymbolTensorHolderCompute(const Operator &op, const OpDescPtr &op_desc,
                                                                      const GeTensorDescPtr &input_desc, size_t idx) {
  auto input_holder = ge::ComGraphMakeUnique<gert::SymbolTensor>();
  GE_ASSERT_NOTNULL(input_holder);
  auto attr = input_desc->GetAttrsGroup<SymbolicDescAttr>();

  // 1. 第一步，先从op上获取const值，调用op.GetConstInputData接口获取常量值
  // 2. 第一步失败，从当前的tensor_desc获取symbolic_value
  const auto index_name = op_desc->GetValidInputNameByIndex(static_cast<uint32_t>(idx));
  auto symbolic_shape = gert::SymbolShape();
  Tensor tensor;
  if (op.GetInputConstData(index_name.c_str(), tensor) == ge::GRAPH_SUCCESS) {
    auto size = tensor.GetTensorDesc().GetShape().GetShapeSize();
    if (tensor.GetTensorDesc().GetShape().GetDims().empty() && tensor.GetSize() != 0U) {
      size = 1;  // 处理scalar场景
    } else {
      for (const auto dim : tensor.GetTensorDesc().GetShape().GetDims()) {
        symbolic_shape.AppendDim(Symbol(dim));
      }
    }

    if (attr != nullptr && SupportSymbolizeValue(op_desc->GetTypePtr(), input_desc)) {
      auto symbolic_value = ge::ComGraphMakeUnique<std::vector<ge::Expression>>();
      GE_ASSERT_NOTNULL(symbolic_value);
      if (CreateExpression(tensor.GetDataType(), tensor.GetData(), size, *symbolic_value.get()) == GRAPH_SUCCESS) {
        input_holder->SetSymbolicValue(std::move(symbolic_value));
      }
    }

    input_holder->MutableOriginSymbolShape() = symbolic_shape;
    GELOGD("%s[%s] build input tensor from const, %s", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           SymbolicInferUtil::DumpSymbolTensor(*input_holder.get()).c_str());
    return input_holder;
  } else if (attr != nullptr) {
    input_holder->MutableOriginSymbolShape() = attr->symbolic_tensor.GetOriginSymbolShape();
    if (SupportSymbolizeValue(op_desc->GetTypePtr(), input_desc) &&
        attr->symbolic_tensor.GetSymbolicValue() != nullptr) {
      auto symbolic_value = ge::MakeUnique<std::vector<Expression>>(*attr->symbolic_tensor.GetSymbolicValue());
      GE_ASSERT_NOTNULL(symbolic_value);
      input_holder->SetSymbolicValue(std::move(symbolic_value));
    }
    GELOGD("%s[%s] build input tensor from attr, %s", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           SymbolicInferUtil::DumpSymbolTensor(*input_holder.get()).c_str());
  } else {
    input_holder.reset();
  }
  return input_holder;
}

Status ConstructComputeSymbolShapeContextInputs(const NodePtr &node,
                                                std::vector<std::unique_ptr<gert::SymbolTensor>> &inputs) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  size_t i = 0;
  for (const auto& input_desc: op_desc->GetAllInputsDescPtr()) {
    auto holder = GetInputSymbolTensorHolderCompute(op, op_desc, input_desc, i++);
    inputs.emplace_back(std::move(holder));
  }
  return ge::GRAPH_SUCCESS;
}

std::unique_ptr<gert::SymbolTensor> GetInputSymbolTensorHolder(const Operator &op, const OpDescPtr &op_desc, size_t idx,
                                                               bool is_data_dependency) {
  const auto &input_desc = op_desc->GetInputDesc(idx);
  auto input_holder = ge::ComGraphMakeUnique<gert::SymbolTensor>();
  GE_ASSERT_NOTNULL(input_holder);
  auto attr = input_desc.GetAttrsGroup<SymbolicDescAttr>();
  // 1. 第一步，先从op上获取const值，调用op.GetConstInputData接口获取常量值
  // 2. 第一步失败，从当前的tensor_desc获取symbolic_value
  if (is_data_dependency) {
    const auto index_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(idx));
    auto symbolic_shape = gert::SymbolShape();
    Tensor tensor;
    std::unique_ptr<std::vector<Expression> > symbolic_value = nullptr;
    if (op.GetInputConstData(index_name.c_str(), tensor) == ge::GRAPH_SUCCESS) {
      auto size = tensor.GetTensorDesc().GetShape().GetShapeSize();
      if (tensor.GetTensorDesc().GetShape().GetDims().empty() && tensor.GetSize() != 0U) {
        size = 1; // 处理scalar场景
        symbolic_shape.AppendDim(Symbol(1));
      } else {
        for (const auto dim : tensor.GetTensorDesc().GetShape().GetDims()) {
          symbolic_shape.AppendDim(Symbol(dim));
        }
      }
      symbolic_value = ge::ComGraphMakeUnique<std::vector<ge::Expression>>();
      GE_ASSERT_NOTNULL(symbolic_value);
      if (CreateExpression(tensor.GetDataType(), tensor.GetData(), size, *symbolic_value) != ge::GRAPH_SUCCESS) {
        symbolic_value.reset();
      }
    } else if ((attr != nullptr) && (attr->symbolic_tensor.GetSymbolicValue() != nullptr)) {
      symbolic_shape = attr->symbolic_tensor.GetOriginSymbolShape();
      symbolic_value = ge::ComGraphMakeUnique<std::vector<ge::Expression>>(*attr->symbolic_tensor.GetSymbolicValue());
      GE_ASSERT_NOTNULL(symbolic_value);
    } else {
      return nullptr;
    }

    input_holder->SetSymbolicValue(std::move(symbolic_value));
    input_holder->MutableOriginSymbolShape() = symbolic_shape;
    return input_holder;
  }
  if (attr != nullptr) {
    input_holder->MutableOriginSymbolShape().MutableDims() = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
    return input_holder;
  }
  return nullptr;
}

Status ConstructInferSymbolShapeContextInputs(const NodePtr &node, const gert::OpImplKernelRegistry::OpImplFunctionsV2 &func,
                                              std::vector<std::unique_ptr<gert::SymbolTensor>> &inputs) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    if (!IsInputDescValid(op_desc->GetInputDesc(i), invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }

    const size_t instance_index = i - invalid_index_num;
    const auto valid_op_ir_map = ge::OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
    GE_ASSERT_TRUE(!valid_op_ir_map.empty(), "Get valid op ir map failed, op[%s]", op_desc->GetName().c_str());
    size_t ir_index;
    GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, instance_index, ir_index),
                              "[Get][InputIrIndexByInstanceIndex] failed, op[%s], instance index[%zu], input_index[%zu]",
                              op_desc->GetName().c_str(), instance_index, i);
    auto holder = GetInputSymbolTensorHolder(op, op_desc, i, func.IsInputDataDependency(ir_index));
    inputs.emplace_back(std::move(holder));
  }
  return ge::GRAPH_SUCCESS;
}

Status ConstructInferSymbolShapeContextOutputs(const ge::OpDescPtr &op_desc,
                                               std::vector<std::unique_ptr<gert::SymbolShape>> &outputs) {
  for (size_t i = 0UL; i < op_desc->GetAllOutputsDescPtr().size(); i++) {
    auto symbol_shape_holder = ge::ComGraphMakeUnique<gert::SymbolShape>();
    GE_ASSERT_NOTNULL(symbol_shape_holder, "Create context holder outputs failed, op[%s]", op_desc->GetName().c_str());
    outputs.emplace_back(std::move(symbol_shape_holder));
  }
  return ge::GRAPH_SUCCESS;
}

Status ConstructComputeSymbolShapeContextOutputs(const ge::OpDescPtr &op_desc,
                                               std::vector<std::unique_ptr<gert::SymbolTensor>> &outputs) {
  for (size_t i = 0UL; i < op_desc->GetAllOutputsDescPtr().size(); i++) {
    auto symbol_shape_holder = ge::ComGraphMakeUnique<gert::SymbolTensor>();
    GE_ASSERT_NOTNULL(symbol_shape_holder, "Create context holder outputs failed, op[%s]", op_desc->GetName().c_str());
    outputs.emplace_back(std::move(symbol_shape_holder));
  }
  return ge::GRAPH_SUCCESS;
}

template <typename T>
std::vector<void *> GetVoidPtr(const std::vector<std::unique_ptr<T>> &outputs_holders) {
  std::vector<void *> outputs;
  outputs.reserve(outputs_holders.size());
  for (const auto &output_holder : outputs_holders) {
    outputs.push_back(output_holder.get());
  }
  return outputs;
}

Status UpdateOpDescOutShape(const ge::OpDescPtr &op_desc, gert::InferSymbolShapeContext *infer_symbol_shape_ctx) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(infer_symbol_shape_ctx);
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
    auto attr = op_desc->MutableOutputDesc(index)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    const auto symbol_shape = infer_symbol_shape_ctx->GetOutputSymbolShape(index);
    GE_ASSERT_NOTNULL(symbol_shape);
    attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = symbol_shape->GetDims();
  }
  return ge::GRAPH_SUCCESS;
}

Status UpdateOpDescOutShape(const ge::OpDescPtr &op_desc, gert::InferSymbolComputeContext *infer_symbol_shape_ctx) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(infer_symbol_shape_ctx);
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
    auto attr = op_desc->MutableOutputDesc(index)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    const auto symbol_tensor = infer_symbol_shape_ctx->GetOutputSymbolTensor(index);
    GE_ASSERT_NOTNULL(symbol_tensor);
    attr->symbolic_tensor = *symbol_tensor;
  }
  return ge::GRAPH_SUCCESS;
}

bool IsSameSymbolicTensor(const ge::GeTensorDescPtr &src, const ge::GeTensorDescPtr &dst) {
  GE_ASSERT_NOTNULL(src);
  GE_ASSERT_NOTNULL(dst);
  auto src_attr = src->GetAttrsGroup<SymbolicDescAttr>();
  auto dst_attr = dst->GetAttrsGroup<SymbolicDescAttr>();
  if (src_attr == nullptr || dst_attr == nullptr) {
    return false;
  }
  return (src_attr->symbolic_tensor.GetOriginSymbolShape() == dst_attr->symbolic_tensor.GetOriginSymbolShape()) &&
         (src_attr->symbolic_tensor.GetSymbolicValue() == dst_attr->symbolic_tensor.GetSymbolicValue());
}

Status ClearOutputSymbol(const ge::NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
    auto output_desc = op_desc->MutableOutputDesc(index);
    GE_ASSERT_NOTNULL(output_desc);
    (void)output_desc->DeleteAttrsGroup<SymbolicDescAttr>();
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

Status SymbolicShapeInference::SimplifyTensorSymbol(const GeTensorDescPtr &ge_tensor_desc) const {
  GE_ASSERT_NOTNULL(ge_tensor_desc);
  auto symbol_attr = ge_tensor_desc->GetAttrsGroup<SymbolicDescAttr>();
  if (symbol_attr == nullptr) {
    return GRAPH_SUCCESS;
  }
  auto &origin_symbol_dims = symbol_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims();
  for (size_t i = 0UL; i < origin_symbol_dims.size(); i++) {
    origin_symbol_dims[i] = origin_symbol_dims[i].Simplify();
  }
  const auto symbol_values = symbol_attr->symbolic_tensor.GetSymbolicValue();
  if (symbol_values == nullptr) {
    return GRAPH_SUCCESS;
  }
  auto simplify_values = symbol_attr->symbolic_tensor.MutableSymbolicValue();
  GE_ASSERT_NOTNULL(simplify_values);
  for (size_t i = 0UL; i < simplify_values->size(); i++) {
    simplify_values->at(i) = simplify_values->at(i).Simplify();
  }
  return GRAPH_SUCCESS;
}

Status SymbolicShapeInference::Simplify(const ComputeGraphPtr &graph) const {
  for (auto &node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
      auto output_desc_ptr = op_desc->MutableOutputDesc(index);
      GE_ASSERT_SUCCESS(SimplifyTensorSymbol(output_desc_ptr));
    }
    GE_ASSERT_SUCCESS(UpdateSymbolShapeAndDtypeToPeerInputs(node));
  }
  return ge::GRAPH_SUCCESS;
}

Status SymbolicShapeInference::Infer(const ComputeGraphPtr &graph) const {
  auto root_graph = ge::GraphUtils::FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);
  // 设置context
  ShapeEnvGuarder guarder(root_graph->GetAttrsGroup<ShapeEnvAttr>());
  for (auto &node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    GuardDfxContext dfx_context("node name:" + node->GetName() + " node type:" + node->GetType());
    // 每个算子不管能不能推导都需要更新对端的符号shape，这里不允许添加continue语句
    const auto ret = InferOneNode(node);
    GE_ASSERT_TRUE((ret == SUCCESS) || (ret == UNSUPPORTED), "[InferOneNode] failed,name[%s]", node->GetName().c_str());
    if (ret == UNSUPPORTED) {
      GELOGW("InferOneNode unsupport, node %s[%s]", node->GetName().c_str(), node->GetType().c_str());
    }
    GE_ASSERT_SUCCESS(UpdateSymbolShapeAndDtypeToPeerInputs(node));
  }
  GE_ASSERT_SUCCESS(Simplify(graph));
  return SUCCESS;
}

Status SymbolicShapeInference::UseStaticShapeIfWeCan(OpDescPtr &op_desc) const {
  GELOGW("We did not find infer symbol shape for op[%s].", op_desc->GetName().c_str());
  for (const auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
    GE_ASSERT_NOTNULL(output_desc);
    if (output_desc->GetOriginShape().IsUnknownShape()) {
      GELOGW("Only support static shape while no infer symbol shape for op[%s].", op_desc->GetName().c_str());
      return ge::UNSUPPORTED;
    }
    auto attr = output_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    attr->symbolic_tensor = gert::SymbolTensor();
    for (const auto &dim : output_desc->GetOriginShape().GetDims()) {
      attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims().push_back(ge::Symbol(dim));
    }
  }
  GE_ASSERT_SUCCESS(PrintSymbolShapeInfo(op_desc, "After_Infer_Symbol"));
  return ge::GRAPH_SUCCESS;
}

Status SymbolicShapeInference::DoInferAndUpdate(const NodePtr &node, const OpDescPtr &op_desc,
                                                const gert::OpImplKernelRegistry::OpImplFunctionsV2 *func) const {
  GE_ASSERT_NOTNULL(func);
  GE_ASSERT_NOTNULL(func->infer_symbol_shape);
  std::vector<std::unique_ptr<gert::SymbolTensor>> inputs_holder;
  std::vector<std::unique_ptr<gert::SymbolShape>> outputs_holder;
  inputs_holder.reserve(op_desc->GetAllInputsDescPtr().size());
  const auto ret = ConstructInferSymbolShapeContextInputs(node, *func, inputs_holder);
  GE_ASSERT_TRUE((ret == SUCCESS) || (ret == UNSUPPORTED), "[Construct][InferShapeContextInputs] failed, name[%s]",
                 op_desc->GetName().c_str());
  if (ret == UNSUPPORTED) {
    GELOGW("UNSUPPORTED infer shape");
    return ge::UNSUPPORTED;
  }

  GE_ASSERT_GRAPH_SUCCESS(ConstructInferSymbolShapeContextOutputs(op_desc, outputs_holder),
                          "[Construct][InferShapeContextOutputs] failed, op_desc[%s]", op_desc->GetName().c_str());

  const auto kernel_context_holder = gert::KernelRunContextBuilder()
                                         .Inputs(GetVoidPtr<gert::SymbolTensor>(inputs_holder))
                                         .Outputs(GetVoidPtr<gert::SymbolShape>(outputs_holder))
                                         .Build(op_desc);
  auto infer_symbol_shape_ctx = reinterpret_cast<gert::InferSymbolShapeContext *>(kernel_context_holder.context_);
  auto infer_ret = func->infer_symbol_shape(infer_symbol_shape_ctx);
  GE_ASSERT_TRUE((infer_ret == SUCCESS) || (infer_ret == UNSUPPORTED),
                 "[Call][InferSymbolShapeFunc] failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str());
  if (infer_ret == UNSUPPORTED) {
    GELOGW("Symbol infer unsupported, node %s[%s].",
        op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return UNSUPPORTED;
  }
  GE_ASSERT_GRAPH_SUCCESS(UpdateOpDescOutShape(op_desc, infer_symbol_shape_ctx));
  GE_ASSERT_SUCCESS(PrintSymbolShapeInfo(op_desc, "After_Infer_Symbol"));
  return ge::GRAPH_SUCCESS;
}


Status SymbolicShapeInference::DoComputeAndUpdate(const NodePtr &node, const OpDescPtr &op_desc,
                                                  const InferSymbolComputeKernelFunc &kernel_func) const {
  GELOGD("start Compute for node %s", op_desc->GetNamePtr());
  std::vector<std::unique_ptr<gert::SymbolTensor>> inputs_holder;
  std::vector<std::unique_ptr<gert::SymbolTensor>> outputs_holder;
  inputs_holder.reserve(op_desc->GetAllInputsDescPtr().size());
  GE_ASSERT_GRAPH_SUCCESS(ConstructComputeSymbolShapeContextInputs(node, inputs_holder));
  GE_ASSERT_GRAPH_SUCCESS(ConstructComputeSymbolShapeContextOutputs(op_desc, outputs_holder));

  const auto kernel_context_holder = gert::KernelRunContextBuilder()
                                         .Inputs(GetVoidPtr<gert::SymbolTensor>(inputs_holder))
                                         .Outputs(GetVoidPtr<gert::SymbolTensor>(outputs_holder))
                                         .Build(op_desc);
  auto infer_symbol_shape_ctx = reinterpret_cast<gert::InferSymbolComputeContext *>(kernel_context_holder.context_);
  auto ret = kernel_func(infer_symbol_shape_ctx);
  GE_ASSERT_TRUE(ret == ge::GRAPH_SUCCESS || ret == ge::UNSUPPORTED,
                 "[Call][InferSymbolCompute] failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str());
  if (ret == ge::UNSUPPORTED) {
    GELOGW("Symbol compute unsupported, node %s[%s].",
        op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ge::UNSUPPORTED;
  }
  GE_ASSERT_GRAPH_SUCCESS(UpdateOpDescOutShape(op_desc, infer_symbol_shape_ctx));
  GE_ASSERT_SUCCESS(PrintSymbolShapeInfo(op_desc, "After_Infer_Symbol"));
  return ge::GRAPH_SUCCESS;
}

Status SymbolicShapeInference::InferOneNode(NodePtr &node) const {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(PrintSymbolShapeInfo(op_desc, "Before_Infer_Symbol"));
  GE_ASSERT_NOTNULL(op_desc);
  if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
    return SUCCESS;
  }
  if (op_desc->GetType() == NETOUTPUT) {
    GELOGD("Netoutput %s no need to infer symbol shape.", op_desc->GetName().c_str());
    return ge::SUCCESS;
  }

  // 推导前清理残留shape
  GE_ASSERT_SUCCESS(ClearOutputSymbol(node));
  if (!IsSupportInfer(node->GetOpDesc())) {
    REPORT_INNER_ERR_MSG("W18888", "op %s[%s] Check support infer symbol shape failed.",
                       node->GetNamePtr(), node->GetTypePtr());
    GELOGW("op %s[%s] Check support infer symbol shape failed.", node->GetNamePtr(), node->GetTypePtr());
    return ge::UNSUPPORTED;
  }

  GE_ASSERT_SUCCESS(ge::RecoverOpDescIrDefinition(op_desc, op_desc->GetType()));
  auto kernel_func = SymbolicKernelFactory::GetInstance().Create(op_desc->GetType());
  if (kernel_func != nullptr) {
    auto ret = DoComputeAndUpdate(node, op_desc, kernel_func);
    GE_ASSERT_TRUE(ret == SUCCESS || ret == UNSUPPORTED,
                   "[DoComputeAndUpdate] failed, name[%s]", op_desc->GetName().c_str());
    if (ret == SUCCESS) {
      GELOGI("symbolic compute success, name[%s]", op_desc->GetName().c_str());
      return ge::SUCCESS;
    }
  }
  auto functions = gert::OpImplInferSymbolShapeRegistry::GetInstance().GetOpImpl(op_desc->GetType().c_str());
  if (functions == nullptr || functions->infer_symbol_shape == nullptr) {
    return UseStaticShapeIfWeCan(op_desc);
  }

  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  GE_ASSERT_NOTNULL(space_registry);
  auto function_new = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(
      space_registry->GetOpImpl(op_desc->GetType().c_str()));
  if (function_new != nullptr) {
    function_new->infer_symbol_shape = functions->infer_symbol_shape;
  } else {
    function_new = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(functions);
  }
  return DoInferAndUpdate(node, op_desc, function_new);
}

graphStatus SymbolicShapeInference::PrintSymbolShapeInfo(const OpDescPtr &op_desc,
    const std::string &stage) const {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return GRAPH_SUCCESS;
  }
  if (stage.compare("After_Infer_Symbol") == 0) {
    GE_ASSERT_SUCCESS(CheckOutputSymbolDimNumValid(op_desc));
  }
  GELOGD("%s", DebugInOutSymbolInfo(op_desc, stage).c_str());
  return GRAPH_SUCCESS;
}

graphStatus SymbolicShapeInference::CheckOutputSymbolDimNumValid(const OpDescPtr &op_desc) const {
  const auto &output_descs = op_desc->GetAllOutputsDescPtr();
  for (size_t i = 0UL; i < output_descs.size(); i++) {
    const auto &output_desc = output_descs.at(i);
    GE_ASSERT_NOTNULL(output_desc);
    auto symbol_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
    if (symbol_attr == nullptr) {
      continue;
    }
    auto symbol_shapes = symbol_attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
    if (output_desc->GetShape().IsUnknownDimNum()) {
      continue;
    }
    auto shapes = output_desc->GetShape().GetDims();
    if (symbol_shapes.size() != shapes.size()) {
      GELOGW("Symbol_DimNum_Check: [Node:%s(%s), output: %zu] Symbol shape dim num:%zu is not equal to shape dim num: %zu.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), i, symbol_shapes.size(), shapes.size());
    }
  }
  return GRAPH_SUCCESS;
}

// todo: 暂时无需考虑某个节点更新后影响其他pass场景，后续迁移到NodePass时需要考虑
graphStatus SymbolicShapeInference::UpdateSymbolShapeAndDtypeToPeerInputs(const NodePtr &node) const {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    auto output_tensor_desc = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    GE_ASSERT_NOTNULL(output_tensor_desc);
    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      auto peer_anchor_opdesc = peer_anchor->GetOwnerNode()->GetOpDesc();
      if (peer_anchor_opdesc == nullptr) {
        continue;
      }
      auto peer_input_desc = peer_anchor_opdesc->MutableInputDesc(peer_anchor->GetIdx());
      if (peer_input_desc == nullptr) {
        continue;
      }
      if (IsSameSymbolicTensor(output_tensor_desc, peer_input_desc)) {
        GELOGD("Peer dst symbolic tensor is same as src symbolic tensor. No need update.");
        continue;
      }
      // src tensor属性调用Get接口，dst tensor属性调用GetOrCreate接口
      auto src_attr = output_tensor_desc->GetAttrsGroup<SymbolicDescAttr>();
      if (src_attr != nullptr) {
        auto dst_attr = peer_input_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>();
        GE_ASSERT_NOTNULL(dst_attr);
        dst_attr->symbolic_tensor.MutableOriginSymbolShape() = src_attr->symbolic_tensor.GetOriginSymbolShape();
        std::unique_ptr<std::vector<Expression> > symbolic_value = nullptr;
        if (src_attr->symbolic_tensor.GetSymbolicValue() != nullptr) {
          symbolic_value = ge::MakeUnique<std::vector<Expression>>(*src_attr->symbolic_tensor.GetSymbolicValue());
        }
        dst_attr->symbolic_tensor.SetSymbolicValue(std::move(symbolic_value));
        GELOGD(
            "UpdatePeerInputDesc from src Node: [%s], origin symbol shape %s, "
            "To dst Node [%s]: origin symbol shape %s.",
            op_desc->GetName().c_str(),
            SymbolicInferUtil::VectorExpressionToStr(src_attr->symbolic_tensor.GetOriginSymbolShape().GetDims()).c_str(),
            peer_anchor_opdesc->GetName().c_str(),
            SymbolicInferUtil::VectorExpressionToStr(dst_attr->symbolic_tensor.GetOriginSymbolShape().GetDims()).c_str());
      } else {
        // 如果是nullptr，则清理对端
        (void)peer_input_desc->DeleteAttrsGroup<SymbolicDescAttr>();
        GELOGW("Cannot update symbol shape of peer input of node: %s due to infer failed",
            op_desc->GetName().c_str());
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
