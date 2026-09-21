/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shape.h"
#include "common/checker.h"
#include "formats/utils/formats_trans_utils.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/inference_rule.h"
#include "framework/common/debug/ge_log.h"
#include "kernel/kernel_log.h"
#include "runtime/mem.h"
#include "core/debug/kernel_tracing.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "aicore/converter/autofuse_node_converter.h"

namespace gert {
namespace kernel {
namespace {
using DfxInputSymbolInfo = ge::graphStatus(*)(const KernelContext *, char *, size_t);
const std::unordered_set<std::string> kAutofuseNodeSet = {"AscBackend", "FusedAscBackend", "AscBackendNoKernelOp"};
void ShapeToStringStream(std::stringstream &ss, const Shape &shape) {
  ss << "[";
  for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
    ss << shape.GetDim(j);
    if (j + 1U < shape.GetDimNum()) {
      ss << ", ";
    }
  }
  ss << "]";
}

void PrintFormatDtypeShape(std::stringstream &ss, ge::Format format, ge::DataType type, const Shape &shape) {
  ss << "[";
  ss << ge::TypeUtils::FormatToSerialString(format) << " ";
  ss << ge::TypeUtils::DataTypeToSerialString(type) << " ";
  ShapeToStringStream(ss, shape);
  ss << "]";
}

bool IsSymbolNode(const KernelContext *context) {
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto node_type = compute_node_info->GetNodeType();
  GE_ASSERT_NOTNULL(node_type);
  return (kAutofuseNodeSet.count(node_type) > 0);
}

std::string PrintInputSymbolInfo(const KernelContext *context) {
  auto input_data_num = context->GetInputValue<size_t>(0U);
  auto all_sym_num = context->GetInputValue<size_t>(
      input_data_num + static_cast<size_t>(SymbolInferShapeInput::kAllSymbolNum));
  auto input_symbol_info_func = context->GetInputValue<DfxInputSymbolInfo>(
      input_data_num + static_cast<size_t>(SymbolInferShapeInput::kDfxInputSymbolInfoFunc));
  GE_ASSERT_NOTNULL(input_symbol_info_func);
  // 使用所有符号的数量预估字符串数量："s1234: 1234,"，每个符号预估长度约12，所以总长度是12乘以符号数量加上一个\0
  size_t size = 12 * all_sym_num + 1;
  std::vector<char> out_symbol_info(size); 
  if (input_symbol_info_func(context, out_symbol_info.data(), size) != ge::GRAPH_SUCCESS) {
    return "Symbolic infos: Get symbol info failed.";
  }
  if (out_symbol_info[0] == '\0') {
    return "Symbolic infos: no symbol.";
  }
  return ("Symbolic infos: " + std::string(out_symbol_info.data()));
}
ge::graphStatus InferShapeByRule(KernelContext *context) {
  const auto ctx = reinterpret_cast<ExtendedKernelContext *>(context);
  const auto input_num = context->GetInputNum();
  GE_ASSERT(input_num > 0U);
  const auto compute_node_info = ctx->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);

  const auto *rule = context->GetInputValue<std::shared_ptr<ge::ShapeInferenceRule> *>(input_num - 1);
  GE_ASSERT_NOTNULL(rule);
  GE_ASSERT_NOTNULL(*rule);
  GE_ASSERT_EQ((*rule)->Error(), "");
  auto ret = (*rule)->InferOnRuntime(reinterpret_cast<InferShapeContext *>(context));
  if (ret != ge::GRAPH_SUCCESS) {
    KLOGE("Failed infer shape for node %s(%s)", ctx->GetNodeName(), ctx->GetNodeType());
    return ret;
  }
  ret = TransformAllOutputsShape(compute_node_info, context);
  if (ret != ge::GRAPH_SUCCESS) {
    KLOGE("Failed transfer shape format for node %s(%s)", ctx->GetNodeName(), ctx->GetNodeType());
    return ret;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadShapeRuleFromJson(KernelContext *context) {
  const auto input_num = context->GetInputNum();
  GE_ASSERT_EQ(input_num, 1U);

  auto *handle = context->GetOutputPointer<std::shared_ptr<ge::ShapeInferenceRule>>(0);
  GE_ASSERT_NOTNULL(handle);

  auto *rule_json = context->GetInputValue<const char *>(0);
  GE_ASSERT_NOTNULL(rule_json);
  KLOGD("Load shape inference rule from json: %s", rule_json);
  auto rule = ge::ShapeInferenceRule::FromJsonString(rule_json);

  handle->swap(rule);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LoadShapeRuleFromBinary(KernelContext *context) {
  const auto input_num = context->GetInputNum();
  GE_ASSERT_EQ(input_num, 3U);

  auto *handle = context->GetOutputPointer<std::shared_ptr<ge::ShapeInferenceRule>>(0);
  GE_ASSERT_NOTNULL(handle);

  auto *compiled_rule = context->GetInputValue<const uint8_t *>(0);
  const auto compiled_rule_size = context->GetInputValue<const size_t>(1);
  GE_ASSERT_NOTNULL(compiled_rule);
  auto *rule_json = context->GetInputValue<const char *>(2);
  GE_ASSERT_NOTNULL(rule_json);
  KLOGD("Load shape inference rule from binary, size %zu of json: %s", compiled_rule_size, rule_json);
  auto rule = std::make_shared<ge::ShapeInferenceRule>(
      ge::ShapeInferenceRule::FromCompiledBinary(compiled_rule, compiled_rule_size));

  handle->swap(rule);
  return ge::GRAPH_SUCCESS;
}
} // namespace

std::string PrintNodeType(const KernelContext *context) {
  std::stringstream ss;
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return "compute_node_info is nullptr";
  }
  ss << "node_type[" << compute_node_info->GetNodeType() << "]";
  return ss.str();
}

std::string PrintInputShapeInfo(const KernelContext *const context, const size_t &input_shape_start_index) {
  std::stringstream original_ss;
  std::stringstream storage_ss;
  original_ss << "input original shapes : ";
  storage_ss << "input storage shapes : ";
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return "compute_node_info is nullptr";
  }
  if (context->GetInputNum() < input_shape_start_index) {
    original_ss << "Trace failed, input num < input_shape_start_index, "
                << "context->GetInputNum:" << context->GetInputNum()
                << ", input_shape_start_index:"<< input_shape_start_index;
    return original_ss.str();
  }

  for (size_t i = 0U; i < compute_node_info->GetInputsNum(); ++i) {
    auto td = compute_node_info->GetInputTdInfo(i);
    if (td == nullptr) {
      return "The " + std::to_string(i) + "th's input tensor desc is nullptr";
    }
    auto storage_shape = context->GetInputPointer<StorageShape>(i + input_shape_start_index);
    if (storage_shape == nullptr) {
      return "The " + std::to_string(i) + "th's input storage shape is nullptr";
    }
    PrintFormatDtypeShape(original_ss, td->GetOriginFormat(), td->GetDataType(), storage_shape->GetOriginShape());
    PrintFormatDtypeShape(storage_ss, td->GetStorageFormat(), td->GetDataType(), storage_shape->GetStorageShape());
    if ((i + 1U) < compute_node_info->GetInputsNum()) {
      original_ss << ", ";
      storage_ss << ", ";
    }
  }
  original_ss << ", " << storage_ss.str();
  return original_ss.str();
}

std::string PrintOutputShapeInfo(const KernelContext *const context) {
  std::stringstream original_ss;
  std::stringstream storage_ss;
  original_ss << "output original shapes: ";
  storage_ss << "output storage shapes: ";
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return "compute_node_info is nullptr";
  }
  for (size_t i = 0U; i < compute_node_info->GetOutputsNum(); ++i) {
    auto td = compute_node_info->GetOutputTdInfo(i);
    if (td == nullptr) {
      return "The " + std::to_string(i) + "th's output tensor desc is nullptr";
    }
    auto storage_shape = context->GetOutputPointer<StorageShape>(i);
    if (storage_shape == nullptr) {
      return "The " + std::to_string(i) + "th's output storage shape is nullptr";
    }
    PrintFormatDtypeShape(original_ss, td->GetOriginFormat(),
                          td->GetDataType(), storage_shape->GetOriginShape());
    PrintFormatDtypeShape(storage_ss, td->GetStorageFormat(),
                          td->GetDataType(), storage_shape->GetStorageShape());
    if (i + 1U < compute_node_info->GetOutputsNum()) {
      original_ss << ", ";
      storage_ss << ", ";
    }
  }
  original_ss << ", " << storage_ss.str();
  return original_ss.str();
}

ge::graphStatus FindInferShapeFunc(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(0);
  auto space_registry = context->GetInputValue<gert::OpImplSpaceRegistryV2 *>(1);
  auto infer_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::InferShapeKernelFunc>(0);
  if (node_type == nullptr || infer_fun_ptr == nullptr || space_registry == nullptr) {
    KLOGE("Failed to find infer shape kernel, input or output is nullptr");
    return ge::GRAPH_FAILED;
  }

  auto op_funcs = space_registry->GetOpImpl(node_type);
  if ((op_funcs == nullptr) || (op_funcs->infer_shape == nullptr)) {
    KLOGE("Failed to find infer shape kernel, node type %s", node_type);
    return ge::GRAPH_FAILED;
  }
  *infer_fun_ptr = op_funcs->infer_shape;
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FindInferShapeFunc).RunFunc(FindInferShapeFunc);

/*
 * input: all storage_shapes , op_infershape_fun;
 * output: all storage_shapes
 */
KernelRegistry::KernelFunc GetOpInferShapeFun(const KernelContext *const context) {
  auto input_num = context->GetInputNum();
  if (input_num < 1U) {
    return nullptr;
  }
  return context->GetInputValue<KernelRegistry::KernelFunc>(input_num - 1);
}

std::vector<std::string> InferShapeKernelTrace(const KernelContext *context) {
  auto input_shape_info =
      IsSymbolNode(context) ? PrintInputSymbolInfo(context) : PrintInputShapeInfo(context, 0U);
  return {PrintNodeType(context),
          input_shape_info,
          PrintOutputShapeInfo(context)};
}

ge::graphStatus InferShape(KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto op_infer_fun = GetOpInferShapeFun(context);
  GE_ASSERT_NOTNULL(op_infer_fun);
  auto ret = op_infer_fun(context);
  if (ret != ge::GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "InferShape failed, node type %s, name %s, error-code %u",
                       extend_context->GetNodeType(), extend_context->GetNodeName(), ret);
    KLOGE("InferShape failed, node type %s, name %s, error-code %u", extend_context->GetNodeType(),
          extend_context->GetNodeName(), ret);
    return ret;
  }
  ret = TransformAllOutputsShape(compute_node_info, context);
  if (ret != ge::GRAPH_SUCCESS) {
    KLOGE("Failed to trans shape to 5D when infer shape for node %s, type %s, error-code %u",
          extend_context->GetNodeName(), extend_context->GetNodeType(), ret);
    return ret;
  }
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(InferShape).RunFunc(InferShape).OutputsCreator(BuildInferShapeOutputs)
    .TracePrinter(InferShapeKernelTrace);

inline ge::graphStatus BuildLoadShapeRuleOutputs(const ge::FastNode *node, KernelContext *context) {
  static_cast<void>(node);
  GE_ASSERT_NOTNULL(context->GetOutput(0));
  context->GetOutput(0)->SetWithDefaultDeleter(new (std::nothrow) std::shared_ptr<ge::ShapeInferenceRule>());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(LoadShapeRuleFromJson)
    .RunFunc(LoadShapeRuleFromJson)
    .OutputsCreator(BuildLoadShapeRuleOutputs);

REGISTER_KERNEL(LoadShapeRuleFromBinary)
    .RunFunc(LoadShapeRuleFromBinary)
    .OutputsCreator(BuildLoadShapeRuleOutputs);

REGISTER_KERNEL(InferShapeByRule).RunFunc(InferShapeByRule).OutputsCreator(BuildInferShapeOutputs)
    .TracePrinter(InferShapeKernelTrace);
}  // namespace kernel
}  // namespace gert