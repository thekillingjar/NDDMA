/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shape_range.h"
#include "infer_shape.h"
#include "graph/ge_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph/compute_graph.h"
#include "graph/operator_factory_impl.h"
#include "graph/buffer.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "compatible_utils.h"
#include "common/util/mem_utils.h"
#include "core/debug/kernel_tracing.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace kernel {
namespace {
std::vector<std::string> CompatibleInferShapeKernelTrace(const KernelContext *context) {
  return {PrintNodeType(context),
          PrintInputShapeInfo(context, static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum)),
          PrintOutputShapeInfo(context)};
}

std::vector<std::string> CompatibleInferShapeRangeKernelTrace(const KernelContext *context) {
  return {PrintNodeType(context),
          PrintInputRangeInfo(context, static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum)),
          PrintOutputRangeInfo(context)};
}

ge::graphStatus UpdateInputShapeToOpDesc(KernelContext *context, ge::OpDescPtr &op_desc) {
  auto other_inputs_size = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum);
  GE_ASSERT_TRUE(context->GetInputNum() >= other_inputs_size);
  auto input_shapes_num = context->GetInputNum() - other_inputs_size;
  auto all_valid_input_desc = op_desc->GetAllInputsDescPtr();
  GE_ASSERT_EQ(all_valid_input_desc.size(), input_shapes_num);
  auto infer_context = reinterpret_cast<InferShapeContext *>(context);
  auto input_shape_start_pos = other_inputs_size;
  // Refreash input shape and format
  for (size_t i = 0U; i < input_shapes_num; ++i) {
    auto input_shape = infer_context->GetInputShape(input_shape_start_pos + i);
    GE_ASSERT_NOTNULL(input_shape);
    std::vector<int64_t> shape_dims;
    for (size_t j = 0U; j < input_shape->GetDimNum(); ++j) {
      shape_dims.emplace_back(input_shape->GetDim(j));
    }
    const auto &input_desc = all_valid_input_desc.at(i);
    GE_ASSERT_NOTNULL(input_desc);
    input_desc->SetShape(ge::GeShape(shape_dims));
    input_desc->SetOriginShape(ge::GeShape(shape_dims));
    // 不需要增加偏移，此处是去取计算节点上第i个desc
    auto input_desc_in_context = infer_context->GetInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc_in_context);
    // RT1时，算子的infershape只能拿到format字段，但是却需要用origin format
    input_desc->SetFormat(input_desc_in_context->GetOriginFormat());
    input_desc->SetOriginFormat(input_desc_in_context->GetOriginFormat());
  }
  // Refresh output format
  auto output_shapes_num = context->GetOutputNum();
  auto output_num_on_op = op_desc->GetOutputsSize();
  GE_ASSERT_EQ(output_shapes_num, output_num_on_op);
  for (size_t i = 0U; i < output_shapes_num; ++i) {
    const auto &output_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_desc);
    auto output_desc_in_context = infer_context->GetOutputDesc(i);
    GE_ASSERT_NOTNULL(output_desc_in_context);
    // RT1时，算子的infershape只能拿到format字段，但是却需要用origin format
    output_desc->SetFormat(output_desc_in_context->GetOriginFormat());
    output_desc->SetOriginFormat(output_desc_in_context->GetOriginFormat());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateOutputShapeToContext(const ge::OpDescPtr &op_desc, KernelContext *context) {
  auto output_num = context->GetOutputNum();
  auto output_num_on_op = op_desc->GetOutputsSize();
  if (output_num_on_op != output_num) {
    GELOGE(ge::PARAM_INVALID, "Output num on op %s is %zu, output num on context is %zu, not match.",
           op_desc->GetName().c_str(), output_num_on_op, output_num);
    return ge::PARAM_INVALID;
  }

  auto infer_context = reinterpret_cast<InferShapeContext *>(context);
  for (size_t i = 0U; i < output_num_on_op; ++i) {
    const auto &output_shape = op_desc->GetOutputDesc(i).GetShape().GetDims();
    auto output_shape_on_context = infer_context->GetOutputShape(i);
    GE_ASSERT_NOTNULL(output_shape_on_context);
    output_shape_on_context->SetDimNum(output_shape.size());
    for (size_t j = 0U; j < output_shape.size(); ++j) {
      output_shape_on_context->SetDim(j, output_shape[j]);
    }
  }
  return ge::GRAPH_SUCCESS;
}

// for infer shape range
ge::graphStatus UpdateInputShapeRangeToOpDesc(KernelContext *context, ge::OpDescPtr &op_desc) {
  auto other_inputs_size = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum);
  GE_ASSERT_TRUE(context->GetInputNum() >= other_inputs_size);
  auto input_shapes_num = context->GetInputNum() - other_inputs_size;
  auto input_num_on_op = op_desc->GetInputsSize();
  if (input_num_on_op != input_shapes_num) {
    GELOGE(ge::PARAM_INVALID, "Input num on op %s is %zu, input num on context is %zu, not match.",
           op_desc->GetName().c_str(), input_num_on_op, input_shapes_num);
    return ge::PARAM_INVALID;
  }
  auto infer_range_context = reinterpret_cast<InferShapeRangeContext *>(context);
  auto input_shape_start_pos = other_inputs_size;

  for (size_t i = 0U; i < input_shapes_num; ++i) {
    auto input_shape_min = infer_range_context->GetInputShapeRange(input_shape_start_pos + i)->GetMin();
    auto input_shape_max = infer_range_context->GetInputShapeRange(input_shape_start_pos + i)->GetMax();
    GE_ASSERT_EQ(input_shape_min->GetDimNum(), input_shape_max->GetDimNum());
    std::vector<int64_t> shape_dims;
    std::vector<std::pair<int64_t, int64_t>> range;
    for (size_t j = 0U; j < input_shape_min->GetDimNum(); ++j) {
      auto dim = (input_shape_min->GetDim(j) == input_shape_max->GetDim(j)) ? input_shape_min->GetDim(j) : -1;
      shape_dims.emplace_back(dim);
      range.emplace_back(std::pair<int64_t, int64_t>(input_shape_min->GetDim(j), input_shape_max->GetDim(j)));
    }
    const auto &input_desc = op_desc->MutableInputDesc(i);
    input_desc->SetOriginShape(ge::GeShape(shape_dims));
    input_desc->SetShape(ge::GeShape(shape_dims));
    input_desc->SetOriginShapeRange(range);
    input_desc->SetShapeRange(range);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateOutputShapeRangeToContext(const ge::OpDescPtr &op_desc, KernelContext *context) {
  auto output_num = context->GetOutputNum();
  size_t size_of_output = 0U;
  GE_ASSERT_TRUE(
      !ge::MulOverflow(static_cast<size_t>(op_desc->GetOutputsSize()), kShapeRangeOutputOfNode, size_of_output));
  if (size_of_output != output_num) {
    GELOGE(ge::PARAM_INVALID, "Output num on op %s is %zu, output num on context is %zu, not match.",
           op_desc->GetName().c_str(), size_of_output, output_num);
    return ge::PARAM_INVALID;
  }

  auto infer_range_context = reinterpret_cast<InferShapeRangeContext *>(context);
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    std::vector<std::pair<int64_t, int64_t>> range;
    GE_CHK_STATUS_RET(op_desc->GetOutputDesc(i).GetShapeRange(range), "Fail to get output shape range from op desc.");
    GE_ASSERT_NOTNULL(infer_range_context->GetOutputShapeRange(i));
    infer_range_context->GetOutputShapeRange(i)->GetMin()->SetDimNum(range.size());
    infer_range_context->GetOutputShapeRange(i)->GetMax()->SetDimNum(range.size());
    for (size_t j = 0U; j < range.size(); ++j) {
      infer_range_context->GetOutputShapeRange(i)->GetMin()->SetDim(j, range[j].first);
      infer_range_context->GetOutputShapeRange(i)->GetMax()->SetDim(j, range[j].second);
    }
  }
  return ge::GRAPH_SUCCESS;
}
} // namespace
ge::graphStatus BuildCreateOpOutputs(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto compute_node_chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(compute_node_chain);
  auto op = new (std::nothrow) ge::Operator();
  GE_ASSERT_NOTNULL(op);
  compute_node_chain->SetWithDefaultDeleter(op);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOpFromBuffer(KernelContext *context) {
   // at least need 2 inputs, including buffer and buffer size.
  auto input_num = context->GetInputNum();
  if (input_num < static_cast<size_t>(CreateOpFromBufferInputIndex::kInputNum)) {
    return ge::GRAPH_FAILED;
  }

  // 1. prepare input_param operator, update input shape from input to operator
  auto operator_buffer_index = static_cast<size_t>(CreateOpFromBufferInputIndex::kOpBuffer);
  auto operator_buffer = context->GetInputPointer<ContinuousVector>(operator_buffer_index);
  if (operator_buffer == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto operator_buffer_size_index = static_cast<size_t>(CreateOpFromBufferInputIndex::kOpBufferSize);
  auto operator_buffer_size = context->GetInputValue<size_t>(operator_buffer_size_index);
  ge::OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET_NOLOG(KernelCompatibleUtils::UnSerializeOpDesc(operator_buffer, operator_buffer_size, op_desc));
  GE_ASSERT_NOTNULL(op_desc);
  // operator GetInputConstData require there is node object on operator
  auto dummy_graph = ge::MakeShared<ge::ComputeGraph>("dummy");
  GE_ASSERT_NOTNULL(dummy_graph);
  auto dummy_node = dummy_graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(dummy_node);
  auto out_op = context->GetOutputPointer<ge::Operator>(0U);
  GE_ASSERT_NOTNULL(out_op);
  *out_op = ge::OpDescUtils::CreateOperatorFromNode(dummy_node);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateOpFromBuffer).RunFunc(CreateOpFromBuffer).OutputsCreator(BuildCreateOpOutputs);

ge::graphStatus BuildFindCompatibleInferShapeFuncOutputs(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  auto infer_func = new (std::nothrow)ge::InferShapeFunc();
  GE_ASSERT_NOTNULL(infer_func);

  chain->SetWithDefaultDeleter(infer_func);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FindCompatibleInferShapeFunc(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(0);
  auto infer_fun_ptr = context->GetOutputPointer<ge::InferShapeFunc>(0);
  if (node_type == nullptr || infer_fun_ptr == nullptr) {
    GELOGE(ge::FAILED, "Failed to find infer shape kernel, node type %s", node_type);
    return ge::GRAPH_FAILED;
  }
  const auto &op_funcs = ge::OperatorFactoryImpl::GetInferShapeFunc(node_type);
  if (op_funcs == nullptr) {
    GELOGE(ge::FAILED, "Failed to find v1 infer shape func, node type %s", node_type);
    return ge::GRAPH_FAILED;
  }
  *infer_fun_ptr = op_funcs;
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FindCompatibleInferShapeFunc)
    .RunFunc(FindCompatibleInferShapeFunc)
    .OutputsCreator(BuildFindCompatibleInferShapeFuncOutputs);

REGISTER_KERNEL(FindCompatibleInferShapeRangeFunc)
    .RunFunc(FindCompatibleInferShapeFunc)
    .OutputsCreator(BuildFindCompatibleInferShapeFuncOutputs);

ge::graphStatus CompatibleInferShape(KernelContext *context) {
  // check input num
  // For v1 infershape kernel, at least need 3 inputs, including infer_func and operator.
  auto input_num = context->GetInputNum();
  if (input_num < static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum)) {
    return ge::GRAPH_FAILED;
  }

  // 1. prepare input_param operator, update input shape from input to operator
  auto op_index = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kOperator);
  auto op = context->GetInputValue<ge::Operator *>(op_index);
  GE_ASSERT_NOTNULL(op);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  GE_ASSERT_NOTNULL(op_desc);
  GE_CHK_STATUS_RET(UpdateInputShapeToOpDesc(context, op_desc), "Fail to update input shapes to Operator.");

  std::vector<ge::GeTensorPtr> tensor_holder;
  if (!op_desc->GetOpInferDepends().empty()) {
    auto callback = [&context, &tensor_holder](const ge::ConstNodePtr &node, const size_t index,
                                               ge::GeTensorPtr &tensor) {
      (void) node;
      auto infer_shape_context = reinterpret_cast<InferShapeContext *>(context);
      auto input_start_pos = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum);
      auto shape_tensor = infer_shape_context->GetInputTensor(index + input_start_pos);
      GE_CHECK_NOTNULL(shape_tensor);
      return KernelCompatibleUtils::ConvertRTTensorToGeTensor(shape_tensor, tensor, tensor_holder);
    };
    ge::OpDescUtils::SetCallbackGetConstInputFuncToOperator(*op, callback);
  }
  
  // 2.get infer_func and infer
  auto op_infer_fun_index = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInferFunc);
  auto op_infer_fun = context->GetInputValue<ge::InferShapeFunc *>(op_infer_fun_index);
  GE_ASSERT_NOTNULL(op_infer_fun);
  auto ret = (*op_infer_fun)(*op);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ret, "Fail to infer shape on op %s.", op_desc->GetName().c_str());
    return ret;
  }
  
  // 3.update output shape from operator to context
  GE_CHK_STATUS_RET(UpdateOutputShapeToContext(op_desc, context), "Fail to update output shapes to Context.");

  // 4.expand dims and tranformer output shapes
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  GE_CHK_STATUS_RET(TransformAllOutputsShape(compute_node_info, context), "Fail to transfrom output shapes.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompatibleInferShapeRange(KernelContext *context) {
  // check input num
  // For v1 infershape kernel, at least need 3 inputs, including infer_func and operator.
  auto input_num = context->GetInputNum();
  if (input_num < static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum)) {
    GELOGE(ge::PARAM_INVALID,
           "input num only has %zu inputs now, at least need %zu inputs, including infer_func and operator.", input_num,
           static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum));
    return ge::GRAPH_FAILED;
  }

  // 1. prepare input_param operator, update input shape from input to operator
  auto op_index = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kOperator);
  auto op = context->GetInputValue<ge::Operator *>(op_index);
  GE_ASSERT_NOTNULL(op);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  GE_ASSERT_NOTNULL(op_desc);
  GE_CHK_STATUS_RET(UpdateInputShapeRangeToOpDesc(context, op_desc), "Fail to update input shapes to Operator.");

  std::vector<ge::GeTensorPtr> tensor_holder;
  if (!op_desc->GetOpInferDepends().empty()) {
    auto callback = [&context, &tensor_holder](const ge::ConstNodePtr &node, const size_t index,
                                               ge::GeTensorPtr &tensor) {
      (void) node;
      auto infer_range_context = reinterpret_cast<InferShapeRangeContext *>(context);
      auto input_start_pos = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInputNum);
      // 此处执行时,上一个节点是CreateTensorRangesAndShapeRanges，这个kernel里面保证了min tensor与max tensor时相等的，因此只需要
      // 将其任意一个用来产生GeTensor即可
      auto tensor_range = infer_range_context->GetInputTensorRange(index + input_start_pos);
      GE_CHECK_NOTNULL(tensor_range);
      auto shape_range_tensor = tensor_range->GetMin();
      GE_CHECK_NOTNULL(shape_range_tensor);
      return KernelCompatibleUtils::ConvertRTTensorToGeTensor(shape_range_tensor, tensor, tensor_holder);
    };
    ge::OpDescUtils::SetCallbackGetConstInputFuncToOperator(*op, callback);
  }

  // 2.get infer_func and infer
  auto op_infer_fun_index = static_cast<size_t>(CompatibleInferShapeOrRangeInputIndex::kInferFunc);
  auto op_infer_fun = context->GetInputValue<ge::InferShapeFunc *>(op_infer_fun_index);
  GE_ASSERT_NOTNULL(op_infer_fun);
  auto ret = (*op_infer_fun)(*op);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ret, "Fail to infer shape on op %s.", op_desc->GetName().c_str());
    return ret;
  }

  // 3.update output shape from operator to context
  GE_CHK_STATUS_RET(UpdateOutputShapeRangeToContext(op_desc, context),
                    "Fail to update output shapes range to Context.");

  // 4.expand dims and tranformer output shapes
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  GE_CHK_STATUS_RET(TransformAllOutputsMaxShape(compute_node_info, context), "Fail to transfrom output shapes.");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(CompatibleInferShape)
    .RunFunc(CompatibleInferShape)
    .OutputsCreator(BuildInferShapeOutputs)
    .TracePrinter(CompatibleInferShapeKernelTrace);
REGISTER_KERNEL(CompatibleInferShapeRange)
    .RunFunc(CompatibleInferShapeRange)
    .OutputsCreator(BuildInferShapeRangeOutputs)
    .TracePrinter(CompatibleInferShapeRangeKernelTrace);
}  // namespace kernel
}  // namespace gert
