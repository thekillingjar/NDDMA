/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/shape_inference.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "graph/compiler_def.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/transformer_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "common/checker.h"
#include "graph/utils/inference_rule.h"

namespace gert {
namespace {
using Index = struct InputIndex {
  size_t input_index;
  size_t invalid_index_num;
};
bool IsInputDescValid(const ge::GeTensorDesc &input_desc, size_t &invalid_index_num) {
  if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
    if (invalid_index_num < std::numeric_limits<size_t>::max()) {
      invalid_index_num++;
    }
    return false;
  }
  return true;
}

void GetStorageShape(const ge::GeTensorDesc &input_desc, gert::StorageShape &storage_shape) {
  const auto &dims = input_desc.GetOriginShape().GetDims();
  for (const auto &dim : dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
}

void GetMinMaxStorageShape(const ge::GeTensorDesc &input_desc, gert::StorageShape &min_storage_shape,
                           gert::StorageShape &max_storage_shape) {
  auto ge_shape = input_desc.GetShape();
  if (ge_shape.IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    (void)input_desc.GetShapeRange(shape_range);
    for (size_t j = 0UL; j < shape_range.size(); ++j) {
      (void)min_storage_shape.MutableOriginShape().AppendDim(shape_range[j].first);
      (void)min_storage_shape.MutableStorageShape().AppendDim(shape_range[j].first);
      (void)max_storage_shape.MutableOriginShape().AppendDim(shape_range[j].second);
      (void)max_storage_shape.MutableStorageShape().AppendDim(shape_range[j].second);
    }
  } else {
    const auto &dims = input_desc.GetOriginShape().GetDims();
    for (const auto &dim : dims) {
      (void)min_storage_shape.MutableOriginShape().AppendDim(dim);
      (void)min_storage_shape.MutableStorageShape().AppendDim(dim);
      (void)max_storage_shape.MutableOriginShape().AppendDim(dim);
      (void)max_storage_shape.MutableStorageShape().AppendDim(dim);
    }
  }
}

ge::graphStatus GetTensorAddress(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                 const Index &index, TensorAddress &address,
                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  GE_ASSERT_NOTNULL(space_registry);
  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  const size_t instance_index = index.input_index - index.invalid_index_num;
  // check valid map
  const auto valid_op_ir_map = ge::OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  if (valid_op_ir_map.empty()) {
    return ge::GRAPH_PARAM_INVALID;
  }
  size_t ir_index;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, instance_index, ir_index),
                          "[Get][InputIrIndexByInstanceIndex] failed, op[%s], instance index[%zu], input_index[%zu]",
                          op_desc->GetName().c_str(), instance_index, index.input_index);
  GE_ASSERT_NOTNULL(functions);
  if (functions->IsInputDataDependency(ir_index)) {
    ge_tensors_holder[index.input_index] = ge::ComGraphMakeUnique<ge::Tensor>();
    GE_ASSERT_NOTNULL(ge_tensors_holder[index.input_index], "Create ge tensor holder inputs failed.");
    const auto index_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(index.input_index));
    if (op.GetInputConstData(index_name.c_str(), *(ge_tensors_holder[index.input_index].get())) == ge::GRAPH_SUCCESS) {
      address = ge_tensors_holder[index.input_index]->GetData();
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool IsTensorDependencyValid(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                             const size_t input_index, const size_t invalid_index_num) {
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  GE_ASSERT_NOTNULL(space_registry);
  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  GE_ASSERT_NOTNULL(functions);
  const size_t instance_index = input_index - invalid_index_num;
  size_t ir_index;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, instance_index, ir_index),
                          "[Get][InputIrIndexByInstanceIndex] failed, op[%s], instance index[%zu], input_index[%zu]",
                          op_desc->GetName().c_str(), instance_index, input_index);
  if (functions->IsInputDataDependency(ir_index)) {
    ge::Tensor data;
    const auto index_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(input_index));
    if (op.GetInputConstData(index_name.c_str(), data) == ge::GRAPH_SUCCESS) {
      return true;
    } else {
      return false;
    }
  }
  return true;
}

ge::graphStatus GetTensorHolder(const ge::GeTensorDesc &input_desc, const gert::StorageShape &storage_shape,
                                TensorAddress address, std::unique_ptr<uint8_t[]> &tensor_holder) {
  tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
  GE_ASSERT_NOTNULL(tensor_holder, "Create context holder inputs failed.");
  if (address == nullptr) {
    new (tensor_holder.get())
        gert::Tensor(storage_shape,
                     {input_desc.GetOriginFormat(), input_desc.GetFormat(), {}},
                     input_desc.GetDataType());
  } else {
    new (tensor_holder.get())
        gert::Tensor(storage_shape,
                     {input_desc.GetOriginFormat(), input_desc.GetFormat(), {}},
                     gert::kOnHost, input_desc.GetDataType(), address);
  }
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus ConstructCompileKernelContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                    std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                    std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    if (!IsInputDescValid(op_desc->GetInputDesc(static_cast<uint32_t>(i)), invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }
    gert::StorageShape storage_shape;
    GetStorageShape(op_desc->GetInputDesc(static_cast<uint32_t>(i)), storage_shape);
    // init tensor address, if can not get const tensor input, set it to nullptr
    TensorAddress address = nullptr;
    Index index;
    index.input_index = i;
    index.invalid_index_num = invalid_index_num;
    auto status = GetTensorAddress(op, op_desc, index, address, ge_tensors_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    std::unique_ptr<uint8_t[]> tensor_holder;
    status = GetTensorHolder(op_desc->GetInputDesc(static_cast<uint32_t>(i)), storage_shape, address, tensor_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    (void)inputs.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructInferShapeContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                 std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextInputs(op, op_desc, inputs, ge_tensors_holder));
  // set infer shape_func to NULL
  (void)inputs.emplace_back(nullptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructInferShapeRangeContextInputs(
    const ge::Operator &op, const ge::OpDescPtr &op_desc, std::vector<std::unique_ptr<uint8_t[]>> &inputs,
    std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder,
    std::vector<std::pair<gert::Tensor, gert::Tensor>> &input_tensor_ranges_holder) {
  size_t invalid_index_num = 0UL;
  GE_ASSERT(input_tensor_ranges_holder.size() == op_desc->GetAllInputsSize());
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    const auto &input_desc = op_desc->GetInputDesc(static_cast<uint32_t>(i));
    if (!IsInputDescValid(input_desc, invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }

    GetMinMaxStorageShape(op_desc->GetInputDesc(static_cast<uint32_t>(i)),
                          input_tensor_ranges_holder[i].first.GetShape(),
                          input_tensor_ranges_holder[i].second.GetShape());
    input_tensor_ranges_holder[i].first.SetOriginFormat(input_desc.GetOriginFormat());
    input_tensor_ranges_holder[i].first.SetStorageFormat(input_desc.GetFormat());
    input_tensor_ranges_holder[i].first.SetDataType(input_desc.GetDataType());
    input_tensor_ranges_holder[i].second.SetOriginFormat(input_desc.GetOriginFormat());
    input_tensor_ranges_holder[i].second.SetStorageFormat(input_desc.GetFormat());
    input_tensor_ranges_holder[i].second.SetDataType(input_desc.GetDataType());

    // init tensor address, if can not get const tensor input, set it to nullptr
    TensorAddress address = nullptr;
    Index index;
    index.input_index = i;
    index.invalid_index_num = invalid_index_num;
    const auto status = GetTensorAddress(op, op_desc, index, address, ge_tensors_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    std::unique_ptr<uint8_t[]> tensor_range_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::TensorRange));
    GE_ASSERT_NOTNULL(tensor_range_holder, "Create context holder inputs failed.");

    if (address != nullptr) {
      (void) input_tensor_ranges_holder[i].first.MutableTensorData().SetAddr(address, nullptr);
      (void) input_tensor_ranges_holder[i].first.MutableTensorData().SetPlacement(gert::kOnHost);
      (void) input_tensor_ranges_holder[i].second.MutableTensorData().SetAddr(address, nullptr);
      input_tensor_ranges_holder[i].second.MutableTensorData().SetPlacement(gert::kOnHost);
    }
    new (tensor_range_holder.get())
        gert::TensorRange(&input_tensor_ranges_holder[i].first, &input_tensor_ranges_holder[i].second);
    (void)inputs.emplace_back(std::move(tensor_range_holder));
  }
  // set infer shape_func to NULL
  (void)inputs.emplace_back(nullptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructInferShapeContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                 std::vector<std::unique_ptr<uint8_t[]>> &min_inputs,
                                                 std::vector<std::unique_ptr<uint8_t[]>> &max_inputs,
                                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    if (!IsInputDescValid(op_desc->GetInputDesc(static_cast<size_t>(i)), invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }
    gert::StorageShape min_storage_shape;
    gert::StorageShape max_storage_shape;
    GetMinMaxStorageShape(op_desc->GetInputDesc(static_cast<size_t>(i)), min_storage_shape, max_storage_shape);

    // init tensor address, if can not get const tensor input, set it to nullptr
    TensorAddress address = nullptr;
    Index index;
    index.input_index = i;
    index.invalid_index_num = invalid_index_num;
    auto status = GetTensorAddress(op, op_desc, index, address, ge_tensors_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    std::unique_ptr<uint8_t[]> min_tensor_holder;
    status = GetTensorHolder(op_desc->GetInputDesc(static_cast<size_t>(i)), min_storage_shape, address, min_tensor_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    std::unique_ptr<uint8_t[]> max_tensor_holder;
    status = GetTensorHolder(op_desc->GetInputDesc(static_cast<size_t>(i)), max_storage_shape, address, max_tensor_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    (void)min_inputs.emplace_back(std::move(min_tensor_holder));
    (void)max_inputs.emplace_back(std::move(max_tensor_holder));
  }
  // set infer shape_func to NULL
  (void)min_inputs.emplace_back(nullptr);
  (void)max_inputs.emplace_back(nullptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructCompileKernelContextOutputs(const ge::OpDescPtr &op_desc,
                                                     std::vector<std::unique_ptr<uint8_t[]>> &outputs) {
  auto size = op_desc->GetAllOutputsDescSize();
  while (size-- > 0) {
    auto tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed, op[%s]", op_desc->GetName().c_str());
    (void)outputs.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructInferShapeRangeContextOutputs(
    const ge::OpDescPtr &op_desc, std::vector<std::unique_ptr<uint8_t[]>> &outputs,
    std::vector<std::pair<StorageShape, StorageShape>> &output_range_holder) {
  for (size_t i = 0UL; i < op_desc->GetAllOutputsDescSize(); i++) {
    auto tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(Range<Shape>));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed, op[%s]", op_desc->GetName().c_str());
    reinterpret_cast<Range<Shape> *>(tensor_holder.get())->SetMin(&(output_range_holder[i].first.MutableOriginShape()));
    reinterpret_cast<Range<Shape> *>(tensor_holder.get())
        ->SetMax(&(output_range_holder[i].second.MutableOriginShape()));
    (void)outputs.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateOpDescOutShape(const ge::OpDescPtr &op_desc, gert::InferShapeContext *infer_shape_ctx) {
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
    auto &dst_out_shape = op_desc->MutableOutputDesc(static_cast<size_t>(index))->MutableShape();
    const auto *shape = infer_shape_ctx->GetOutputShape(index);
    GE_ASSERT_NOTNULL(shape);
    dst_out_shape.SetDimNum(shape->GetDimNum());
    for (size_t dim = 0UL; dim < shape->GetDimNum(); dim++) {
      (void)dst_out_shape.SetDim(dim, shape->GetDim(dim));
    }
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetOriginShape(dst_out_shape);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRange(std::vector<std::pair<int64_t, int64_t>> &shape_range,
                         const gert::Shape * const min_shape, const gert::Shape * const max_shape) {
  for (size_t i = 0UL; i < min_shape->GetDimNum(); ++i) {
    GELOGD("min dim:%ld, max dim:%ld", min_shape->GetDim(i), max_shape->GetDim(i));
    if (max_shape->GetDim(i) != -1) {
      GE_CHECK_LE(min_shape->GetDim(i), max_shape->GetDim(i));
    }
    (void)shape_range.emplace_back(std::make_pair(min_shape->GetDim(i), max_shape->GetDim(i)));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateOpDescOutShapeRange(const ge::OpDescPtr &op_desc, gert::InferShapeContext *min_ctx,
                                          gert::InferShapeContext *max_ctx) {
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
    auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
    auto ge_shape = output_desc->GetShape();
    if (ge_shape.IsUnknownShape()) {
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      const auto *min_shape = min_ctx->GetOutputShape(index);
      const auto *max_shape = max_ctx->GetOutputShape(index);
      GE_ASSERT_NOTNULL(min_shape);
      GE_ASSERT_NOTNULL(max_shape);
      GELOGD("min dim num:%zu, max dim num:%zu", min_shape->GetDimNum(), max_shape->GetDimNum());
      GE_RETURN_WITH_LOG_IF_TRUE((min_shape->GetDimNum()) != (max_shape->GetDimNum()));
      const auto ret = AddRange(shape_range, min_shape, max_shape);
      if (ret != ge::GRAPH_SUCCESS) {
        return ret;
      }
      (void)output_desc->SetShapeRange(shape_range);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateOpDescOutShapeRange(const ge::OpDescPtr &op_desc,
                                          gert::InferShapeRangeContext *infer_shape_range_ctx) {
  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); ++i) {
    const auto &output_tensor = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    const auto out_range = infer_shape_range_ctx->GetOutputShapeRange(i);
    GE_ASSERT_NOTNULL(out_range, "out range is nullptr.");
    GE_ASSERT_NOTNULL(out_range->GetMax(), "out range max is nullptr.");
    GE_ASSERT_NOTNULL(out_range->GetMin(), "out range min is nullptr.");
    for (size_t j = 0UL; j < out_range->GetMax()->GetDimNum(); ++j) {
      (void)shape_range.emplace_back(std::make_pair(out_range->GetMin()->GetDim(j), out_range->GetMax()->GetDim(j)));
    }
    (void)output_tensor->SetShapeRange(shape_range);
  }
  return ge::GRAPH_SUCCESS;
}

void ConstructDataTypeContextInputs(const ge::OpDescPtr &op_desc, std::vector<void *> &inputs) {
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const auto &compile_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (compile_tensor == nullptr) {
      GELOGD("OpDesc[%s]type[%s], input desc[%zu] is nullptr, skip constructing rt2 ctx for it.", op_desc->GetNamePtr(),
             op_desc->GetTypePtr(), i);
      continue;
    }
    (void)inputs.emplace_back(reinterpret_cast<void *>(compile_tensor->GetDataType()));
  }
}

void ConstructDataTypeContextOutputs(const ge::OpDescPtr &op_desc, std::vector<void *> &outputs) {
  for (size_t i = 0UL; i < op_desc->GetAllOutputsDescSize(); i++) {
    const auto &compile_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    (void)outputs.emplace_back(reinterpret_cast<void *>(compile_tensor.GetDataType()));
  }
}

// inputs layout is input tensors
std::vector<void *> GetInputs(const std::vector<std::unique_ptr<uint8_t[]>> &inputs_holders) {
  std::vector<void *> inputs;
  inputs.reserve(inputs_holders.size());
  for (const auto &input_holder : inputs_holders) {
    (void)inputs.emplace_back(input_holder.get());
  }
  return inputs;
}

std::vector<void *> GetInputs(const ge::Operator &op, const std::vector<std::unique_ptr<uint8_t[]>> &inputs_holders) {
  std::vector<void *> inputs;
  inputs.reserve(inputs_holders.size() + 1UL);
  for (const auto &input_holder : inputs_holders) {
    (void)inputs.emplace_back(input_holder.get());
  }
  // inputs layout is input tensors + infer func + inference context ptr
  (void)inputs.emplace_back(op.GetInferenceContext().get());
  return inputs;
}

std::vector<void *> GetOutputs(const std::vector<std::unique_ptr<uint8_t[]>> &outputs_holders) {
  std::vector<void *> outputs;
  outputs.reserve(outputs_holders.size());
  for (const auto &output_holder : outputs_holders) {
    (void)outputs.emplace_back(output_holder.get());
  }
  return outputs;
}

bool NeedInferShapeRange(const ge::Operator &op, const ge::OpDescPtr &op_desc) {
  bool need_infer = false;
  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const auto &input_desc = op_desc->GetInputDesc(static_cast<size_t>(i));
    if (!IsInputDescValid(input_desc, invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }
    auto ge_shape = input_desc.GetShape();
    if (ge_shape.IsUnknownShape()) {
      std::vector<std::pair<int64_t, int64_t>> shape_range;
      need_infer = true;
      (void)input_desc.GetShapeRange(shape_range);
      if (shape_range.size() == 0UL) {
        GELOGD("No need to infer shape range, because shape is unknown shape but no shape range, input[%zu].", i);
        return false;
      }
      if (!IsTensorDependencyValid(op, op_desc, i, invalid_index_num)) {
        GELOGD("No need to infer shape range, because dependency tensor is not const, input[%zu].", i);
        return false;
      }
    }
  }
  return need_infer;
}

ge::graphStatus InferShapeRangeCustom(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                      OpImplRegisterV2::InferShapeRangeKernelFunc const infer_shape_range) {
  std::vector<std::unique_ptr<uint8_t[]>> inputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> outputs_holder;
  std::vector<std::unique_ptr<ge::Tensor>> ge_tensors_holder;
  std::vector<std::pair<gert::Tensor, gert::Tensor>> input_tensor_range_holder;
  std::vector<std::pair<StorageShape, StorageShape>> output_range_holder;

  ge_tensors_holder.resize(op_desc->GetAllInputsSize());
  input_tensor_range_holder.resize(static_cast<size_t>(op_desc->GetAllInputsSize()));
  output_range_holder.resize(static_cast<size_t>(op_desc->GetAllOutputsDescSize()));
  GE_ASSERT_GRAPH_SUCCESS(
      ConstructInferShapeRangeContextInputs(op, op_desc, inputs_holder, ge_tensors_holder, input_tensor_range_holder),
      "[Construct][InferShapeContextInputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ConstructInferShapeRangeContextOutputs(op_desc, outputs_holder, output_range_holder),
                          "[Construct][InferShapeContextOutputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  const auto kernel_context_holder = gert::KernelRunContextBuilder()
      .Inputs(GetInputs(op, inputs_holder)).Outputs(GetOutputs(outputs_holder)).Build(op_desc);
  auto infer_shape_range_ctx = reinterpret_cast<gert::InferShapeRangeContext *>(kernel_context_holder.context_);
  const auto ret = infer_shape_range(infer_shape_range_ctx);
  GE_CHK_STATUS_RET(ret, "[Call][InferShapeRange] failed, ret[%d]", ret);
  (void)UpdateOpDescOutShapeRange(op_desc, infer_shape_range_ctx);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRangeAutomaticly(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                           OpImplRegisterV2::InferShapeKernelFunc const infer_shape) {
  GELOGD("Need to infer shape range, op[%s]", op_desc->GetName().c_str());
  std::vector<std::unique_ptr<uint8_t[]>> min_inputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> max_inputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> min_outputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> max_outputs_holder;
  std::vector<std::unique_ptr<ge::Tensor>> ge_tensors_holder;
  ge_tensors_holder.resize(op_desc->GetAllInputsSize());
  GE_ASSERT_GRAPH_SUCCESS(
      ConstructInferShapeContextInputs(op, op_desc, min_inputs_holder, max_inputs_holder, ge_tensors_holder),
      "[Construct][InferShapeRangeAutomaticly] failed, op_desc[%s]", op_desc->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextOutputs(op_desc, min_outputs_holder),
                          "[Construct][InferShapeRangeAutomaticly] failed, op_desc[%s]", op_desc->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextOutputs(op_desc, max_outputs_holder),
                          "[Construct][InferShapeRangeAutomaticly] failed, op_desc[%s]", op_desc->GetName().c_str());
  // min output
  const auto min_kernel_context_holder = gert::KernelRunContextBuilder()
      .Inputs(GetInputs(op, min_inputs_holder)).Outputs(GetOutputs(min_outputs_holder)).Build(op_desc);
  auto min_infer_shape_ctx = reinterpret_cast<gert::InferShapeContext *>(min_kernel_context_holder.context_);
  auto ret = infer_shape(min_infer_shape_ctx);
  GE_CHK_STATUS_RET(ret, "[InferV2][MinShape] failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str(), ret);
  // max output
  const auto max_kernel_context_holder = gert::KernelRunContextBuilder()
      .Inputs(GetInputs(op, max_inputs_holder)).Outputs(GetOutputs(max_outputs_holder)).Build(op_desc);
  auto max_infer_shape_ctx = reinterpret_cast<gert::InferShapeContext *>(max_kernel_context_holder.context_);
  ret = infer_shape(max_infer_shape_ctx);
  GE_CHK_STATUS_RET(ret, "[InferV2][MaxShape] failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str(), ret);
  ret = UpdateOpDescOutShapeRange(op_desc, min_infer_shape_ctx, max_infer_shape_ctx);
  return ret;
}

ge::graphStatus UpdateOpDescOutFormat(const ge::OpDescPtr &op_desc, gert::InferFormatContext *infer_format_ctx) {
  size_t in_index = 0UL;
  for (size_t index = 0UL; index < op_desc->GetInputsSize(); index++) {
    const auto desc = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
    if (desc == nullptr) {
      continue;
    }
    const auto format = infer_format_ctx->GetInputFormat(in_index++);
    GE_ASSERT_NOTNULL(format);
    desc->SetOriginFormat(format->GetOriginFormat());
    desc->SetFormat(format->GetStorageFormat());
  }

  size_t out_index = 0UL;
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); index++) {
    const auto desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
    if (desc == nullptr) {
      continue;
    }
    const auto format = infer_format_ctx->GetOutputFormat(out_index++);
    GE_ASSERT_NOTNULL(format);
    desc->SetOriginFormat(format->GetOriginFormat());
    desc->SetFormat(format->GetStorageFormat());
  }
  return ge::GRAPH_SUCCESS;
}


ge::graphStatus InferShapeByRegisteredFuncOrRule(const OpImplKernelRegistry::OpImplFunctionsV2 *functions,
                                                 const ge::OpDescPtr &op_desc,
                                                 gert::InferShapeContext *infer_shape_ctx) {
  if (functions != nullptr && functions->infer_shape != nullptr) {
    if (functions->IsOutputShapeDependOnCompute()) {
      GELOGD("OpDesc %s(%s) is third class operator", op_desc->GetNamePtr(), op_desc->GetTypePtr());
      (void) ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE,
                                   static_cast<int64_t>(ge::DEPEND_SHAPE_RANGE));
    }
    GELOGD("Infer shape for %s[%s] by registered func", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return functions->infer_shape(infer_shape_ctx);
  }
  const auto shape_infer_rule = ge::ShapeInferenceRule::FromOpDesc(op_desc);
  if (shape_infer_rule == nullptr) {
    REPORT_INNER_ERR_MSG("EZ9999",
                         "Can not find infer_shape func of node %s[%s]. Please confirm whether the op_proto shared "
                         "library (.so) has been loaded "
                         "successfully, and that you have already developed the infer_shape func.",
                         op_desc->GetNamePtr(), op_desc->GetTypePtr());
    GELOGE(ge::GRAPH_FAILED,
           "Can not find infer_shape func of node %s[%s]. Please confirm whether the op_proto shared library (.so) "
           "has been loaded "
           "successfully, and that you have already developed the infer_shape func.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  if (!shape_infer_rule->IsValid()) {
    REPORT_INNER_ERR_MSG(
        "EZ9999",
        "No infer shape func registered for node %s[%s], and inference rule: %s is set but failed to parse: %s.",
        op_desc->GetNamePtr(), op_desc->GetTypePtr(), ge::InferenceRule::GetInferenceRule(op_desc).c_str(),
        shape_infer_rule->Error().c_str());
    GELOGE(ge::GRAPH_FAILED,
           "No infer shape func registered for node %s[%s], and inference rule: %s is set but failed to parse: %s.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), ge::InferenceRule::GetInferenceRule(op_desc).c_str(),
           shape_infer_rule->Error().c_str());
    return ge::GRAPH_FAILED;
  }
  GELOGD("Infer shape for %s[%s] by inference rule", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return shape_infer_rule->InferOnCompile(infer_shape_ctx);
}

ge::graphStatus InferDtypeByRegisteredFuncOrRule(const OpImplKernelRegistry::OpImplFunctionsV2 *functions,
                                                 const ge::OpDescPtr &op_desc,
                                                 gert::InferDataTypeContext *infer_dtype_ctx) {
  if (functions != nullptr && functions->infer_datatype != nullptr) {
    GELOGD("Infer dtype for %s[%s] by registered func", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return functions->infer_datatype(infer_dtype_ctx);
  }
  const auto dtype_infer_rule = ge::DtypeInferenceRule::FromOpDesc(op_desc);
  if (dtype_infer_rule == nullptr) {
    REPORT_INNER_ERR_MSG("EZ9999",
                         "Can not find Node %s[%s] custom infer_datatype func. Please confirm whether the op_proto "
                         "shared library (.so) has been "
                         "loaded successfully, and that you have already developed the infer_datatype func or marked "
                         "the T-derivation rules on the IR.",
                         op_desc->GetNamePtr(), op_desc->GetTypePtr());
    GELOGE(ge::GRAPH_FAILED,
           "Can not find Node %s[%s] custom infer_datatype func. Please confirm whether the op_proto shared library "
           "(.so) has been "
           "loaded successfully, and that you have already developed the infer_datatype func or marked "
           "the T-derivation rules on the IR.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  if (!dtype_infer_rule->IsValid()) {
    REPORT_INNER_ERR_MSG(
        "EZ9999",
        "No infer dtype func registered for node %s[%s], and inference rule: %s is set but failed to parse: %s.",
        op_desc->GetNamePtr(), op_desc->GetTypePtr(), ge::InferenceRule::GetInferenceRule(op_desc).c_str(),
        dtype_infer_rule->Error().c_str());
    GELOGE(ge::GRAPH_FAILED,
           "No infer dtype func registered for node %s[%s], and inference rule: %s is set but failed to parse: %s.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), ge::InferenceRule::GetInferenceRule(op_desc).c_str(),
           dtype_infer_rule->Error().c_str());
    return ge::GRAPH_FAILED;
  }
  GELOGD("Infer dtype for %s[%s] by inference rule", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return dtype_infer_rule->InferDtype(infer_dtype_ctx);
}
}

ge::graphStatus InferShapeRangeOnCompile(const ge::Operator &op, const ge::OpDescPtr &op_desc) {
  if (!NeedInferShapeRange(op, op_desc)) {
    GELOGD("No need to infer shape range, op[%s]", op_desc->GetName().c_str());
    return ge::GRAPH_SUCCESS;
  }
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  GE_ASSERT_NOTNULL(space_registry);
  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  GE_ASSERT_NOTNULL(functions);
  if (functions->infer_shape_range != nullptr) {
    GELOGD("Op[%s], type[%s] use custom derivation strategy.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return InferShapeRangeCustom(op, op_desc, functions->infer_shape_range);
  }
  if (functions->infer_shape != nullptr) {
    GELOGD("Can not get infer shape range func op[%s], type[%s], will use an automatic derivation strategy.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return InferShapeRangeAutomaticly(op, op_desc, functions->infer_shape);
  }
  GELOGD("Skip infer shape range for node[%s], type[%s] as no infer shape range func.", op_desc->GetName().c_str(),
         op_desc->GetType().c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeOnCompile(const ge::Operator &op, const ge::OpDescPtr &op_desc) {
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  GE_ASSERT_NOTNULL(space_registry);

  ge::NodeShapeTransUtils transformer(op_desc);
  GE_CHK_BOOL_RET_STATUS(transformer.Init(), ge::GRAPH_FAILED, "Failed to init transformer for %s",
                         op_desc->GetNamePtr());
  GE_CHK_BOOL_RET_STATUS(transformer.CatchFormatAndShape(), ge::GRAPH_FAILED, "Failed to catch format and shape for %s",
                         op_desc->GetNamePtr());
  std::vector<std::unique_ptr<uint8_t[]>> inputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> outputs_holder;
  std::vector<std::unique_ptr<ge::Tensor>> ge_tensors_holder;
  ge_tensors_holder.resize(op_desc->GetAllInputsSize());
  auto ret = ConstructInferShapeContextInputs(op, op_desc, inputs_holder, ge_tensors_holder);
  if (ret == ge::GRAPH_PARAM_INVALID) {
    return ret;
  }
  GE_ASSERT_GRAPH_SUCCESS(ret, "[Construct][InferShapeContextInputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextOutputs(op_desc, outputs_holder),
                          "[Construct][InferShapeContextOutputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  const auto kernel_context_holder = gert::KernelRunContextBuilder()
                                         .Inputs(GetInputs(op, inputs_holder))
                                         .Outputs(GetOutputs(outputs_holder))
                                         .Build(op_desc);
  auto infer_shape_ctx = reinterpret_cast<gert::InferShapeContext *>(kernel_context_holder.context_);

  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());

  ret = InferShapeByRegisteredFuncOrRule(functions, op_desc, infer_shape_ctx);
  GE_CHK_STATUS_RET(ret, "[Call][InferShapeV2Func] failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str(), ret);

  GE_ASSERT_GRAPH_SUCCESS(UpdateOpDescOutShape(op_desc, infer_shape_ctx),
                          "UpdateOpDescOutShape failed, OutputShape is nullptr. op_desc[%s]",
                          op_desc->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(transformer.UpdateFormatAndShape(), ge::GRAPH_FAILED,
                         "Failed to update format and shape for %s", op_desc->GetNamePtr());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeOnCompile(const ge::OpDescPtr &op_desc) {
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  if (space_registry == nullptr) {
    GELOGW("Default space registry has not been initialized!");
    if (op_desc->IsSupportSymbolicInferDataType()) {
      return op_desc->SymbolicInferDataType();
    }
    GELOGW("Space_registry is null, neither Node %s[%s] not support symbolic infer datatype. Please declare symbol T "
           "on IR or check Space_registry.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }

  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  if ((functions == nullptr || functions->infer_datatype == nullptr) && op_desc->IsSupportSymbolicInferDataType()) {
    GELOGD("Infer dtype for %s[%s] by ir symbol", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return op_desc->SymbolicInferDataType();
  }

  std::vector<void *> inputs;
  std::vector<void *> outputs;
  ConstructDataTypeContextInputs(op_desc, inputs);
  ConstructDataTypeContextOutputs(op_desc, outputs);
  const auto kernel_context_holder = gert::KernelRunContextBuilder().Inputs(inputs).Outputs(outputs).Build(op_desc);
  const auto kernel_context = reinterpret_cast<gert::InferDataTypeContext *>(kernel_context_holder.context_);

  const ge::graphStatus ret = InferDtypeByRegisteredFuncOrRule(functions, op_desc, kernel_context);
  GE_CHK_STATUS_RET(ret, "[Check][InferDataType] result failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str(), ret);
  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); i++) {
    const auto &out_desc = op_desc->MutableOutputDesc(static_cast<size_t>(i));
    out_desc->SetDataType(kernel_context->GetOutputDataType(i));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferFormatOnCompile(const ge::Operator &op, const ge::OpDescPtr &op_desc) {
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  GE_ASSERT_NOTNULL(space_registry);
  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  if ((functions == nullptr) || (functions->infer_format_func == nullptr)) {
    REPORT_INNER_ERR_MSG("EZ9999",
                         "Can not find infer_format func of node %s[%s]. Please confirm whether the op_proto shared "
                         "library (.so) has been loaded "
                         "successfully, and that you have already developed the infer_format func.",
                         op_desc->GetNamePtr(), op_desc->GetTypePtr());
    GELOGE(
        ge::GRAPH_FAILED,
        "Can not find infer_format func of node %s[%s]. Please confirm whether the op_proto shared library (.so) has been loaded "
        "successfully, and that you have already developed the infer_format func.",
        op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }

  std::vector<std::unique_ptr<uint8_t[]>> inputs_holder;
  std::vector<std::unique_ptr<uint8_t[]>> outputs_holder;
  std::vector<std::unique_ptr<ge::Tensor>> ge_tensors_holder;
  ge_tensors_holder.resize(op_desc->GetAllInputsSize());
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextInputs(op, op_desc, inputs_holder, ge_tensors_holder),
                          "[Construct][InferFormatContextInputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextOutputs(op_desc, outputs_holder),
                          "[Construct][InferShapeContextOutputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  const auto kernel_context_holder = gert::KernelRunContextBuilder()
                                         .Inputs(GetInputs(inputs_holder))
                                         .Outputs(GetOutputs(outputs_holder))
                                         .Build(op_desc);
  const auto infer_format_ctx = reinterpret_cast<gert::InferFormatContext *>(kernel_context_holder.context_);
  const auto ret = functions->infer_format_func(infer_format_ctx);
  GE_CHK_STATUS_RET(ret, "[Call][InferFormatV2Func] failed, op_desc[%s], ret[%d]", op_desc->GetName().c_str(), ret);
  GE_ASSERT_GRAPH_SUCCESS(UpdateOpDescOutFormat(op_desc, infer_format_ctx), "UpdateOpDescOutFormat failed for op[%s]",
                          op_desc->GetName().c_str());
  return ge::GRAPH_SUCCESS;
}

bool IsInferFormatV2Registered(const ge::OpDescPtr &op_desc) {
  const auto *const space_registry =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  if (space_registry != nullptr) {
    const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
    if ((functions != nullptr) && (functions->infer_format_func != nullptr)) {
      return true;
    }
  }
  return false;
}

bool IsInferShapeV2Registered(const ge::OpDescPtr &op_desc) {
  const auto *const space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
    .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
    .get();
  if (space_registry == nullptr) {
    GELOGI("[%s][%s] Space registry is null.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }

  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  if (functions == nullptr) {
    GELOGI("[%s][%s] Registry functions is null.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }

  if ((functions->infer_shape == nullptr) && (ge::ShapeInferenceRule::FromOpDesc(op_desc) == nullptr)) {
    GELOGI(
      "[%s][%s] Infer shape is null.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }

  return true;
}


class CompileAdaptFunctionsRegister {
 public:
  CompileAdaptFunctionsRegister() {
    // only infer shape is necessary, as register all infer func in infer shape
    (void) ge::OperatorFactoryImpl::RegisterInferShapeV2Func(&gert::InferShapeOnCompile);
    (void) ge::OperatorFactoryImpl::RegisterInferShapeRangeFunc(&gert::InferShapeRangeOnCompile);
    (void) ge::OperatorFactoryImpl::RegisterInferDataTypeFunc(&gert::InferDataTypeOnCompile);
    (void) ge::OperatorFactoryImpl::RegisterInferFormatV2Func(&gert::InferFormatOnCompile);
    (void) ge::OperatorFactoryImpl::RegisterIsInferFormatV2RegisteredFunc(&gert::IsInferFormatV2Registered);
    (void) ge::OperatorFactoryImpl::RegisterIsInferShapeV2RegisteredFunc(&gert::IsInferShapeV2Registered);
  }
};
static CompileAdaptFunctionsRegister VAR_UNUSED g_register_adapt_funcs;
}  // namespace gert
