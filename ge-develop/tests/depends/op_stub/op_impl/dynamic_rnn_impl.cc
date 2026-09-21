/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dynamic_rnn_impl.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "framework/common/debug/ge_log.h"

#include "nlohmann/json.hpp"
#include "exe_graph/runtime/tiling_context.h"

namespace gert {
namespace {
constexpr int DEFAULT_SHAPE_LIST_SIZE = 3;
constexpr int DEFAULT_INDEX_TWO = 2;
constexpr int DEFAULT_RETURN = -2;
constexpr int DEFAULT_PARAS_INPUT_SIZE = 3;
constexpr int DEFAULT_XSHAPE_SIZE = 3;
constexpr int NUM_SIXTEEN = 16;
constexpr int NUM_FIFTEEN = 15;

int32_t GetRnnV3LibItem(const DynamicRNNV3CompileInfo *compile_info, const Shape x_shape) {
  for (const auto &tune_shape : compile_info->tune_shape_list) {
    if (tune_shape.size() < DEFAULT_SHAPE_LIST_SIZE) {
      return DEFAULT_RETURN;
    }
    if ((tune_shape[0] == -1) && (tune_shape[1] == -1)) {
      return static_cast<int32_t>(tune_shape[DEFAULT_INDEX_TWO]);
    }
    // todo 除法很贵，是否可以使用乘法代替？
    if ((tune_shape[0] == x_shape.GetDim(0)) &&
        (((tune_shape[1] + NUM_FIFTEEN) / NUM_SIXTEEN) == x_shape.GetDim(DEFAULT_INDEX_TWO))) {
      return static_cast<int32_t>(tune_shape[DEFAULT_INDEX_TWO]);
    }
  }
  return DEFAULT_RETURN;
}
}  // namespace
ge::graphStatus InferShapeForDynamicRNNV3(InferShapeContext *context) {
  auto x_shape = context->GetInputShape(0);
  auto w_shape = context->GetInputShape(1);
  if (x_shape == nullptr || w_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  int64_t stateSize = 0;
  // TODO need use optinal inputshape
  // auto project_shape = context->GetOptionalInputShape(11);
  auto project_shape = context->GetInputShape(5);
  if (project_shape != nullptr) {
    if (project_shape->GetDimNum() < 2) {
      return ge::GRAPH_FAILED;
    }
    stateSize = project_shape->GetDim(1);
  }
  if (x_shape->GetDimNum() != 3) {
    GELOGE(ge::GRAPH_FAILED, "The input shape of X not equal 3, please check!");
    return ge::GRAPH_FAILED;
  }

  auto hidden_size = w_shape->GetDim(1) / 4;
  auto num_step = x_shape->GetDim(0);
  auto batch_size = x_shape->GetDim(1);

  auto outputY_shape = context->GetOutputShape(0);
  auto outputH_shape = context->GetOutputShape(1);
  auto outputC_shape = context->GetOutputShape(2);
  auto outputI_shape = context->GetOutputShape(3);
  auto outputJ_shape = context->GetOutputShape(4);
  auto outputF_shape = context->GetOutputShape(5);
  auto outputO_shape = context->GetOutputShape(6);
  auto outputTanhc_shape = context->GetOutputShape(7);

  if (outputY_shape == nullptr || outputH_shape == nullptr || outputC_shape == nullptr || outputI_shape == nullptr ||
      outputJ_shape == nullptr || outputF_shape == nullptr || outputO_shape == nullptr ||
      outputTanhc_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  *outputY_shape = *outputH_shape = {num_step, batch_size, stateSize};

  *outputC_shape = *outputI_shape = *outputJ_shape = *outputF_shape = *outputO_shape =
      *outputTanhc_shape = {num_step, batch_size, hidden_size};

  return ge::GRAPH_SUCCESS;
}

// todo，其实这个节点tiling所需要的信息非常少，把所有的输入、输出shape等信息构造好较为浪费
ge::graphStatus TilingForDynamicRNNV3(TilingContext *context) {
  if (context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE) {
    return ge::GRAPH_FAILED;
  }
  auto x_shape = context->GetInputShape(0);
  if (x_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &x_storage_shape = x_shape->GetStorageShape();
  if (x_storage_shape.GetDimNum() < DEFAULT_XSHAPE_SIZE) {
    return ge::GRAPH_FAILED;
  }

  auto compile_info = reinterpret_cast<const DynamicRNNV3CompileInfo *>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto runParams = context->GetTilingData<DynamicRnnV3Param>();
  if (runParams == nullptr) {
    return ge::GRAPH_FAILED;
  }

  runParams->sequenceLength = static_cast<int32_t>(x_storage_shape.GetDim(0));
  runParams->dynamicRnnBatch = static_cast<int32_t>(x_storage_shape.GetDim(DEFAULT_INDEX_TWO));
  runParams->chequeIndex = GetRnnV3LibItem(compile_info, x_storage_shape);
  if (runParams->chequeIndex == DEFAULT_RETURN) {
    return false;
  }
  if (context->SetTilingKey(runParams->chequeIndex) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  // todo set blockdim and workspace when loading
  context->SetBlockDim(32);
  auto workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = 4096;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynamicRNNV3(KernelContext *context) {
  auto compile_info = context->GetOutputPointer<DynamicRNNV3CompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  if (compile_info == nullptr || json_str == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  if (parsed_object_cinfo == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const nlohmann::json &allVars = (*parsed_object_cinfo)["vars"];
  if (allVars.empty()) {
    GELOGE(ge::GRAPH_FAILED, "DynamicRnnV3TilingParse: get vars failed.");
    return ge::GRAPH_FAILED;
  }
  compile_info->tune_shape_list = allVars.at("tune_shape_list").get<std::vector<std::vector<int64_t>>>();
  if (compile_info->tune_shape_list.empty()) {
    GELOGE(ge::GRAPH_FAILED, "DynamicRnnV3TilingParse: get tune_shape_list failed.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynamicRNNV3)
    .InferShape(InferShapeForDynamicRNNV3)
    .Tiling(TilingForDynamicRNNV3)
    .TilingParse<DynamicRNNV3CompileInfo>(TilingPrepareForDynamicRNNV3);
}  // namespace gert
