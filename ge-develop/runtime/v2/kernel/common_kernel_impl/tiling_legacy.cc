/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling.h"
#include "ge/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/math_util.h"
#include "graph/args_format_desc.h"
#include "common/checker.h"
#include "engine/node_converter_utils.h"
#include "exe_graph/lowering/shape_utils.h"
#include "common/op_tiling/tiling_dfx.h"
#include "adump_pub.h"
#include "mmpa/mmpa_api.h"
#include "compatible_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling/op_compile_info_manager.h"
#include "register/op_tiling_registry.h"
#include "register/op_tiling/op_tiling_utils.h"

// tiling kernels for FFTS, which will be deprecated in the future
namespace gert {
namespace kernel {
namespace {
using TilingFuncV4 = bool (*)(const ge::Operator &, const optiling::CompileInfoPtr, optiling::OpRunInfoV2 &);
using TilingFuncV3 = bool (*)(const ge::Operator &, const void *, optiling::OpRunInfoV2 &);

ge::graphStatus TilingV4Legacy(const KernelContext *const context, const ge::Operator &op,
                               optiling::OpRunInfoV2 &op_run_info) {
  auto tiling_func_index = static_cast<size_t>(CompatibleTilingInputIndex::kTilingFwkData);
  auto tiling_func = context->GetInputValue<TilingFuncV4>(tiling_func_index);
  GE_ASSERT_NOTNULL(tiling_func);
  auto compile_info_index = static_cast<size_t>(CompatibleTilingInputIndex::kCompileInfo);
  auto compile_info = context->GetInputPointer<optiling::CompileInfoBase *>(compile_info_index);
  GE_ASSERT_NOTNULL(compile_info);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  const bool ret = (tiling_func)(op, (const std::shared_ptr<optiling::CompileInfoBase> &)(*compile_info), op_run_info);
  GE_ASSERT_TRUE(ret, "Fail to call op tiling function v4 of op[%s, %s].", op_desc->GetName().c_str(),
                 op_desc->GetType().c_str());
  GELOGI("Do optiling v4 succeed. op_name:%s, op_type:%s.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingV3Legacy(const KernelContext *const context, const ge::Operator &op,
                               optiling::OpRunInfoV2 &op_run_info) {
  auto tiling_func_index = static_cast<size_t>(CompatibleTilingInputIndex::kTilingFwkData);
  auto tiling_func = context->GetInputValue<TilingFuncV3>(tiling_func_index);
  GE_ASSERT_NOTNULL(tiling_func);
  auto compile_info_index = static_cast<size_t>(CompatibleTilingInputIndex::kCompileInfo);
  auto compile_info = context->GetInputValue<void *>(compile_info_index);
  GE_ASSERT_NOTNULL(compile_info);
  const bool ret = (tiling_func)(op, compile_info, op_run_info);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(ret, "Fail to call op tiling function v3 of op[%s, %s].", op_desc->GetName().c_str(),
                 op_desc->GetType().c_str());
  GELOGI("Do optiling v3 succeed. op_name:%s, op_type:%s.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus TilingProcLegacy(KernelContext *context, ge::graphStatus &tiling_func_result) {
  auto input_num = context->GetInputNum();
  GE_ASSERT_TRUE(input_num >= 1, "[Tiling] para check failed input_num %" PRId64 "", input_num);
  auto tiling_data = reinterpret_cast<TilingContext *>(context)->GetRawTilingData();
  if (tiling_data != nullptr) {
    tiling_data->SetDataSize(0);
  }
  auto tiling_func = context->GetInputValue<KernelRegistry::KernelFunc>(input_num - 1);
  GE_ASSERT_NOTNULL(tiling_func);
  tiling_func_result = tiling_func(context);
  if (tiling_func_result == ge::GRAPH_SUCCESS) {
    return AlignWorkspaceSizes(context);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingLegacy(KernelContext *context) {
  ge::graphStatus tiling_func_result;
  GE_ASSERT_SUCCESS(TilingProcLegacy(context, tiling_func_result));
  return tiling_func_result;
}
REGISTER_KERNEL(TilingLegacy)
    .RunFunc(TilingLegacy)
    .OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo)
    .TracePrinter(PrintTilingData);

ge::graphStatus InnerCompatibleTilingLegacy(KernelContext *context, ge::graphStatus &tiling_func_result) {
  auto input_num = context->GetInputNum();
  if (input_num < static_cast<size_t>(CompatibleTilingInputIndex::kTilingFuncInputNum)) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "[Tiling] para check failed input_num %ld", input_num);
    return ge::GRAPH_FAILED;
  }
  auto tiling_data = reinterpret_cast<TilingContext *>(context)->GetRawTilingData();
  if (tiling_data != nullptr) {
    tiling_data->SetDataSize(0);
  }
  // unserialize operator
  auto op = context->GetInputValue<ge::Operator *>(static_cast<size_t>(CompatibleTilingInputIndex::kOp));
  GE_ASSERT_NOTNULL(op);
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  GE_ASSERT_NOTNULL(op_desc);

  std::vector<ge::GeTensorPtr> tensor_holder;
  if (!op_desc->GetOpInferDepends().empty()) {
    auto callback = [&context, &tensor_holder](const ge::ConstNodePtr &node, const size_t index,
                                               ge::GeTensorPtr &tensor) {
      (void)node;
      auto infer_shape_context = reinterpret_cast<TilingContext *>(context);
      auto input_start_pos = static_cast<size_t>(CompatibleTilingInputIndex::kTilingFuncInputNum);
      auto shape_tensor = infer_shape_context->GetInputTensor(index + input_start_pos);
      return KernelCompatibleUtils::ConvertRTTensorToGeTensor(shape_tensor, tensor, tensor_holder);
    };
    ge::OpDescUtils::SetCallbackGetConstInputFuncToOperator(*op, callback);
  }
  // update input/output shape on operator
  GE_CHK_STATUS_RET_NOLOG(UpdateIOShapeToOp(context, *op));
  std::vector<int32_t> indexes;
  optiling::ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);
  // get tiling func version
  auto tiling_version_index = static_cast<size_t>(CompatibleTilingInputIndex::kTilingVersion);
  auto tiling_version = context->GetInputValue<size_t>(tiling_version_index);
  optiling::OpRunInfoV2 op_run_info;
  if (tiling_version == static_cast<uint64_t>(TilingVersion::kV4)) {
    tiling_func_result = TilingV4Legacy(context, *op, op_run_info);
  } else if (tiling_version == static_cast<uint64_t>(TilingVersion::kV3)) {
    tiling_func_result = TilingV3Legacy(context, *op, op_run_info);
  } else {
    GELOGE(ge::GRAPH_FAILED, "Failed to tiling parse, not support v3 or v4.");
    return ge::GRAPH_FAILED;
  }
  optiling::RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);

  if (tiling_func_result == ge::GRAPH_SUCCESS) {
    UpdateTilingOutputsToContext(op_run_info, context);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompatibleTilingLegacy(KernelContext *context) {
  ge::graphStatus tiling_func_result;
  GE_ASSERT_SUCCESS(InnerCompatibleTilingLegacy(context, tiling_func_result));
  return tiling_func_result;
}
REGISTER_KERNEL(CompatibleTilingLegacy)
    .RunFunc(CompatibleTilingLegacy)
    .OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo);
}  // namespace kernel
}  // namespace gert
