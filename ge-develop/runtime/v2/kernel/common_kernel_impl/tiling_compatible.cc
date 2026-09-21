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
#include "graph/ge_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "register/kernel_registry.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "register/op_tiling/op_compile_info_manager.h"
#include "register/op_tiling_registry.h"
#include "register/op_tiling/op_tiling_utils.h"
#include "common/checker.h"
#include "compatible_utils.h"
#include "graph/utils/math_util.h"
#include "ge/ge_error_codes.h"

namespace gert {
namespace kernel {
namespace {
using TilingParseFuncV4 = optiling::CompileInfoPtr (*)(const ge::Operator &, const ge::AscendString &);
using TilingFuncV4 = bool (*)(const ge::Operator &, const optiling::CompileInfoPtr, optiling::OpRunInfoV2 &);
using TilingParseFuncV3 = void *(*)(const ge::Operator &, const ge::AscendString &);
using TilingFuncV3 = bool (*)(const ge::Operator &, const void *, optiling::OpRunInfoV2 &);

ge::graphStatus GetIONumOfOp(const TilingContext *const context, ge::Operator &op, size_t &input_num,
                             size_t &output_num) {
  input_num = context->GetComputeNodeInputNum();
  auto input_num_on_op = op.GetInputsSize();
  if (input_num_on_op != input_num) {
    ge::AscendString op_name;
    (void)op.GetName(op_name);
    GELOGE(ge::PARAM_INVALID, "Input num on op %s is %zu, input num on context is %zu, not match.", op_name.GetString(),
           input_num_on_op, input_num);
    return ge::PARAM_INVALID;
  }
  auto compute_node_info = context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  output_num = compute_node_info->GetOutputsNum();
  auto output_num_on_op = op.GetOutputsSize();
  if (output_num_on_op != output_num) {
    ge::AscendString op_name;
    (void)op.GetName(op_name);
    GELOGE(ge::PARAM_INVALID, "Output num on op %s is %zu, output num on context is %zu, not match.",
           op_name.GetString(), output_num_on_op, output_num);
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}
ge::Shape ConvertRtShapeToGeShape(const gert::Shape &src_shape) {
  std::vector<int64_t> shape_dims;
  for (size_t j = 0U; j < src_shape.GetDimNum(); ++j) {
    shape_dims.emplace_back(src_shape.GetDim(j));
  }
  return ge::Shape(shape_dims);
}
}  // namespace

ge::graphStatus UpdateIOShapeToOp(KernelContext *context, ge::Operator &op) {
  auto tiling_context = reinterpret_cast<TilingContext *>(context);
  size_t input_num = 0U;
  size_t output_num = 0U;
  GE_CHK_STATUS_RET_NOLOG(GetIONumOfOp(tiling_context, op, input_num, output_num));
  const auto &op_dsec = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_dsec);

  auto input_start_pos = static_cast<size_t>(CompatibleTilingInputIndex::kTilingFuncInputNum);
  for (size_t i = 0U; i < input_num; ++i) {
    auto input_shape = context->GetInputPointer<StorageShape>(input_start_pos + i);
    GE_ASSERT_NOTNULL(input_shape);
    ge::Shape storage_shape = ConvertRtShapeToGeShape(input_shape->GetStorageShape());
    ge::Shape origin_shape = ConvertRtShapeToGeShape(input_shape->GetOriginShape());
    auto input_desc = op.GetInputDesc(i);
    input_desc.SetShape(storage_shape);
    input_desc.SetOriginShape(origin_shape);
    const auto &input_name = op_dsec->GetInputNameByIndex(i);
    op.UpdateInputDesc(input_name.c_str(), input_desc);
  }

  auto output_start_pos = input_start_pos + input_num;
  for (size_t i = 0U; i < output_num; ++i) {
    auto output_shape = context->GetInputPointer<StorageShape>(output_start_pos + i);
    GE_ASSERT_NOTNULL(output_shape);
    ge::Shape storage_shape = ConvertRtShapeToGeShape(output_shape->GetStorageShape());
    ge::Shape origin_shape = ConvertRtShapeToGeShape(output_shape->GetOriginShape());
    const auto &output_name = op_dsec->GetOutputNameByIndex(i);
    auto output_desc = op.GetOutputDesc(i);
    output_desc.SetShape(storage_shape);
    output_desc.SetOriginShape(origin_shape);
    op.UpdateOutputDesc(output_name.c_str(), output_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FindCompatibleTilingFunc(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(0);
  if (node_type == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto &op_func_map = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(node_type);
  if (iter == op_func_map.end()) {
    GELOGI("Do optiling function is not found by op type[%s].", node_type);
    iter = op_func_map.find(optiling::OP_TYPE_AUTO_TILING);
    if (iter == op_func_map.end()) {
      GELOGW("Optiling function old version of op type[%s] is not found by Autotiling.", node_type);
      REPORT_INNER_ERR_MSG("EZ9999", "Optiling function old version is not found. op_type[%s].", node_type);
      return ge::GRAPH_FAILED;
    }
  }
  // build outputs
  size_t out_idx_tiling_version = static_cast<size_t>(FindCompatibleTilingFuncOutputIndex::kTilingVersion);
  size_t out_idx_tiling_parse_func = static_cast<size_t>(FindCompatibleTilingFuncOutputIndex::kTilingParseFunc);
  size_t out_idx_tiling_func = static_cast<size_t>(FindCompatibleTilingFuncOutputIndex::kTilingFunc);

  auto &tiling_func_info = iter->second;
  auto tiling_version = context->GetOutputPointer<uint64_t>(out_idx_tiling_version);
  GE_ASSERT_NOTNULL(tiling_version);
  if (tiling_func_info.IsFunctionV4()) {
    *tiling_version = static_cast<uint64_t>(TilingVersion::kV4);
    auto tiling_parse_fun_ptr = context->GetOutputPointer<TilingParseFuncV4>(out_idx_tiling_parse_func);
    GE_ASSERT_NOTNULL(tiling_parse_fun_ptr);
    *tiling_parse_fun_ptr = *(tiling_func_info.GetOpParseFuncV4().target<TilingParseFuncV4>());
    auto tiling_fun_ptr = context->GetOutputPointer<TilingFuncV4>(out_idx_tiling_func);
    GE_ASSERT_NOTNULL(tiling_fun_ptr);
    *tiling_fun_ptr = *(tiling_func_info.GetOpTilingFuncV4().target<TilingFuncV4>());
  } else if (tiling_func_info.IsFunctionV3()) {
    *tiling_version = static_cast<uint64_t>(TilingVersion::kV3);
    auto tiling_parse_fun_ptr = context->GetOutputPointer<TilingParseFuncV3>(out_idx_tiling_parse_func);
    GE_ASSERT_NOTNULL(tiling_parse_fun_ptr);
    *tiling_parse_fun_ptr = *(tiling_func_info.GetOpParseFuncV3().target<TilingParseFuncV3>());
    auto tiling_fun_ptr = context->GetOutputPointer<TilingFuncV3>(out_idx_tiling_func);
    GE_ASSERT_NOTNULL(tiling_fun_ptr);
    *tiling_fun_ptr = *(tiling_func_info.GetOpTilingFuncV3().target<TilingFuncV3>());
  } else {
    GELOGE(ge::GRAPH_FAILED, "Node %s not support v3 or v4 tiling.", node_type);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FindCompatibleTilingFunc).RunFunc(FindCompatibleTilingFunc);

ge::graphStatus TilingParseV4(KernelContext *context, const ge::Operator &op, const std::string &compile_info_key,
                              const std::string &compile_info_json) {
  auto update_compile_info_to_context = [](KernelContext *ctx, optiling::CompileInfoPtr &op_compile_info_ptr) {
    // update compile info to output
    auto compile_info = ctx->GetOutputPointer<optiling::CompileInfoBase *>(0);
    if (compile_info != nullptr) {
      *compile_info = op_compile_info_ptr.get();
    }
  };

  // try get tiling_parse_func from cache
  optiling::CompileInfoPtr op_compile_info_ptr =
      optiling::CompileInfoManager::Instance().GetCompileInfo(compile_info_key);
  if (op_compile_info_ptr != nullptr) {
    update_compile_info_to_context(context, op_compile_info_ptr);
    return ge::GRAPH_SUCCESS;
  }

  auto tiling_parse_func_index = static_cast<uint64_t>(CompatibleTilingParseInputIndex::kTilingParseFun);
  auto op_parse_fun = context->GetInputValue<TilingParseFuncV4>(tiling_parse_func_index);
  GE_ASSERT_NOTNULL(op_parse_fun);
  op_compile_info_ptr = op_parse_fun(op, ge::AscendString(compile_info_json.c_str()));
  GE_ASSERT_NOTNULL(op_compile_info_ptr);
  optiling::CompileInfoManager::Instance().SetCompileInfo(compile_info_key, op_compile_info_ptr);
  update_compile_info_to_context(context, op_compile_info_ptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseV3(KernelContext *context, const ge::Operator &op, const std::string &compile_info_key,
                              const std::string &compile_info_json) {
  auto update_compile_info_to_context = [](KernelContext *ctx, void *op_compile_info_ptr) {
    // update compile info to output
    auto compile_info = ctx->GetOutputPointer<void *>(0);
    if (compile_info != nullptr) {
      *compile_info = op_compile_info_ptr;
    }
  };
  // try get tiling_parse_func from cache
  void *op_compile_info_ptr = optiling::CompileInfoCache::Instance().GetCompileInfo(compile_info_key);
  if (op_compile_info_ptr != nullptr) {
    update_compile_info_to_context(context, op_compile_info_ptr);
    return ge::GRAPH_SUCCESS;
  }

  auto tiling_parse_func_index = static_cast<uint64_t>(CompatibleTilingParseInputIndex::kTilingParseFun);
  auto op_parse_fun = context->GetInputValue<TilingParseFuncV3>(tiling_parse_func_index);
  GE_ASSERT_NOTNULL(op_parse_fun);
  op_compile_info_ptr = op_parse_fun(op, ge::AscendString(compile_info_json.c_str()));
  GE_ASSERT_NOTNULL(op_compile_info_ptr);
  optiling::CompileInfoCache::Instance().SetCompileInfo(compile_info_key, op_compile_info_ptr);

  update_compile_info_to_context(context, op_compile_info_ptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompatibleTilingParse(KernelContext *context) {
  // prepare input_param operator, update input shape from input to operator
  auto op = context->GetInputValue<ge::Operator *>(static_cast<size_t>(CompatibleTilingParseInputIndex::kOp));
  GE_ASSERT_NOTNULL(op);

  // get compile_info_key
  auto compile_info_key_index = static_cast<uint64_t>(CompatibleTilingParseInputIndex::kCompileInfoKey);
  auto compile_info_key = context->GetInputValue<char *>(compile_info_key_index);
  GE_ASSERT_NOTNULL(compile_info_key);
  // get compile_info_json
  auto compile_info_json_index = static_cast<uint64_t>(CompatibleTilingParseInputIndex::kCompileInfoJson);
  auto compile_info_json = context->GetInputValue<char *>(compile_info_json_index);
  GE_ASSERT_NOTNULL(compile_info_json);

  auto tiling_version_index = static_cast<uint64_t>(CompatibleTilingParseInputIndex::kTilingVersion);
  auto tiling_version = context->GetInputValue<size_t>(static_cast<size_t>(tiling_version_index));
  if (tiling_version == static_cast<uint64_t>(TilingVersion::kV4)) {
    return TilingParseV4(context, *op, compile_info_key, compile_info_json);
  } else if (tiling_version == static_cast<uint64_t>(TilingVersion::kV3)) {
    return TilingParseV3(context, *op, compile_info_key, compile_info_json);
  } else {
    GELOGE(ge::GRAPH_FAILED, "Failed to tiling parse, not support v3 or v4.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CompatibleTilingParse).RunFunc(CompatibleTilingParse);

ge::graphStatus TilingV4(const KernelContext *const context, const ge::Operator &op,
                         const TilingFwkData &tiling_fwk_data, optiling::OpRunInfoV2 &op_run_info) {
  auto tiling_func = reinterpret_cast<TilingFuncV4>(tiling_fwk_data.tiling_func);
  GE_ASSERT_NOTNULL(tiling_func);
  auto compile_info_index = static_cast<size_t>(CompatibleTilingInputIndex::kCompileInfo);
  auto compile_info = context->GetInputPointer<optiling::CompileInfoBase *>(compile_info_index);
  GE_ASSERT_NOTNULL(compile_info);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  const bool ret = (tiling_func)(op, (const std::shared_ptr<optiling::CompileInfoBase> &)(*compile_info), op_run_info);
  if (!ret) {
    GELOGW("Fail to call op tiling function v4 of op[%s, %s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  GELOGI("Do optiling v4 succeed. op_name:%s, op_type:%s.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingV3(const KernelContext *const context, const ge::Operator &op,
                         const TilingFwkData &tiling_fwk_data, optiling::OpRunInfoV2 &op_run_info) {
  auto tiling_func = reinterpret_cast<TilingFuncV3>(tiling_fwk_data.tiling_func);
  GE_ASSERT_NOTNULL(tiling_func);
  auto compile_info_index = static_cast<size_t>(CompatibleTilingInputIndex::kCompileInfo);
  auto compile_info = context->GetInputValue<void *>(compile_info_index);
  GE_ASSERT_NOTNULL(compile_info);
  const bool ret = (tiling_func)(op, compile_info, op_run_info);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  if (!ret) {
    GELOGW("Fail to call op tiling function v3 of op[%s, %s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  GELOGI("Do optiling v3 succeed. op_name:%s, op_type:%s.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateTilingOutputsToContext(const optiling::OpRunInfoV2 &op_run_info, KernelContext *context) {
  auto tiling_context = reinterpret_cast<TilingContext *>(context);
  GE_CHK_STATUS_RET(tiling_context->SetTilingKey(op_run_info.GetTilingKey()), "Para check failed, tiling key null.");
  GE_CHK_STATUS_RET(tiling_context->SetBlockDim(op_run_info.GetBlockDim()), "Para check failed, block_dim null.");
  GE_CHK_STATUS_RET(tiling_context->SetNeedAtomic(op_run_info.GetClearAtomic()),
                    "Para check failed, atomic flag null.");
  GE_CHK_STATUS_RET(tiling_context->SetTilingCond(op_run_info.GetTilingCond()),
                    "Para check failed, tiling condition null.");
  auto node_type = tiling_context->GetNodeType();
  auto tiling_data_av = context->GetOutput(TilingContext::kOutputTilingData);
  auto workspace_av = context->GetOutput(TilingContext::kOutputWorkspace);
  if (node_type == nullptr || tiling_data_av == nullptr || workspace_av == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "[BuildTilingOutputs] para check failed, node_type %p, tiling_data_av %p, workspace_av %p", node_type,
           tiling_data_av, workspace_av);
    return ge::GRAPH_FAILED;
  }
  std::string tiling_data = op_run_info.GetAllTilingData().str();
  auto tiling_data_in_context = tiling_context->GetRawTilingData();
  GE_ASSERT_NOTNULL(tiling_data_in_context);
  tiling_data_in_context->SetDataSize(0);
  auto tiling_data_base = tiling_data_in_context->GetData();
  GE_ASSERT_TRUE(tiling_data_in_context->GetCapacity() > tiling_data_in_context->GetDataSize());
  auto left = tiling_data_in_context->GetCapacity() - tiling_data_in_context->GetDataSize();
  if (memcpy_s(tiling_data_base, static_cast<size_t>(left), tiling_data.data(), tiling_data.size()) != EOK) {
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  tiling_data_in_context->SetDataSize(tiling_data.size());

  auto workspace_data = tiling_context->GetWorkspaceSizes(op_run_info.GetWorkspaceNum());
  auto workspace = op_run_info.GetAllWorkspaces();
  for (size_t i = 0U; i < op_run_info.GetWorkspaceNum(); i++) {
    GE_ASSERT_TRUE(
        !ge::RoundUpOverflow(static_cast<size_t>(workspace[i]), kAiCoreWorkspaceAlignment, workspace_data[i]));
  }
  return ge::GRAPH_SUCCESS;
}

// inputs
// inputs[0] op buffer head
// inputs[1] op buffer size
// inputs[2] compile info
// inputs[3] tiling version
// inputs[4] TilingFwkData
// inputs[5, 5+n) input shapes
// inputs[5+n,m) output shapes

ge::graphStatus InnerCompatibleTiling(KernelContext *context, ge::graphStatus &tiling_func_result) {
  auto input_num = context->GetInputNum();
  if (input_num < static_cast<size_t>(CompatibleTilingInputIndex::kTilingFuncInputNum)) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "[Tiling] para check failed input_num %ld", input_num);
    return ge::GRAPH_FAILED;
  }
  const auto tiling_fwk_data =
      context->GetInputPointer<TilingFwkData>(static_cast<size_t>(CompatibleTilingInputIndex::kTilingFwkData));
  GE_ASSERT_NOTNULL(tiling_fwk_data);
  RefreshOutputAddr(context, tiling_fwk_data->launch_arg);
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
    tiling_func_result = TilingV4(context, *op, *tiling_fwk_data, op_run_info);
  } else if (tiling_version == static_cast<uint64_t>(TilingVersion::kV3)) {
    tiling_func_result = TilingV3(context, *op, *tiling_fwk_data, op_run_info);
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

ge::graphStatus CompatibleTiling(KernelContext *context) {
  ge::graphStatus tiling_func_result;
  GE_ASSERT_SUCCESS(InnerCompatibleTiling(context, tiling_func_result));
  return tiling_func_result;
}
REGISTER_KERNEL(CompatibleTiling).RunFunc(CompatibleTiling).OutputsCreator(BuildTilingOutputs)
    .ExceptionDumpInfoFiller(FillTilingInfo);

ge::graphStatus FallibleCompatibleTiling(KernelContext *context) {
  auto result = context->GetOutputPointer<uint32_t>(static_cast<size_t>(FallibleTilingExOutputIndex::kTilingStatus));
  if (result == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::graphStatus op_tiling_result;
  GE_ASSERT_SUCCESS(InnerCompatibleTiling(context, op_tiling_result));
  *result = op_tiling_result == ge::GRAPH_SUCCESS ? 0U : 1U;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FallibleCompatibleTiling).RunFunc(FallibleCompatibleTiling).OutputsCreator(BuildTilingOutputs);
}  // namespace kernel
}  // namespace gert
