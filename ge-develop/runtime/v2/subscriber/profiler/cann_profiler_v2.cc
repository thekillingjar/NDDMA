/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_profiler_v2.h"

#include <nlohmann/json.hpp>
#include "runtime/model_v2_executor.h"
#include "core/builder/node_types.h"
#include "graph/def_types.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "lowering/placement/placed_lowering_result.h"
#include "lowering/pass_changed_kernels_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "subscriber/subscriber_utils.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "aprof_pub.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "common/sgt_slice_type.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "kernel/known_subgraph/davinci_model_kernel.h"
#include "exe_graph/lowering/value_holder_utils.h"

namespace gert {
namespace {
constexpr const char *kInvalidGraphName = "Invalid";
constexpr uint32_t kEnableHf32 = 0x40U;
constexpr uint32_t kOpImplHf32ModeEnable = 1U;
constexpr uint32_t kOpImplHf32ModeDisable = 0U;

const std::map<std::string, MsprofGeTaskType> core_type_to_task_types {
    {"AI_CORE", MSPROF_GE_TASK_TYPE_AI_CORE},
    {"AI_CPU", MSPROF_GE_TASK_TYPE_AI_CPU},
    {"MIX_AIC", MSPROF_GE_TASK_TYPE_MIX_AIC},
    {"MIX_AIV", MSPROF_GE_TASK_TYPE_MIX_AIV},
    {"MIX_AICORE", MSPROF_GE_TASK_TYPE_AI_CORE},
    {"MIX_VECTOR_CORE", MSPROF_GE_TASK_TYPE_AI_CORE},
    {"AIV", MSPROF_GE_TASK_TYPE_AIV},
    {"DSA", MSPROF_GE_TASK_TYPE_DSA},
    {"WRITE_BACK", MSPROF_GE_TASK_TYPE_WRITE_BACK},
    {"INVALID", MSPROF_GE_TASK_TYPE_INVALID}
};

const std::map<rtFftsPlusContextType_t, MsprofGeTaskType> ctx_type_to_task_types{
    {RT_CTX_TYPE_AICORE, MSPROF_GE_TASK_TYPE_AI_CORE},
    {RT_CTX_TYPE_AIV, MSPROF_GE_TASK_TYPE_AIV},
    {RT_CTX_TYPE_MIX_AIC, MSPROF_GE_TASK_TYPE_MIX_AIC},
    {RT_CTX_TYPE_MIX_AIV, MSPROF_GE_TASK_TYPE_MIX_AIV},
    {RT_CTX_TYPE_AICPU, MSPROF_GE_TASK_TYPE_AI_CPU},
    {RT_CTX_TYPE_WRITEBACK_DATA, MSPROF_GE_TASK_TYPE_WRITE_BACK},
    {RT_CTX_TYPE_INVALIDATE_DATA, MSPROF_GE_TASK_TYPE_INVALID},
    {RT_CTX_TYPE_DSA, MSPROF_GE_TASK_TYPE_DSA}
};

bool IsNoNeedProfNodes(ge::NodePtr &node) {
  if (IsTypeNetOutput(node->GetTypePtr())) {
    return true;
  }

  // 静态子图里的node不在动态流程里处理
  const auto &own_graph = node->GetOwnerComputeGraphBarePtr();
  if ((own_graph != nullptr) && (!own_graph->GetGraphUnknownFlag())) {
    GELOGD("node name:%s, graph name:%s", node->GetNamePtr(), own_graph->GetName().c_str());
    return true;
  }

  return false;
}

KernelNameAndIdx GetKernelNameAndIdxAfterPass(const ge::OpDesc *op_desc, const KernelNameAndIdx &kernel_name_and_idx) {
  const auto pass_changed_info = op_desc->TryGetExtAttr(kPassChangedInfo, PassChangedKernels{});
  for (const auto &pass_changed_kernel : pass_changed_info.pass_changed_kernels) {
    if (kernel_name_and_idx == pass_changed_kernel.first) {
      return pass_changed_kernel.second;
    }
  }
  return kernel_name_and_idx;
}

void InitProfTensorData(const ComputeNodeInfo *compute_node_info, const size_t index, const uint64_t offset_idx,
                        MsprofTensorInfo *tensor_info) {
  const auto initTensorDesc = [&tensor_info](const MsprofGeTensorType tensor_type, const CompileTimeTensorDesc *td,
                                             const uint64_t offset_index) {
    tensor_info->tensorData[offset_index].tensorType = static_cast<uint32_t>(tensor_type);
    // when enum Format is changed, profiling analyze needs to be synchronized
    tensor_info->tensorData[offset_index].format = static_cast<uint32_t>(td->GetStorageFormat());
    // when enum DataType is changed, profiling analyze needs to be synchronized
    const ge::DataType data_type = td->GetDataType();
    const auto prof_dtype = (static_cast<uint32_t>(data_type) < static_cast<uint32_t>(ge::DT_MAX))
                                ? static_cast<uint32_t>(data_type)
                                : static_cast<uint32_t>(ge::DT_UNDEFINED);
    tensor_info->tensorData[offset_index].dataType = prof_dtype;
  };

  const size_t input_size = compute_node_info->GetInputsNum();
  if (index < input_size) {
    // when current index is smaller than input_size, build tensor by input tensor
    initTensorDesc(MSPROF_GE_TENSOR_TYPE_INPUT, compute_node_info->GetInputTdInfo(index), offset_idx);
  } else {
    // when current index is bigger than input_size, build tensor by output tensor, use index - input_size as
    // index of output tensor
    initTensorDesc(MSPROF_GE_TENSOR_TYPE_OUTPUT, compute_node_info->GetOutputTdInfo(index - input_size), offset_idx);
  }
}

void BuildSingleTensorInfo(const ComputeNodeInfo *compute_node_info, const uint64_t node_name, const size_t index,
                           const uint32_t tensor_num, TensorInfoWrapper &tensor_info_wrapper) {
  auto &tensor_info = tensor_info_wrapper.tensor_info;
  tensor_info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensor_info.level = MSPROF_REPORT_NODE_LEVEL;
  tensor_info_wrapper.tensor_num = tensor_num;
  tensor_info.dataLen =
      kTensorInfoBytesWithCap + kTensorInfoBytes * (static_cast<uint32_t>(tensor_num) - 1U);
  auto prof_tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
  prof_tensor_data->opName = node_name;
  prof_tensor_data->tensorNum = tensor_num;
  for (size_t k = 0UL; k < static_cast<size_t>(tensor_num); ++k) {
    const size_t tensor_index = (index * static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM)) + k;
    InitProfTensorData(compute_node_info, tensor_index, k, prof_tensor_data);
  }
}

void InitTensorInfo(const ComputeNodeInfo *compute_node_info, const uint64_t node_name,
                    std::vector<TensorInfoWrapper> &tensor_infos) {
  auto total_num = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
  GELOGD("[Cann Profiling] node name is %s, node type is %s, tensor size is %zu", compute_node_info->GetNodeName(),
         compute_node_info->GetNodeType(), total_num);
  const size_t index = total_num / static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  for (size_t j = 0UL; j < index; ++j) {
    TensorInfoWrapper tensor_info_wrapper{};
    BuildSingleTensorInfo(compute_node_info, node_name, j, MSPROF_GE_TENSOR_DATA_NUM, tensor_info_wrapper);
    tensor_infos.emplace_back(tensor_info_wrapper);
  }

  const size_t remain_index = total_num % static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  if (remain_index == 0UL) {
    return;
  }
  TensorInfoWrapper tensor_info_wrapper{};
  BuildSingleTensorInfo(compute_node_info, node_name, index, remain_index, tensor_info_wrapper);
  tensor_infos.emplace_back(tensor_info_wrapper);
}

ge::Status ParseContextByNode(Node *node, const char **kernel_type, const ComputeNodeInfo **compute_node_info) {
  const auto kernel_context = reinterpret_cast<KernelContext *>(&node->context);
  const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
  GE_ASSERT_NOTNULL(kernel_extend_info);
  *kernel_type = kernel_extend_info->GetKernelType();
  *compute_node_info = static_cast<const ComputeNodeInfo *>(kernel_context->GetComputeNodeExtend());
  return ge::SUCCESS;
}

ge::Status GetShapeInfoFromJson(const std::string &json_str, std::vector<std::vector<int64_t>> &input_shapes,
                                std::vector<std::vector<int64_t>> &output_shapes) {
  GE_ASSERT_TRUE(nlohmann::json::accept(json_str), "[Cann Profiling] Json string %s is invalid.", json_str.c_str());
  nlohmann::json slice_info_json = nlohmann::json::parse(json_str);
  GE_ASSERT_TRUE(!slice_info_json.is_null(), "[Cann Profiling] Failed to parse json string %s.", json_str.c_str());
  const auto input_tensor_slice = slice_info_json.find("input_tensor_slice");
  const auto output_tensor_slice = slice_info_json.find("output_tensor_slice");
  GE_ASSERT_TRUE(input_tensor_slice != slice_info_json.end());
  GE_ASSERT_TRUE(output_tensor_slice != slice_info_json.end());
  if (input_tensor_slice->size() == 0U && output_tensor_slice->size() == 0U) {
    GELOGI("[Cann Profiling] No shape info in sgt json.");
    return ge::FAILED;
  }
  const auto convert_shape = [&json_str](const nlohmann::json &json,
                                         std::vector<std::vector<int64_t>> &shapes) -> ge::Status {
    if ((!json.is_array()) || (json.size() <= 0U)) {
      return ge::SUCCESS;
    }
    const auto non_tail = json[0U];
    if ((!non_tail.is_array()) || (non_tail.empty())) {
      return ge::SUCCESS;
    }
    for (size_t i = 0U; i < non_tail.size(); ++i) {
      shapes.emplace_back();
      if (!non_tail[i].is_array() || non_tail[i].empty()) {
        continue;
      }
      for (size_t j = 0U; j < non_tail[i].size(); ++j) {
        const auto lower_json = non_tail[i][j].find("lower");
        const auto higher_json = non_tail[i][j].find("higher");
        GE_ASSERT_TRUE((lower_json != non_tail[i][j].end()) && (higher_json != non_tail[i][j].end()),
                       "Invalid json str %s", json_str.c_str());
        const auto lower_val = lower_json.value().get<int64_t>();
        const auto higher_val = higher_json.value().get<int64_t>();
        GE_ASSERT_TRUE(higher_val >= lower_val, "Invalid shape, higher %lld, lower %lld", higher_val, lower_val);
        shapes[i].emplace_back(higher_val - lower_val);
      }
    }
    return ge::SUCCESS;
  };
  GE_ASSERT_SUCCESS(convert_shape(input_tensor_slice.value(), input_shapes));
  GE_ASSERT_SUCCESS(convert_shape(output_tensor_slice.value(), output_shapes));
  return ge::SUCCESS;
}
}  // namespace

void TensorInfoWrapper::FillShapeInfo() {
  for (size_t i = 0UL; i < tensor_num; ++i) {
    if ((i >= shapes.size()) || (shapes[i] == nullptr)) {
      continue;
    }
    const auto storage_shape = shapes[i]->GetPointer<StorageShape>();
    if (storage_shape == nullptr) {
      continue;
    }
    const auto shape = storage_shape->GetStorageShape();
    const auto dim_num = shape.GetDimNum();
    auto prof_tensor_info = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
    if (dim_num < static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN)) {
      for (size_t j = 0UL; j < dim_num; ++j) {
        prof_tensor_info->tensorData[i].shape[j] = static_cast<uint32_t>(shape.GetDim(j));
      }
      prof_tensor_info->tensorData[i].shape[dim_num] = 0U;
    } else {
      for (size_t j = 0UL; j < static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN); ++j) {
        prof_tensor_info->tensorData[i].shape[j] = static_cast<uint32_t>(shape.GetDim(j));
      }
    }
  }
}

CannProfilerV2::CannProfilerV2(const std::shared_ptr<const SubscriberExtendInfo> &extend_info)
    : extend_info_(extend_info) {
  if ((extend_info_ != nullptr) && (extend_info_->executor != nullptr) && IsEnabled(ProfilingType::kTaskTime)) {
    Init();
  }
}

void CannProfilerV2::Init() {
  if (is_device_prof_inited_ || !IsEnabled(ProfilingType::kTaskTime)) {
    return;
  }
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  if (execution_data == nullptr) {
    GELOGW("[Cann Profiling] Execution data is empty, do not init profiler.");
    return;
  }
  GELOGI("[Cann Profiling] Init for cann device profiling.");
  InitForCannDevice(execution_data);
  is_device_prof_inited_ = true;
}

void CannProfilerV2::InitNodeContextInfo(const ge::OpDescPtr &op_desc) {
  bool have_context_id = false;
  bool is_fftsplus_task = false;
  std::vector<uint32_t> op_ctx_id_vec;
  if ((ge::AttrUtils::GetBool(op_desc, "_is_fftsplus_task", is_fftsplus_task) && is_fftsplus_task)) {
    op_ctx_id_vec.emplace_back(0);  // mix l2 kernel launch by aic launch, need context id 0 to match pmu data
    have_context_id = true;
  } else {
    have_context_id = ge::AttrUtils::GetListInt(op_desc, kContextIdList, op_ctx_id_vec);
  }

  if (!have_context_id) {
    return;
  }

  const auto ctx_num = op_ctx_id_vec.size();
  if (ctx_num == 0 || ctx_num > MSPROF_CTX_ID_MAX_NUM) {
    GELOGW("[Cann Profiling] Op(%s) context size [%zu] is invalid.", op_desc->GetName().c_str(), ctx_num);
    return;
  }

  const uint64_t name_hash = MsprofGetHashId(op_desc->GetNamePtr(), op_desc->GetName().length());
  auto node_iter = name_hash_to_node_id_.find(name_hash);
  if (node_iter == name_hash_to_node_id_.end()) {
    GELOGW("[Cann Profiling] Op(%s) name hash %llu is not found.", op_desc->GetName().c_str(), name_hash);
    return;
  }

  const auto node_id = node_iter->second;
  auto context_info = &(node_addition_infos_[node_id][0U]->context_info);
  context_info->level = MSPROF_REPORT_NODE_LEVEL;
  context_info->type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  context_info->dataLen = sizeof(MsprofContextIdInfo);
  auto context_data = reinterpret_cast<MsprofContextIdInfo *>(context_info->data);
  context_data->opName = name_hash;
  context_data->ctxIdNum = ctx_num;
  for (size_t index = 0U; index < ctx_num; index++) {
    context_data->ctxIds[index] = op_ctx_id_vec[index];
  }

  if (is_fftsplus_task) {
    bool is_mix_aiv = false;
    (void)ge::AttrUtils::GetBool(op_desc, "_mix_is_aiv", is_mix_aiv);
    prof_extend_infos_[node_id].engine_type = (is_mix_aiv ? MSPROF_GE_TASK_TYPE_MIX_AIV : MSPROF_GE_TASK_TYPE_MIX_AIC);
  } else {
    std::string core_type;
    u_int32_t ctx_type;
    (void)ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
    const auto core_type_iter = core_type_to_task_types.find(core_type);
    if (core_type_iter != core_type_to_task_types.cend()) {
      prof_extend_infos_[node_id].engine_type = core_type_iter->second;
    } else if (ge::AttrUtils::GetInt(op_desc, "_ffts_prof_ctx_type", ctx_type)) {
      const auto ctx_type_iter = ctx_type_to_task_types.find(static_cast<rtFftsPlusContextType_t>(ctx_type));
      if (ctx_type_iter != ctx_type_to_task_types.end()) {
        prof_extend_infos_[node_id].engine_type = ctx_type_iter->second;
      }
    }
  }
  GELOGI("[Cann Profiling] Op(%s) prof task type:%u, is fftsplus task:%d.", op_desc->GetName().c_str(),
         prof_extend_infos_[node_id].engine_type, static_cast<int32_t>(is_fftsplus_task));
}

// 反序列化零拷贝DfxExtendInfo内容
void CannProfilerV2::ReadDfxExtendInfo(const ge::ExecuteGraph *const execute_graph,
                     std::unordered_map<uint64_t, DfxExtendInfo *> &node_name_to_dfx_extend_info) {
  ge::Buffer buffer;
  if (!ge::AttrUtils::GetZeroCopyBytes(execute_graph, "DfxExtendInfo", buffer)) {
    GELOGW("Failed to get DfxExtendInfo buffer from compute graph[%s]", execute_graph->GetName().c_str());
    return;
  }

  auto dfx_extend_infos = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
  if (dfx_extend_infos == nullptr) {
    return;
  }

  if (!ge::AttrUtils::GetZeroCopyBytes(execute_graph, "buffer", buffer)) {
    GELOGW("Failed to get node name buffer from compute graph");
    return;
  }
  auto node_name_bufferpool = reinterpret_cast<const ContinuousBuffer *>(buffer.GetData());
  if (node_name_bufferpool == nullptr) {
    return;
  }

  for (size_t i = 0; i < dfx_extend_infos->GetNum(); ++i) {
    auto dfx_extend_info = dfx_extend_infos->Get<DfxExtendInfo>(i);
    if (dfx_extend_info == nullptr) {
      GELOGW("Failed to get dfx_extend_info pod from buffer");
      return;
    }
    auto node_name = node_name_bufferpool->Get<char>(
        reinterpret_cast<size_t>(dfx_extend_info->GetNodeName()));
    // 以hash id方式加入map
    const uint64_t node_name_hash = MsprofGetHashId(node_name, strlen(node_name));
    node_name_to_dfx_extend_info.emplace(node_name_hash, const_cast<DfxExtendInfo*>(dfx_extend_info));
  }
}

void CannProfilerV2::InitSingleCMOContextInfo(const ge::OpDescPtr &op_desc,
                                              const std::vector<uint32_t> &cmo_context_ids,
                                              std::string &cmo_context_name, uint32_t task_type, size_t cmo_idx) {
  const auto node_iter =
      name_hash_to_node_id_.find(MsprofGetHashId(op_desc->GetNamePtr(), op_desc->GetName().length()));
  if (node_iter == name_hash_to_node_id_.end()) {
    GELOGW("[Cann Profiling] Op(%s) name hash is not found.", op_desc->GetName().c_str());
    return;
  }

  ProfNodeAdditionInfo info{};
  info.node_basic_info.level = MSPROF_REPORT_NODE_LEVEL;
  info.node_basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  auto &prof_node_basic_info = info.node_basic_info.data.nodeBasicInfo;
  auto cmo_name_hash = MsprofGetHashId(cmo_context_name.c_str(), cmo_context_name.length());
  prof_node_basic_info.opName = cmo_name_hash;
  prof_node_basic_info.taskType = task_type;

  // cmo context information
  info.context_info.level = MSPROF_REPORT_NODE_LEVEL;
  info.context_info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  info.context_info.dataLen = sizeof(MsprofContextIdInfo);
  auto context = reinterpret_cast<MsprofContextIdInfo *>(info.context_info.data);
  context->ctxIdNum = cmo_context_ids.size();
  for (size_t i = 0UL; i < cmo_context_ids.size(); i++) {
    context->ctxIds[i] = cmo_context_ids[i];
  }
  context->opName = cmo_name_hash;
  if (name_hashes_to_node_addition_infos_.find(cmo_name_hash) == name_hashes_to_node_addition_infos_.end()) {
    name_hashes_to_node_addition_infos_[cmo_name_hash] = std::move(info);
    node_addition_infos_[node_iter->second][cmo_idx + 1] = &name_hashes_to_node_addition_infos_[cmo_name_hash];
  }
}

void CannProfilerV2::InitVectorCoreProfInfo(const ge::OpDescPtr &op_desc) {
  const std::string *mix_core_type = ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  if ((mix_core_type == nullptr) || (*mix_core_type != "MIX_AICORE" && *mix_core_type != "MIX_VECTOR_CORE")) {
    return;
  }

  const uint64_t name_hash = MsprofGetHashId(op_desc->GetNamePtr(), op_desc->GetName().length());
  auto iter = name_hash_to_node_id_.find(name_hash);
  if (iter == name_hash_to_node_id_.end()) {
    GELOGW("[Cann Profiling] Op(%s) name hash is not found.", op_desc->GetNamePtr());
    return;
  }
  auto addition_info_iter = name_hashes_to_node_addition_infos_.find(name_hash);
  if (addition_info_iter == name_hashes_to_node_addition_infos_.end()) {
    GELOGW("[Cann Profiling] Op(%s) name hash is not found.", op_desc->GetNamePtr());
    return;
  }

  const std::string vector_name = op_desc->GetName() + "__VECTORCORE__";
  const uint64_t vector_name_hash = MsprofGetHashId(vector_name.c_str(), vector_name.length());
  name_hashes_to_input_shapes_[vector_name_hash] = name_hashes_to_input_shapes_[name_hash];
  name_hashes_to_output_shapes_[vector_name_hash] = name_hashes_to_output_shapes_[name_hash];
  name_hashes_to_node_addition_infos_[vector_name_hash] = addition_info_iter->second;
  name_hashes_to_node_addition_infos_[vector_name_hash].node_basic_info.data.nodeBasicInfo.taskType =
      static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV);

  exe_node_id_to_profiling_wrappers_[iter->second].SetProfNodeAdditionInfo(
      NodeProfInfoType::kMixVectorCore, &name_hashes_to_node_addition_infos_[vector_name_hash]);
  node_addition_infos_[iter->second][static_cast<size_t>(NodeProfInfoType::kMixVectorCore)] =
      &name_hashes_to_node_addition_infos_[vector_name_hash];
}

void CannProfilerV2::InitNodeCMOContextInfo(const ge::OpDescPtr &op_desc) {
  std::vector<uint32_t> cmo_context_ids;
  std::string cmo_context_name;
  uint32_t cmo_context_type;
  for (size_t idx = 0; idx < CMO_IDX_KEY.size(); ++idx) {
    if (!ge::AttrUtils::GetListInt(op_desc, kDataProfCtxIdV + CMO_IDX_KEY[idx], cmo_context_ids)) {
      continue;
    }
    const auto ctx_num = cmo_context_ids.size();
    if (ctx_num == 0 || ctx_num > MSPROF_CTX_ID_MAX_NUM) {
      GELOGW("[Cann Profiling] Op(%s) context size [%zu] is invalid.", op_desc->GetName().c_str(), ctx_num);
      continue;
    }
    if (!ge::AttrUtils::GetStr(op_desc, kDataProfName + CMO_IDX_KEY[idx], cmo_context_name)) {
      continue;
    }
    if (!ge::AttrUtils::GetInt(op_desc, kDataProfType + CMO_IDX_KEY[idx], cmo_context_type)) {
      continue;
    }
    const auto it = ctx_type_to_task_types.find(static_cast<rtFftsPlusContextType_t>(cmo_context_type));
    if (it == ctx_type_to_task_types.end()) {
      GELOGW("[Cann Profiling] Op(%s) task type %u is invalid.", op_desc->GetName().c_str(), cmo_context_type);
      continue;
    }
    InitSingleCMOContextInfo(op_desc, cmo_context_ids, cmo_context_name, it->second, idx);
  }
}

ge::Status CannProfilerV2::InitNodeAtomicContextInfo(const ge::OpDescPtr &op_desc) {
  std::string atomic_kernel_name;
  if (!ge::AttrUtils::GetStr(op_desc, "_atomic_kernelname", atomic_kernel_name) || atomic_kernel_name.empty()) {
    return ge::SUCCESS;
  }

  const uint64_t node_hash = MsprofGetHashId(op_desc->GetNamePtr(), op_desc->GetName().length());
  auto atomic_hash_iter = node_hash_to_atomic_node_hash_.find(node_hash);
  GE_ASSERT_TRUE(atomic_hash_iter != node_hash_to_atomic_node_hash_.end(),
                 "[Cann Profiling] Op(%s) name has no atomic node.", op_desc->GetNamePtr());
  const uint64_t atomic_hash = atomic_hash_iter->second;
  auto node_id_iter = name_hash_to_node_id_.find(atomic_hash);
  GE_ASSERT_TRUE(node_id_iter != name_hash_to_node_id_.end(),
                 "[Cann Profiling] The atomic name hash for op(%s) is not found.", op_desc->GetNamePtr());
  auto atomic_info_iter = name_hashes_to_node_addition_infos_.find(atomic_hash);
  GE_ASSERT_TRUE(atomic_info_iter != name_hashes_to_node_addition_infos_.end(),
                 "[Cann Profiling] Op(%s) name has no atomic addition info.", op_desc->GetNamePtr());

  ProfNodeAdditionInfo &add_info = atomic_info_iter->second;
  std::string op_type = "MemSet";
  constexpr char const *kAtomicCoreTypeKey = "_atomic_cube_vector_core_type";
  MsprofGeTaskType memset_core_type = MSPROF_GE_TASK_TYPE_AIV;
  std::string memset_core_type_attr;
  if (ge::AttrUtils::GetStr(op_desc, kAtomicCoreTypeKey, memset_core_type_attr) && !memset_core_type_attr.empty()) {
    GELOGI("[Cann Profiling] Memset core type attr is %s.", memset_core_type_attr.c_str());
    if (memset_core_type_attr == "AIC") {
      memset_core_type = MSPROF_GE_TASK_TYPE_AI_CORE;
    }
  }
  add_info.node_basic_info.data.nodeBasicInfo.opType = MsprofGetHashId(op_type.c_str(), op_type.length());
  prof_extend_infos_[node_id_iter->second].engine_type = static_cast<uint32_t>(memset_core_type);
  std::vector<uint32_t> ctx_ids;
  if (ge::AttrUtils::GetListInt(op_desc, "_atomic_context_id_list", ctx_ids) && !ctx_ids.empty()) {
    add_info.context_info.level = MSPROF_REPORT_NODE_LEVEL;
    add_info.context_info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    add_info.context_info.dataLen = sizeof(MsprofContextIdInfo);
    auto context = reinterpret_cast<MsprofContextIdInfo *>(add_info.context_info.data);
    context->ctxIdNum = ctx_ids.size();
    for (size_t i = 0UL; i < ctx_ids.size(); i++) {
      context->ctxIds[i] = ctx_ids[i];
    }
    context->opName = atomic_hash;
  }
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::InitProfInfoByGraphNode(const ge::NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  InitNodeContextInfo(op_desc);
  InitNodeCMOContextInfo(op_desc);
  GE_ASSERT_SUCCESS(InitNodeAtomicContextInfo(op_desc));
  InitVectorCoreProfInfo(op_desc);
  return ge::SUCCESS;
}

void CannProfilerV2::InitBasicInfo(const uint64_t node_id, const DfxExtendInfo *dfx_extend_info,
                                   MsprofCompactInfo &basic_info) const {
  basic_info.level = MSPROF_REPORT_NODE_LEVEL;
  basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  auto &prof_node_basic_info = basic_info.data.nodeBasicInfo;
  prof_node_basic_info.opName = prof_extend_infos_[node_id].node_name_idx;
  prof_node_basic_info.opType = prof_extend_infos_[node_id].node_type_idx;

  if ((dfx_extend_info != nullptr) && (*(dfx_extend_info->GetOpImplModeAddr()) == kEnableHf32)) {
    prof_node_basic_info.opFlag = kOpImplHf32ModeEnable;
  } else {
    prof_node_basic_info.opFlag = kOpImplHf32ModeDisable;
  }
}

Chain *CannProfilerV2::InitOutputChainFromEquivalentDataEdges(
    const bg::ValueHolderPtr &out_shape,
    std::unordered_map<std::string, ge::FastNode *> &kernel_names_2_exe_nodes) const {
  if (!bg::ValueHolderUtils::IsNodeValid(out_shape)) {
    return nullptr;
  }
  const auto &name_and_idx =
      GetKernelNameAndIdxAfterPass(bg::ValueHolderUtils::GetNodeOpDescBarePtr(out_shape),
                                   {bg::ValueHolderUtils::GetNodeName(out_shape), out_shape->GetOutIndex()});
  ge::FastNode *node = nullptr;
  const auto &iter = kernel_names_2_exe_nodes.find(name_and_idx.kernel_name);
  if (iter != kernel_names_2_exe_nodes.cend()) {
    node = iter->second;
  } else {
    GELOGW("[Cann profiling] Can not find node %s", name_and_idx.kernel_name.c_str());
    return nullptr;
  }
  GE_ASSERT_NOTNULL(node->GetExtendInfo());
  const auto symbol = node->GetExtendInfo()->GetOutputSymbol(name_and_idx.idx);
  GE_ASSERT_TRUE(symbol != ge::kInvalidSymbol,
                 "[Cann Profiling] Can not find out kernel [%s] from equivalent data edges.", node->GetNamePtr());
  const auto value_iter = extend_info_->symbols_to_value.find(symbol);
  GE_ASSERT_TRUE(value_iter != extend_info_->symbols_to_value.end(),
                 "[Cann Profiling] Can not find out kernel [%s] from symbol to values.", node->GetNamePtr());
  return reinterpret_cast<Chain *>(value_iter->second);
}

void CannProfilerV2::InitShapes() {
  for (auto &name_hash_to_tensor_info : name_hashes_to_node_addition_infos_) {
    const auto input_iter = name_hashes_to_input_shapes_.find(name_hash_to_tensor_info.first);
    if (input_iter == name_hashes_to_input_shapes_.end()) {
      continue;
    }
    const auto output_iter = name_hashes_to_output_shapes_.find(name_hash_to_tensor_info.first);
    if (output_iter == name_hashes_to_output_shapes_.end()) {
      continue;
    }
    size_t tensor_count = 0UL;
    std::for_each(name_hash_to_tensor_info.second.tensor_info_wrappers.cbegin(),
                  name_hash_to_tensor_info.second.tensor_info_wrappers.cend(),
                  [&tensor_count](const TensorInfoWrapper &tensor_info_wrapper) {
                    tensor_count += tensor_info_wrapper.tensor_num;
                  });

    if ((input_iter->second.size() + output_iter->second.size()) != tensor_count) {
      GELOGW("[Cann Profiling] shape size [%zu] does not equal to tensor size [%zu].",
             input_iter->second.size() + output_iter->second.size(), tensor_count);
      continue;
    }
    name_hash_to_tensor_info.second.input_tensor_num = input_iter->second.size();
    name_hash_to_tensor_info.second.output_tensor_num = output_iter->second.size();
    if (!name_hash_to_tensor_info.second.json_info.empty()) {
      const auto iter = name_hash_to_node_id_.find(name_hash_to_tensor_info.first);
      std::vector<std::vector<int64_t>> input_shapes, output_shapes;
      if ((iter != name_hash_to_node_id_.end()) &&
          (GetShapeInfoFromJson(name_hash_to_tensor_info.second.json_info, input_shapes, output_shapes) ==
           ge::SUCCESS)) {
        (void)exe_node_id_to_profiling_wrappers_[iter->second].FillShapeInfo(input_shapes, output_shapes);
        GELOGI("[Cann Profiling] Shape filled by attr for node %u.", iter->second);
      }
    }
    for (size_t i = 0UL; i < input_iter->second.size(); ++i) {
      const auto tensor_idx = i / (static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_NUM));
      name_hash_to_tensor_info.second.tensor_info_wrappers[tensor_idx].shapes.emplace_back(input_iter->second[i]);
    }
    for (size_t i = 0UL; i < output_iter->second.size(); ++i) {
      const auto tensor_idx = (i + input_iter->second.size()) / (static_cast<uint64_t>(MSPROF_GE_TENSOR_DATA_NUM));
      name_hash_to_tensor_info.second.tensor_info_wrappers[tensor_idx].shapes.emplace_back(output_iter->second[i]);
    }
  }
}

ge::Status CannProfilerV2::InitOutputShapeInfo(const ge::NodePtr &node, std::vector<Chain *> &output_shapes,
                                               std::unordered_map<std::string, ge::FastNode *> &kernel_names_2_exe_nodes) {
  const auto placed_lower_result =
      node->GetOpDescBarePtr()->TryGetExtAttr(kLoweringResult, PlacedLoweringResult(nullptr, LowerResult()));
  for (const auto &out_shape : placed_lower_result.GetResult()->out_shapes) {
    const auto out_chain = InitOutputChainFromEquivalentDataEdges(out_shape, kernel_names_2_exe_nodes);
    if (out_chain != nullptr) {
      output_shapes.emplace_back(out_chain);
    }
  }
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::InitInputShapeInfo(const ge::NodePtr &node, const uint64_t name_hash,
                                              std::vector<Chain *> &input_shapes) {
  if (name_hashes_to_node_addition_infos_.find(name_hash) == name_hashes_to_node_addition_infos_.end()) {
    return ge::SUCCESS;
  }
  const size_t input_num = node->GetOpDescBarePtr()->GetAllInputsSize();
  for (size_t i = 0UL; i < input_num; ++i) {
    const auto &in_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    if (in_anchor == nullptr) {
      continue;
    }
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    const auto peer_node = out_anchor->GetOwnerNodeBarePtr();
    if (peer_node == nullptr) {
      continue;
    }
    const auto &out_idx = out_anchor->GetIdx();
    const auto peer_output_shapes = name_hashes_to_output_shapes_.find(
        MsprofGetHashId(peer_node->GetNamePtr(), peer_node->GetName().length()));
    if (peer_output_shapes == name_hashes_to_output_shapes_.end()) {
      continue;
    }
    if (static_cast<size_t>(out_idx) >= peer_output_shapes->second.size()) {
      GELOGW("[Cann Profiling] out idx [%d] is over peer node out_shapes size [%zu].", out_idx,
             peer_output_shapes->second.size());
      continue;
    }
    const auto &out_shape = peer_output_shapes->second[out_idx];
    input_shapes.emplace_back(out_shape);
  }
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::InitShapeInfo(const ge::NodePtr &node,
                                         std::unordered_map<std::string, ge::FastNode *> &kernel_names_2_exe_nodes) {
  const uint64_t name_hash = MsprofGetHashId(node->GetNamePtr(), node->GetName().length());
  auto &output_shapes = name_hashes_to_output_shapes_[name_hash];
  auto &input_shapes = name_hashes_to_input_shapes_[name_hash];
  InitOutputShapeInfo(node, output_shapes, kernel_names_2_exe_nodes);
  InitInputShapeInfo(node, name_hash, input_shapes);
  std::string ffts_str;
  const auto ret = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), ffts::kAttrSgtJsonInfo, ffts_str);
  if (ret && !ffts_str.empty()) {
    auto iter = name_hash_to_node_id_.find(name_hash);
    if (iter != name_hash_to_node_id_.end()) {
      node_addition_infos_[iter->second][0U]->json_info = ffts_str;
    }
  }
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::InitShapeAndContextInfo(const ge::ComputeGraphPtr &graph) {
  std::unordered_map<std::string, ge::FastNode *> kernel_names_2_exe_nodes{};
  for (const auto exe_node : extend_info_->exe_graph->GetAllNodes()) {
    kernel_names_2_exe_nodes[exe_node->GetName()] = exe_node;
  }
  for (auto &node : graph->GetAllNodes()) {
    if (IsNoNeedProfNodes(node)) {
      continue;
    }
    (void)InitShapeInfo(node, kernel_names_2_exe_nodes);
    (void)InitProfInfoByGraphNode(node);
  }

  return ge::SUCCESS;
}

void CannProfilerV2::InitLaunchApi(const uint64_t name_hash, const char *kernel_type, MsprofApi &api) const {
  auto prof_kernel_type = SubscriberUtils::GetProfKernelType(kernel_type, true);
  api.type = prof_kernel_type;
  api.level = MSPROF_REPORT_NODE_LEVEL;
  api.itemId = name_hash;
}

void CannProfilerV2::SetEngineType(NodeIdentity node_id, const std::string &kernel_type,
                                   const DfxExtendInfo *dfx_extend_info) {
  ProfExtendInfo &info = prof_extend_infos_[node_id];
  if (IsAiCoreLaunchNode(kernel_type.c_str())) {
    uint32_t engine_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE);
    if (dfx_extend_info != nullptr) {
      const auto attr_engine_type = dfx_extend_info->GetTaskTypeAddr();
      if ((attr_engine_type != nullptr) && (*attr_engine_type == static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AIV))) {
        engine_type = *attr_engine_type;
      }
    }
    info.engine_type = engine_type;
    info.block_dim_input_idx = static_cast<uint32_t>(kernel::InputCommon::kBlockDim);
  } else if (IsAiCpuLaunchNode(kernel_type.c_str())) {
    info.engine_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CPU);
  } else if (IsHcomLaunchNode(kernel_type.c_str())) {
    info.engine_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_HCCL);
  } else if (IsHostAicpuCpuLaunchNode(kernel_type.c_str())) {
    info.engine_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CPU);
  } else if (IsCustomOpFuncNode(kernel_type.c_str())) {
    // 临时方案：当前自定义算子默认设置为aic，正式方案走runtime上报
    info.engine_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE);
  } else {
    info.engine_type = std::numeric_limits<uint32_t>::max();
  }
  GELOGI("[Cann Profiling] node_id %zu, EngineType: %u", node_id, info.engine_type);
}

ge::Status CannProfilerV2::InitInfoByExecutionNode(uint64_t name_hash, NodeIdentity node_id, const char *kernel_type,
                                                   const ComputeNodeInfo *compute_node_info,
                                                   const DfxExtendInfo *dfx_extend_info) {
  GE_ASSERT_NOTNULL(compute_node_info);
  ProfNodeAdditionInfo addition_info{};
  InitBasicInfo(node_id, dfx_extend_info, addition_info.node_basic_info);
  if (!IsHcomLaunchNode(kernel_type) && prof_extend_infos_[node_id].origin_name_hash_for_hash == UINT64_MAX) {
    InitTensorInfo(compute_node_info, prof_extend_infos_[node_id].node_name_idx, addition_info.tensor_info_wrappers);
  }
  InitLaunchApi(name_hash, kernel_type, addition_info.api);
  name_hashes_to_node_addition_infos_[name_hash] = std::move(addition_info);
  const auto add_info_ptr = &name_hashes_to_node_addition_infos_[name_hash];
  exe_node_id_to_profiling_wrappers_[node_id] = CannProfilingInfoWrapper(add_info_ptr);
  node_addition_infos_[node_id][static_cast<size_t>(NodeProfInfoType::kOriginalNode)] = add_info_ptr;
  name_hashes_to_node_addition_infos_[name_hash].node_basic_info.data.nodeBasicInfo.taskType =
      prof_extend_infos_[node_id].engine_type;

  GELOGI("[Cann Profiling] node_id: %zu, name_hash: %llu, kernel_type: %s, engine_type: %u", node_id, name_hash,
         kernel_type, prof_extend_infos_[node_id].engine_type);
  return ge::SUCCESS;
}

void CannProfilerV2::FormatNodeIdMap(const std::unordered_map<uint64_t, std::vector<NodeIdentity>>
                                         &name_hash_to_node_ids) {
  for (auto hash_iter = name_hash_to_node_ids.begin(); hash_iter != name_hash_to_node_ids.end(); ++hash_iter) {
    auto id_iter = name_hash_to_node_id_.find(hash_iter->first);
    if (id_iter == name_hash_to_node_id_.end()) {
      continue;
    }
    for (size_t i = 0; i < hash_iter->second.size(); ++i) {
      node_id_to_profiler_node_id_[hash_iter->second[i]] = id_iter->second;
    }
  }
}

ge::Status CannProfilerV2::InitBasicInfoAndTensorInfo(
    const ExecutionData &execution_data, const ge::ComputeGraphPtr &compute_graph,
    const std::unordered_map<uint64_t, DfxExtendInfo *> &node_name_to_dfx_extend_info) {
  node_addition_infos_.resize(execution_data.base_ed.node_num, {nullptr});
  exe_node_id_to_profiling_filler_.resize(execution_data.base_ed.node_num, nullptr);
  exe_node_id_to_profiling_wrappers_.resize(execution_data.base_ed.node_num);
  node_id_to_profiler_node_id_.resize(execution_data.base_ed.node_num, UINT32_MAX);

  ProfilerRegistry &ins = ProfilerRegistry::GetInstance();
  std::unordered_map<uint64_t, std::vector<NodeIdentity>> name_hash_to_node_ids{};

  for (size_t i = 0UL; i < execution_data.base_ed.node_num; ++i) {
    const auto node = execution_data.base_ed.nodes[i];
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_TRUE(node->node_id < execution_data.base_ed.node_num);
    const auto node_id = node->node_id;
    const auto name_hash = prof_extend_infos_[node_id].node_name_idx;
    name_hash_to_node_ids[name_hash].emplace_back(node_id);
    node_hash_to_atomic_node_hash_[prof_extend_infos_[node_id].origin_name_hash_for_hash] =
        prof_extend_infos_[node_id].node_name_idx;
    const char *kernel_type = nullptr;
    const ComputeNodeInfo *compute_node_info = nullptr;
    GE_ASSERT_SUCCESS(ParseContextByNode(node, &kernel_type, &compute_node_info));
    if (compute_node_info == nullptr) {
      GELOGW("Kernel %s has no compute node info.", kernel_type);
      continue;
    }
    if (ins.IsProfDavinciModelExecuteType(kernel_type)) {
      davinci_model_node_ids_.insert(node_id);
      continue;
    }
    const auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel_type);
    exe_node_id_to_profiling_filler_[node_id] =
        (kernel_funcs != nullptr ? kernel_funcs->profiling_info_filler : nullptr);
    if (!ins.IsProfLaunchType(kernel_type) && !ins.IsProfLaunchType(kernel_type, false)) {
      continue;
    }

    DfxExtendInfo *dfx_extend_info = nullptr;
    const auto iter = node_name_to_dfx_extend_info.find(prof_extend_infos_[node_id].node_name_idx);
    if (iter != node_name_to_dfx_extend_info.end()) {
      dfx_extend_info = iter->second;
    }
    SetEngineType(node_id, kernel_type, dfx_extend_info);

    // some op have 2 nodes, one aic/aiv launch and one aicpu launch
    // the aicpu node only need kOriginalNode info
    if (name_hashes_to_node_addition_infos_.find(name_hash) != name_hashes_to_node_addition_infos_.end()) {
      node_addition_infos_[node_id][static_cast<size_t>(NodeProfInfoType::kOriginalNode)] =
          &name_hashes_to_node_addition_infos_[name_hash];
      GELOGI("Node %zu, kernel %s name hash %lu is exits.", node_id, kernel_type, name_hash);
      continue;
    }
    name_hash_to_node_id_[name_hash] = node_id;
    GE_ASSERT_SUCCESS(InitInfoByExecutionNode(name_hash, node_id, kernel_type, compute_node_info, dfx_extend_info));
  }
  FormatNodeIdMap(name_hash_to_node_ids);
  GE_ASSERT_SUCCESS(InitShapeAndContextInfo(compute_graph));
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::InitForCannDevice(const ExecutionData *execution_data) {
  InitNameAndTypeWithHash(*execution_data);
  const auto compute_graph =
      extend_info_->exe_graph->TryGetExtAttr(kComputeGraph, std::make_shared<ge::ComputeGraph>(kInvalidGraphName));
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_TRUE(compute_graph->GetName() != kInvalidGraphName);
  std::unordered_map<uint64_t, DfxExtendInfo *> node_name_to_dfx_extend_info = {};
  ReadDfxExtendInfo(extend_info_->exe_graph.get(), node_name_to_dfx_extend_info);
  GE_ASSERT_SUCCESS(InitBasicInfoAndTensorInfo(*execution_data, compute_graph, node_name_to_dfx_extend_info));
  InitShapes();
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::RecordNodeBasicInfo(NodeIdentity node_id, uint64_t prof_time, int32_t tid) const {
  GE_ASSERT_TRUE(node_id < prof_extend_infos_.size());
  GE_ASSERT_TRUE(node_id < node_addition_infos_.size());
  const auto addition_info = node_addition_infos_[node_id][0U];
  GE_ASSERT_NOTNULL(addition_info);
  auto basic_info = &addition_info->node_basic_info;
  basic_info->timeStamp = prof_time;
  basic_info->threadId = static_cast<uint32_t>(tid);
  basic_info->data.nodeBasicInfo.taskType = prof_extend_infos_[node_id].engine_type;
  GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(true, basic_info, sizeof(MsprofCompactInfo)));
  GELOGI("[Cann Profiling] node_id: %zu, type %u, threadId %u, opname %llu, taskType %u, opType %llu, "
         "blockDim %u, opFlag %u.",
         node_id, basic_info->type, basic_info->threadId, basic_info->data.nodeBasicInfo.opName,
         basic_info->data.nodeBasicInfo.taskType, basic_info->data.nodeBasicInfo.opType,
         basic_info->data.nodeBasicInfo.blockDim, basic_info->data.nodeBasicInfo.opFlag);
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::RecordTensorInfo(const uint64_t prof_time, const int32_t tid,
                                            ProfNodeAdditionInfo &addition_info) const {
  for (auto &tensor_info_wrapper : addition_info.tensor_info_wrappers) {
    tensor_info_wrapper.tensor_info.timeStamp = prof_time;
    tensor_info_wrapper.tensor_info.threadId = static_cast<uint32_t>(tid);
    if (!tensor_info_wrapper.shapes.empty() && !tensor_info_wrapper.filled_by_kernel) {
      tensor_info_wrapper.FillShapeInfo();
    }
    GE_ASSERT_MSPROF_OK(
        MsprofReportAdditionalInfo(true, &tensor_info_wrapper.tensor_info, sizeof(MsprofAdditionalInfo)));
  }
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::DoProfByNodeId(NodeIdentity report_node, uint64_t prof_time, uint32_t tid) {
  const auto &addition_infos = node_addition_infos_[report_node];
  if (GetGlobalProf()->IsEnabled(ProfilingType::kDevice) &&
      prof_extend_infos_[report_node].engine_type != std::numeric_limits<uint32_t>::max()) {
    (void)RecordNodeBasicInfo(report_node, prof_time, tid);
    (void)RecordTensorInfo(prof_time, tid, *addition_infos[static_cast<size_t>(NodeProfInfoType::kOriginalNode)]);
  }

  for (size_t i = static_cast<size_t>(NodeProfInfoType::kOriginalNode); i < addition_infos.size(); ++i) {
    const auto addition_info = addition_infos[i];
    if (addition_info == nullptr) {
      continue;
    }
    if ((i != static_cast<size_t>(NodeProfInfoType::kOriginalNode)) &&
        (GetGlobalProf()->IsEnabled(ProfilingType::kDevice)) &&
        (addition_info->node_basic_info.type == MSPROF_REPORT_NODE_BASIC_INFO_TYPE) &&
        (addition_info->node_basic_info.level != 0)) {
      addition_info->node_basic_info.timeStamp = prof_time;
      addition_info->node_basic_info.threadId = static_cast<uint32_t>(tid);
      GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(true, &addition_info->node_basic_info, sizeof(MsprofCompactInfo)));
    }
    if (addition_info->context_info.type == MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE) {
      addition_info->context_info.threadId = static_cast<uint32_t>(tid);
      addition_info->context_info.timeStamp = prof_time;
      GE_ASSERT_MSPROF_OK(MsprofReportAdditionalInfo(true, &addition_info->context_info, sizeof(MsprofAdditionalInfo)));
    }
  }

  return ge::SUCCESS;
}

ge::Status CannProfilerV2::ReportMixLaunchKernel(const NodeIdentity node_id) {
  thread_local static int32_t tid = mmGetTid();
  auto addition_info = node_addition_infos_[node_id][static_cast<size_t>(NodeProfInfoType::kOriginalNode)];
  addition_info->api.threadId = tid;
  addition_info->api.type = prof_extend_infos_[node_id].kernel_prof_type;
  (void)MsprofReportApi(true, &addition_info->api);

  GELOGI("[Cann Profiling] node_id: %zu, mix_launch_enable: %d, prof type: %u", node_id,
         static_cast<int32_t>(addition_info->mix_launch_enable), addition_info->api.type);

  const bool enable_prof = GetGlobalProf()->IsEnabled(ProfilingType::kDevice);
  if (enable_prof && prof_extend_infos_[node_id].engine_type != std::numeric_limits<uint32_t>::max()) {
    auto basic_info = &addition_info->node_basic_info;
    basic_info->timeStamp = addition_info->api.endTime;
    basic_info->threadId = static_cast<uint32_t>(tid);
    basic_info->data.nodeBasicInfo.taskType = prof_extend_infos_[node_id].engine_type;
    GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(true, basic_info, sizeof(MsprofCompactInfo)));
    (void)RecordTensorInfo(addition_info->api.endTime, tid, *addition_info);
  }
  if (!addition_info->mix_launch_enable) {
    return ge::SUCCESS;
  }

  auto mix_vector_addition_info = node_addition_infos_[node_id][static_cast<size_t>(NodeProfInfoType::kMixVectorCore)];
  mix_vector_addition_info->api.threadId = tid;
  (void)MsprofReportApi(true, &mix_vector_addition_info->api);

  if (enable_prof) {
    auto basic_info = &mix_vector_addition_info->node_basic_info;
    basic_info->timeStamp = mix_vector_addition_info->api.endTime;
    basic_info->threadId = static_cast<uint32_t>(tid);
    GE_ASSERT_MSPROF_OK(MsprofReportCompactInfo(true, basic_info, sizeof(MsprofCompactInfo)));
    (void)RecordTensorInfo(mix_vector_addition_info->api.endTime, tid, *mix_vector_addition_info);
  }
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::DoProf(const Node *node) {
  GELOGI("[Cann Profiling] node_id %zu.", node->node_id);
  if (davinci_model_node_ids_.count(node->node_id) > 0) {
    const auto context = reinterpret_cast<const KernelContext *>(&node->context);
    GE_CHECK_NOTNULL(context);
    auto davinci_model =
        context->MutableInputPointer<ge::DavinciModel>(static_cast<int32_t>(gert::kernel::InputsCommon::kDavinciModel));
    if ((davinci_model != nullptr) &&
        davinci_model->GetReportedProfCount() != GlobalProfilingWrapper::GetInstance()->GetProfCount()) {
      GELOGI("[Cann Profiling] ReportedProfCount %llu, ProfCount %llu.", davinci_model->GetReportedProfCount(),
             GlobalProfilingWrapper::GetInstance()->GetProfCount());
      (void)davinci_model->ReportProfilingData();
      davinci_model->SetReportedProfCount(GlobalProfilingWrapper::GetInstance()->GetProfCount());
    }
    return ge::SUCCESS;
  }

  GE_ASSERT_TRUE(node->node_id < node_addition_infos_.size());
  const auto filler = exe_node_id_to_profiling_filler_[node->node_id];
  const auto wrapper_id = node_id_to_profiler_node_id_[node->node_id];
  if ((filler != nullptr) && (wrapper_id != UINT32_MAX)) {
    const auto ctx = reinterpret_cast<const KernelContext *>(&node->context);
    auto &wrapper = static_cast<CannProfilingInfoWrapper &>(exe_node_id_to_profiling_wrappers_[wrapper_id]);
    GE_ASSERT_SUCCESS(filler(ctx, wrapper));
  }

  auto addition_info = node_addition_infos_[node->node_id][static_cast<size_t>(NodeProfInfoType::kOriginalNode)];
  if (addition_info == nullptr) {
    return ge::SUCCESS;
  }

  cur_launch_node_ids_.emplace_back(node->node_id);
  if (addition_info->api.type == kInvalidProfKernelType) {
    return ge::SUCCESS;
  }

  // report mix type
  auto mix_vector_addition_info =
      node_addition_infos_[node->node_id][static_cast<size_t>(NodeProfInfoType::kMixVectorCore)];
  if (mix_vector_addition_info != nullptr) {
    cur_launch_node_ids_.clear();
    return ReportMixLaunchKernel(node->node_id);
  }

  // todo:这块代码应该被上面CannProfilingInfoWrapper的逻辑收编，当前避免影响性能先不修改
  const auto block_dim_input_idx = prof_extend_infos_[node->node_id].block_dim_input_idx;
  if ((block_dim_input_idx != UINT32_MAX) && (wrapper_id != UINT32_MAX)) {
    const auto block_dim =
        *reinterpret_cast<const KernelContext *>(&node->context)->GetInputPointer<uint32_t>(block_dim_input_idx);
    exe_node_id_to_profiling_wrappers_[wrapper_id].SetBlockDim(block_dim);
  }

  const auto prof_time = MsprofSysCycleTime();
  thread_local static int32_t tid = mmGetTid();

  // some op have 2 nodes, current node may have differet type with the prof node
  addition_info->api.type = prof_extend_infos_[node->node_id].kernel_prof_type;
  addition_info->api.endTime = prof_time;
  addition_info->api.threadId = tid;
  (void)MsprofReportApi(true, &addition_info->api);
  GELOGI(
      "[Cann Profiling] node_id %zu, type %u, threadId %u, beginTime %llu, endTime %llu, itemId %llu, prof node %zu.",
      node->node_id, addition_info->api.type, addition_info->api.threadId, addition_info->api.beginTime,
      addition_info->api.endTime, addition_info->api.itemId, wrapper_id);
  for (const auto report_node_id : cur_launch_node_ids_) {
    GE_ASSERT_SUCCESS(DoProfByNodeId(report_node_id, prof_time, static_cast<uint32_t>(tid)));
  }
  cur_launch_node_ids_.clear();
  return ge::SUCCESS;
}

ge::Status CannProfilerV2::RecordLaunchBeginTime(const Node &node) {
  GE_ASSERT_TRUE(node.node_id < node_addition_infos_.size());
  auto time_stamp = MsprofSysCycleTime();
  const auto addition_info = node_addition_infos_[node.node_id][0U];
  if (addition_info != nullptr && addition_info->api.type != kInvalidProfKernelType) {
    addition_info->api.beginTime = time_stamp;
  }
  return ge::SUCCESS;
}

void CannProfilerV2::OnExecuteEvent(int32_t sub_exe_graph_type, CannProfilerV2 *profiler, ExecutorEvent event,
                                    const void *node, KernelStatus result) {
  (void)result;
  if (profiler == nullptr) {
    return;
  }

  if (sub_exe_graph_type == static_cast<int32_t>(kInitExeGraph)) {
    if (event == kModelStart) {
      profiler->GetGlobalProf()->IncProfModelId();
    }
    return;
  }

  if (sub_exe_graph_type == static_cast<int32_t>(kDeInitExeGraph)) {
    return;
  }

  if (event == kExecuteStart) {
    (void)profiler->RecordLaunchBeginTime(*static_cast<const Node *>(node));
    return;
  }

  if (event == kExecuteEnd) {
    (void)profiler->DoProf(static_cast<const Node *>(node));
    return;
  }

  if (event == kModelStart) {
    profiler->Init();
    return;
  }

  if (event == kModelEnd) {
    profiler->IncreaseIterationNum();
    return;
  }
}
}  // namespace gert
