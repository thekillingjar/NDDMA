/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/aicore/nano_aicore_task_info.h"
#include <map>
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"
#include "common/preload/task_info/pre_task_status.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
Status GenArgsInfo(const std::vector<uint64_t> &offset_list, rtParamBufDesc_t &param_buf_desc,
                   const std::vector<KernelArgsParam> &args_param) {
  GELOGD("begin to generate args info.");
  const size_t offset_size = offset_list.size();
  GE_ASSERT_TRUE((offset_size == args_param.size()), "offset_list size[%llu] is not equal with args_param size[%zu]",
                 offset_size, args_param.size());
  for (size_t i = 0U; i < offset_size; ++i) {
    const uint64_t offset = offset_list.at(i);
    param_buf_desc.paramBufInfo[param_buf_desc.paramBufSize] = offset;
    param_buf_desc.paramBufSize++;
    GELOGD("offset[%llu] is normal offset.", offset);
  }
  GELOGD("success generate args info.");
  return SUCCESS;
}

void GenPrefetchInfo(const OpDescPtr &op_desc, rtParamBufDesc_t &param_buf_desc) {
  GELOGD("begin to generate prefetch info.");
  std::vector<std::string> weight_prefetch_type;
  const bool ret = ge::AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, weight_prefetch_type);
  if (ret) {
    std::vector<int64_t> weight_prefetch_src_offset;
    std::vector<int64_t> weight_prefetch_dst_offset;
    std::vector<int64_t> weight_prefetch_data_size;
    (void)ge::AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchSrcOffset, weight_prefetch_src_offset);
    (void)ge::AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDstOffset, weight_prefetch_dst_offset);
    (void)ge::AttrUtils::GetListInt(op_desc, kAttrWeightPrefetchDataSize, weight_prefetch_data_size);
    if ((weight_prefetch_type.size() != weight_prefetch_src_offset.size()) ||
        (weight_prefetch_type.size() != weight_prefetch_dst_offset.size()) ||
        (weight_prefetch_type.size() != weight_prefetch_data_size.size())) {
      GELOGW(
          "weight prefetch size not equal, type size[%zu], src_offset size[%zu], dst_offset size[%zu], data_size "
          "size[%zu]",
          weight_prefetch_type.size(), weight_prefetch_src_offset.size(), weight_prefetch_dst_offset.size(),
          weight_prefetch_data_size.size());
      return;
    }
    for (size_t i = 0U; i < weight_prefetch_type.size(); ++i) {
      rtPrefetchBufInfo_t &prefetch_info = param_buf_desc.prefetchBufInfo[param_buf_desc.prefetchBufSize];
      int32_t op_type = 0;
      (void)ge::ConvertToInt32(weight_prefetch_type.at(i), op_type);
      prefetch_info.opType = static_cast<uint32_t>(op_type);
      prefetch_info.dataSize = static_cast<uint32_t>(weight_prefetch_data_size.at(i));
      prefetch_info.dstOffset = static_cast<uint32_t>(weight_prefetch_dst_offset.at(i));
      prefetch_info.srcOffset = static_cast<uint32_t>(weight_prefetch_src_offset.at(i));
      GELOGD("op[%s] prefetch info opType:%u, dataSize:%u, dstOffset:0x%x, srcOffset:0x%x", op_desc->GetName().c_str(),
             prefetch_info.opType, prefetch_info.dataSize, prefetch_info.dstOffset, prefetch_info.srcOffset);
      param_buf_desc.prefetchBufSize++;
    }
  }
  GELOGD("success generate prefetch info.");
}

void GenParamBufDesc(const OpDescPtr &op_desc, const PreTaskInput &pre_task_input, rtParamBufDesc_t &param_buf_desc) {
  std::vector<KernelArgsParam> args_param;
  std::vector<uint64_t> args_offset_vals;
  GELOGD("begin to generate input param buf desc.");
  const auto input_data_addr_offset =
      PreModelUtils::GetInputDataAddrOffset(pre_task_input.rts_param, op_desc, args_param, args_offset_vals);
  (void)GenArgsInfo(args_offset_vals, param_buf_desc, args_param);

  GELOGD("begin to generate output param buf desc.");
  args_param.clear();
  args_offset_vals.clear();
  const auto output_data_addr_offset =
      PreModelUtils::GetOutputDataAddrOffset(pre_task_input.rts_param, op_desc, args_param, args_offset_vals);
  (void)GenArgsInfo(args_offset_vals, param_buf_desc, args_param);

  GELOGD("begin to generate workspace param buf desc.");
  args_param.clear();
  args_offset_vals.clear();
  const auto workspace_data_addr_offset =
      PreModelUtils::GetWorkspaceDataAddrOffset(pre_task_input.rts_param, op_desc, args_param, args_offset_vals);
  (void)GenArgsInfo(args_offset_vals, param_buf_desc, args_param);

  GenPrefetchInfo(op_desc, param_buf_desc);
  GELOGD("success generate param buf desc.");
}

void GenerateNanoAiCoreStaticTaskDesc(rtHwtsStaticTaskDesc_t &hwts_static_task_desc,
                                      const OpDescPtr &op_desc, ModelTaskType task_type) {
  hwts_static_task_desc.type = static_cast<uint16_t>(NanoTaskDescType::NANO_AI_CORE);
  bool sw_enable = false;
  (void)ge::AttrUtils::GetBool(op_desc, "_slow_context_switch", sw_enable);
  GELOGD("get attr _slow_context_switch success, sw_enable is [%s]", sw_enable? "true" : "false");
  hwts_static_task_desc.sw = sw_enable;
  string switch_buffer_type;
  if (ge::AttrUtils::GetStr(op_desc, "_switch_buffer_type", switch_buffer_type)) {
    GELOGD("get attr switch_buffer_type success, switch_buffer_type is [%s]", switch_buffer_type.c_str());
    hwts_static_task_desc.uf = static_cast<uint16_t>(switch_buffer_type.compare("UF") == 0 ? 1U : 0U);
  }

  if ((task_type == ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX) || (task_type == ModelTaskType::MODEL_TASK_STREAM_LABEL_GOTO)) {
    hwts_static_task_desc.conds = static_cast<uint16_t>(DEFAULT_INFO_VALUE_ONE);
  }

  std::vector<std::string> weight_prefetch_type;
  if (ge::AttrUtils::GetListStr(op_desc, kAttrWeightPrefetchType, weight_prefetch_type)) {
    hwts_static_task_desc.prefetchNum = static_cast<uint16_t>(weight_prefetch_type.size());
    GELOGD("get attr weight_prefetch_type success, prefetchNum: %u", hwts_static_task_desc.prefetchNum);
  }
}

bool PreValidTaskDescInfo(PreTaskResult &result, const OpDescPtr &op_desc, const std::unique_ptr<PreTaskDescInfo> &pre_task_desc_info) {
  if (op_desc == nullptr) {
    result.status = PreTaskStatus::ErrorStatus("op_desc is nullptr.");
    return false;
  }
  GELOGD("Init generate nano aicore task %s.", op_desc->GetName().c_str());
  if (pre_task_desc_info == nullptr) {
    result.status = PreTaskStatus::ErrorStatus("pre_task_desc_info is nullptr.");
    return false;
  }
  return true;
}
} // namespace

PreTaskResult GenerateNanoAiCoreTask(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                     const PreTaskInput &pre_task_input) {
  PreTaskResult result;
  std::unique_ptr<PreTaskDescInfo> pre_task_desc_info = ge::MakeUnique<PreTaskDescInfo>();
  if (!PreValidTaskDescInfo(result, op_desc, pre_task_desc_info)) {
    return result;
  }
  const domi::KernelDef &kernel_def = task_def.kernel();
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  // step1. static task desc
  pre_task_desc_info->seq_info.taskType = RT_TASK_TYPE_KERNEL_NANO_AICORE;
  pre_task_desc_info->seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info->seq_info.u.nanoAicoreTask.type = HWTS_STATIC_TASK_DESC;
  pre_task_desc_info->seq_info.bufType = HWTS_STATIC_TASK_DESC;
  // preP posP dump conds uf sw prefetchNum softUser kernelCredit init 0
  // taskParamOffset init 0 and need refresh for rts
  rtHwtsStaticTaskDesc_t hwts_static_task_desc = {};
  GenerateNanoAiCoreStaticTaskDesc(hwts_static_task_desc, op_desc, task_type);
  pre_task_desc_info->seq_info.u.nanoAicoreTask.u.hwtsTaskDesc = hwts_static_task_desc;
  result.pre_task_desc_infos.push_back(*pre_task_desc_info);

  // step2. dynamic task desc
  pre_task_desc_info->seq_info.taskType = RT_TASK_TYPE_KERNEL_NANO_AICORE;
  pre_task_desc_info->seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info->seq_info.u.nanoAicoreTask.type = HWTS_DYNAMIC_TASK_DESC;
  pre_task_desc_info->seq_info.bufType = HWTS_DYNAMIC_TASK_DESC;
  rtHwtsDynamicTaskDesc_t hwts_dynamic_task_desc = {};
  hwts_dynamic_task_desc.vld = static_cast<uint8_t>(DEFAULT_INFO_VALUE_ONE);
  hwts_dynamic_task_desc.codeSize = DEFAULT_INFO_VALUE_EIGHT;
  hwts_dynamic_task_desc.dynTaskDescSize = DEFAULT_INFO_VALUE_ONE;
  hwts_dynamic_task_desc.blockDim = (kernel_def.block_dim() == 0U) ? 1U : kernel_def.block_dim();
  const auto itr = pre_task_input.names_to_bin_offset.find(kernel_def.kernel_name());
  if (itr != pre_task_input.names_to_bin_offset.end()) {
    hwts_dynamic_task_desc.taskPcOffset = itr->second;
  } else {
    result.status = PreTaskStatus::ErrorStatus("can't find bin offset from op_name[%s] kernel_name[%s]",
                                               op_desc->GetName().c_str(), kernel_def.kernel_name().c_str());
    return result;
  }
  pre_task_desc_info->seq_info.u.nanoAicoreTask.u.hwtsDynamicTaskDesc = hwts_dynamic_task_desc;
  result.pre_task_desc_infos.push_back(*pre_task_desc_info);

  // step3. task param info desc
  pre_task_desc_info->seq_info.taskType = RT_TASK_TYPE_KERNEL_NANO_AICORE;
  pre_task_desc_info->seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info->seq_info.u.nanoAicoreTask.type = PARAM_TASK_INFO_DESC;
  pre_task_desc_info->seq_info.bufType = PARAM_TASK_INFO_DESC;
  rtParamBufDesc_t &param_buf_desc = pre_task_desc_info->seq_info.u.nanoAicoreTask.u.paramBufDesc;
  param_buf_desc.prefetchBufSize = 0U;
  param_buf_desc.paramBufSize = 0U;
  GenParamBufDesc(op_desc, pre_task_input, param_buf_desc);
  result.pre_task_desc_infos.push_back(*pre_task_desc_info);

  result.status = PreTaskStatus::Success();
  GELOGD("Init generate nano aicore task success");
  return result;
}

REFISTER_PRE_GENERATE_TASK(kPreEngineNanoAiCore, GenerateNanoAiCoreTask);
}  // namespace ge
