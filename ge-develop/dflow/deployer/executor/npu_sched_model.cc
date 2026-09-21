/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/inc/data_flow/deploy/npu_sched_model.h"
#include "acl/acl.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "executor/npu_sched_model_loader.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/df_chk.h"

namespace {
constexpr uint32_t kDummyQId = UINT32_MAX;
constexpr uint32_t kDynamicFlag = 2U;
}

uint32_t InitializeNpuSched(int32_t device_id) {
  DF_CHK_ACL_RET(aclrtSetDevice(device_id));
  GELOGI("Npu sched init success, device_id = %d.", device_id);
  return 0U;
}

NpuSchedModelHandler CreateNpuSchedModelHandler() {
  return new (std::nothrow) ge::NpuSchedModelLoader();
}

uint32_t LoadNpuSchedModel(NpuSchedModelHandler handler, NpuSchedLoadParam *load_param) {
  auto loader = static_cast<ge::NpuSchedModelLoader *>(handler);
  GE_CHECK_NOTNULL(loader);
  GE_CHECK_NOTNULL(load_param);
  loader->SetModelId(load_param->model_id);
  loader->SetDeviceId(load_param->device_id);
  auto inputs_num = load_param->model_queue_param.input_queues_attrs.size();
  std::vector<uint32_t> output_queue_ids;
  auto outputs_num = std::count_if(load_param->model_queue_param.output_queues_attrs.begin(),
                                   load_param->model_queue_param.output_queues_attrs.end(),
                                   [&output_queue_ids](const ge::QueueAttrs &attr) {
    if (attr.queue_id == kDummyQId) {
      return false;
    }
    output_queue_ids.emplace_back(attr.queue_id);
    return true;
  });
  loader->SetEnablePostProcessV2Flag(true);
  loader->SetInputDynamicFlags(std::vector<bool>(inputs_num, true));
  loader->SetOutputTensorSizes(std::vector<int64_t>(outputs_num, 0));
  loader->SetOutputDynamicFlags(std::vector<uint32_t>(outputs_num, kDynamicFlag));
  loader->SetOutputQueueIds(output_queue_ids);
  loader->SetInputAlignAttrs(load_param->model_queue_param.input_align_attrs);
  loader->SetSkipMarkStep(true);
  uint32_t runtime_model_id = 0U;
  GE_CHK_STATUS_RET(loader->LoadModel(load_param->model_queue_param, runtime_model_id),
                    "Fail to load npu sched model, device_id = %d, model_id = %u.",
                    load_param->device_id, load_param->model_id);
  load_param->req_msg_queue_id = loader->GetReqMsgQueueId();
  load_param->resp_msg_queue_id = loader->GetRespMsgQueueId();
  GELOGI("Npu sched model load success, device_id = %d, model_id = %u, req_msg_queue_id = %u, resp_msg_queue_id = %u.",
         load_param->device_id, load_param->model_id, load_param->req_msg_queue_id, load_param->resp_msg_queue_id);
  return 0U;
}

uint32_t UnloadNpuSchedModel(NpuSchedModelHandler handler) {
  auto loader = static_cast<ge::NpuSchedModelLoader *>(handler);
  GE_CHECK_NOTNULL(loader);
  GE_CHK_STATUS_RET(loader->UnloadModel(), "Failed to unload npu sched model.");
  GELOGI("Npu sched model unload success.");
  return 0U;
}

uint32_t DestroyNpuSchedModelHandler(NpuSchedModelHandler handler) {
  if (handler != nullptr) {
    auto loader = static_cast<ge::NpuSchedModelLoader *>(handler);
    delete loader;
  }
  return 0U;
}

uint32_t FinalizeNpuSched(int32_t device_id) {
  DF_CHK_ACL_RET(aclrtResetDevice(device_id));
  GELOGI("Npu sched finalize success, device_id = %d.", device_id);
  return 0U;
}