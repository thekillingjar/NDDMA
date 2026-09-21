/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_executor.h"
#include "model_manager.h"
#include "maintain_manager.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "maintain_manager.h"
#include "runtime/stream.h"
#include "runtime/mem.h"
#include "runtime/rt_model.h"
static void *ModelGetIoAddr(uint32_t in_nums, uint32_t out_nums,
                            const InputData *input_data,
                            OutputData *output_data, GeModelDesc *mdlDesc) {
  uint32_t ioa_size = in_nums + out_nums;
  GELOGI("output->ioa_size:%u, ioa_size:%u", output_data->ioa_size, ioa_size);
  if ((output_data->ioa_size < ioa_size) && (output_data->io_addr != NULL)) {
    rtFree(output_data->io_addr);
    output_data->io_addr = NULL;
  }
  uint32_t fifo_num = mdlDesc->fifoInfo.fifoNum;
  uint64_t addr_size = sizeof(uint64_t) * (ioa_size + fifo_num);
  if (output_data->io_addr == NULL) {
    if (rtMalloc((void **)&output_data->io_addr, addr_size, mdlDesc->memType, 0) != RT_ERROR_NONE) {
      output_data->io_addr = NULL;
      output_data->ioa_size = 0;
      return NULL;
    }
    output_data->ioa_size = ioa_size + fifo_num;
  }
  if (output_data->io_addr_host == NULL) {
    output_data->io_addr_host = mmMalloc(addr_size);
    if (output_data->io_addr_host == NULL) {
      (void)rtFree(output_data->io_addr);
      output_data->io_addr = NULL;
      output_data->ioa_size = 0;
      return NULL;
    }
  }
  uint64_t *ioa_src_addr = output_data->io_addr;
  uint64_t *ioa_src_addr_host = output_data->io_addr_host;
  uint32_t index = 0;
  for (size_t i = 0UL; i < in_nums; ++i) {
    ioa_src_addr_host[index++] =
        (uint64_t)(uintptr_t)(((const DataBlob *)(ConstVectorAt(&input_data->blobs, i)))->dataBuffer->data);
  }
  for (size_t i = 0UL; i < out_nums; ++i) {
    ioa_src_addr_host[index++] =
        (uint64_t)(uintptr_t)(((DataBlob *)(VectorAt(&output_data->blobs, i)))->dataBuffer->data);
  }
  for (size_t i = 0UL; i < fifo_num; ++i) {
    ioa_src_addr_host[index++] = mdlDesc->fifoInfo.fifoAllAddr[i];
  }
  if (rtMemcpy(ioa_src_addr, addr_size, ioa_src_addr_host, addr_size, RT_MEMCPY_HOST_TO_DEVICE) != RT_ERROR_NONE) {
    (void)rtFree(output_data->io_addr);
    (void)mmFree(output_data->io_addr_host);
    output_data->io_addr = NULL;
    output_data->io_addr_host = NULL;
    output_data->ioa_size = 0;
    return NULL;
  }
  return (void *)ioa_src_addr;
}

static Status ModelExecute(ExecHandleDesc *execDesc, bool sync, const InputData *input_data, OutputData *output_data,
                           GeModelDesc *mdlDesc) {
  uint32_t sqId;
  Status rtRet = (Status)rtStreamGetSqid((rtStream_t)execDesc->stream, &sqId);
  if (rtRet != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "get sqId failed ret[%d].", rtRet);
    return rtRet;
  }

  uint32_t mdl_in_nums = (uint32_t)VectorSize(&mdlDesc->ioInfo.input_desc);
  uint32_t cur_in_nums = (uint32_t)VectorSize(&input_data->blobs);
  uint32_t mdl_out_nums = (uint32_t)VectorSize(&mdlDesc->ioInfo.output_desc);
  uint32_t cur_out_nums = (uint32_t)VectorSize(&output_data->blobs);
  if ((mdl_in_nums > cur_in_nums) || (mdl_out_nums > cur_out_nums)) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  void *ioa_src_addr = ModelGetIoAddr(mdl_in_nums, mdl_out_nums, input_data, output_data, mdlDesc);
  if (ioa_src_addr == NULL) {
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  rtMdlExecute_t rtMdlExec = {0};
  rtMdlExec.sqid = sqId;
  rtMdlExec.vld = 1;
  rtMdlExec.taskProf = GetProfEnable();
  rtMdlExec.ioaSrcAddr = ioa_src_addr;
  rtMdlExec.dynamicTaskPtr = mdlDesc->part.dynTaskPtr;
  rtMdlExec.workPtr = execDesc->workPtr;
  rtMdlExec.mid = (uint8_t)mdlDesc->phyModelId;
  rtMdlExec.sync = sync;
  rtMdlExec.ioaSize = output_data->ioa_size; // (input + output + fifo) num
  rtMdlExec.mpamId = execDesc->mpamId;
  rtMdlExec.mecTimeThreshHold = execDesc->mecTimeThreshHold;
  rtMdlExec.aicQos = execDesc->aicQos;
  rtMdlExec.aicOst = execDesc->aicOst;

  rtRet = (Status)rtNanoModelExecute(&rtMdlExec);
  if (rtRet != SUCCESS) {
    return rtRet;
  }
  StepIdConuterPlus(mdlDesc->modelDbgHandle);
  return SUCCESS;
}

Status ModelExecuteInner(uint32_t modelId, ExecHandleDesc *execDesc, bool sync, const InputData *input_data,
                         OutputData *output_data) {
  GeModelDesc *mdlDesc = GetModelDescRef(modelId);
  if (mdlDesc == NULL) {
    return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
  }
  Status ret = ModelExecute(execDesc, sync, input_data, output_data, mdlDesc);
  ReleaseModelDescRef(mdlDesc);
  return ret;
}
