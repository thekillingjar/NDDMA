/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include "dlhccl_function.h"

DlHcclFunction::DlHcclFunction() : dl_hccl_handle(nullptr), dl_hcomm_handle(nullptr) {}

DlHcclFunction::~DlHcclFunction() {
  deinit();
}

DlHcclFunction &DlHcclFunction::get_instance() {
  static DlHcclFunction dlHcclFunction;
  return dlHcclFunction;
}

HcclResult DlHcclFunction::init() {
  if (dl_hccl_handle != nullptr && dl_hcomm_handle != nullptr) {
    return HCCL_SUCCESS;
  }

  std::lock_guard<std::mutex> lock(handleMutex_);

  if (dl_hccl_handle == nullptr) {
    dl_hccl_handle = dlopen("libhccl.so", RTLD_LAZY);
    CHK_PRT_RET(dl_hccl_handle == nullptr, HCCL_ERROR("load fail: libhccl.so no found"), HCCL_E_PTR);
  }

  if (dl_hcomm_handle == nullptr) {
    dl_hcomm_handle = dlopen("libhcomm.so", RTLD_LAZY);
    CHK_PRT_RET(dl_hcomm_handle == nullptr, HCCL_ERROR("load fail: libhcomm.so no found"), HCCL_E_PTR);
  }

  // libhccl.so func
  dlHcclAllGatherFunc = (HcclResult (*)(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                                        HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclAllGather");
  CHK_PTR_NULL(dlHcclAllGatherFunc);

  dlHcclAllGatherVFunc =
      (HcclResult (*)(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls,
                      HcclDataType dataType, HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclAllGatherV");
  CHK_PTR_NULL(dlHcclAllGatherVFunc);

  dlHcclAllReduceFunc =
      (HcclResult (*)(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclAllReduce");
  CHK_PTR_NULL(dlHcclAllReduceFunc);

  dlHcclBroadcastFunc = (HcclResult (*)(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                                        aclrtStream stream))dlsym(dl_hccl_handle, "HcclBroadcast");
  CHK_PTR_NULL(dlHcclBroadcastFunc);

  dlHcclReduceScatterFunc =
      (HcclResult (*)(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                      HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclReduceScatter");
  CHK_PTR_NULL(dlHcclReduceScatterFunc);

  dlHcclReduceScatterVFunc =
      (HcclResult (*)(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount,
                      HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                      aclrtStream stream))dlsym(dl_hccl_handle, "HcclReduceScatterV");
  CHK_PTR_NULL(dlHcclReduceScatterVFunc);

  dlHcclAlltoAllVCFunc =
      (HcclResult (*)(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf,
                      HcclDataType recvType, HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclAlltoAllVC");
  CHK_PTR_NULL(dlHcclAlltoAllVCFunc);

  dlHcclAlltoAllVFunc =
      (HcclResult (*)(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                      const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                      HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclAlltoAllV");
  CHK_PTR_NULL(dlHcclAlltoAllVFunc);

  dlHcclAlltoAllFunc = (HcclResult (*)(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                                       const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm,
                                       aclrtStream stream))dlsym(dl_hccl_handle, "HcclAlltoAll");
  CHK_PTR_NULL(dlHcclAlltoAllFunc);

  dlHcclReduceFunc =
      (HcclResult (*)(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclReduce");
  CHK_PTR_NULL(dlHcclReduceFunc);

  dlHcclSendFunc = (HcclResult (*)(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                   HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclSend");
  CHK_PTR_NULL(dlHcclSendFunc);

  dlHcclRecvFunc = (HcclResult (*)(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                   HcclComm comm, aclrtStream stream))dlsym(dl_hccl_handle, "HcclRecv");
  CHK_PTR_NULL(dlHcclRecvFunc);

  // libhcomm.so func
  dlHcomGetandClearOverFlowTasksFunc = (HcclResult (*)(const char *group, hccl::HcclDumpInfo **hcclDumpInfoPtr,
                                                       s32 *len))dlsym(dl_hcomm_handle, "HcomGetandClearOverFlowTasks");
  CHK_PTR_NULL(dlHcomGetandClearOverFlowTasksFunc);

  return HCCL_SUCCESS;
}

void DlHcclFunction::deinit() {
  if (dl_hccl_handle != nullptr) {
    dlclose(dl_hccl_handle);
    dl_hccl_handle = nullptr;
  }

  if (dl_hcomm_handle != nullptr) {
    dlclose(dl_hcomm_handle);
    dl_hcomm_handle = nullptr;
  }
}

HcclResult DlHcclFunction::dlHcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                                           HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  return dlHcclAllReduceFunc(sendBuf, recvBuf, count, dataType, op, comm, stream);
}

HcclResult DlHcclFunction::dlHcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
                                           HcclComm comm, aclrtStream stream) {
  return dlHcclBroadcastFunc(buf, count, dataType, root, comm, stream);
}

HcclResult DlHcclFunction::dlHcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                               HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  return dlHcclReduceScatterFunc(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult DlHcclFunction::dlHcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
                                                void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                                HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  return dlHcclReduceScatterVFunc(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult DlHcclFunction::dlHcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                                           HcclComm comm, aclrtStream stream) {
  return dlHcclAllGatherFunc(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

HcclResult DlHcclFunction::dlHcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
                                            const void *recvDispls, HcclDataType dataType, HcclComm comm,
                                            aclrtStream stream) {
  return dlHcclAllGatherVFunc(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
}

HcclResult DlHcclFunction::dlHcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                      HcclComm comm, aclrtStream stream) {
  return dlHcclSendFunc(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult DlHcclFunction::dlHcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                      HcclComm comm, aclrtStream stream) {
  return dlHcclRecvFunc(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult DlHcclFunction::dlHcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                            const void *recvBuf, HcclDataType recvType, HcclComm comm,
                                            aclrtStream stream) {
  return dlHcclAlltoAllVCFunc(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
}

HcclResult DlHcclFunction::dlHcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
                                           HcclDataType sendType, const void *recvBuf, const void *recvCounts,
                                           const void *rdispls, HcclDataType recvType, HcclComm comm,
                                           aclrtStream stream) {
  return dlHcclAlltoAllVFunc(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm,
                             stream);
}

HcclResult DlHcclFunction::dlHcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                                          const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm,
                                          aclrtStream stream) {
  return dlHcclAlltoAllFunc(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
}

HcclResult DlHcclFunction::dlHcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                                        HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream) {
  return dlHcclReduceFunc(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
}

HcclResult DlHcclFunction::dlHcomGetandClearOverFlowTasks(const char *group, hccl::HcclDumpInfo **hcclDumpInfoPtr,
                                                          s32 *len) {
  return dlHcomGetandClearOverFlowTasksFunc(group, hcclDumpInfoPtr, len);
}