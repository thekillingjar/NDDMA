/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_HCOM_OPS_KERNEL_INFO_STORE_H
#define AUTO_TUNING_HCOM_OPS_KERNEL_INFO_STORE_H

#include "hcom_ops_kernel_info_store.h"
#include <map>
#include "acl/acl_rt.h"

namespace hccl {
struct AclHostMemDeleter {
  void operator()(void *ptr) const {
    if (ptr != nullptr) {
      (void)hrtFreeHost(ptr);
    }
  }
};

using AclHostMemPtr = std::unique_ptr<void, AclHostMemDeleter>;

// Ge适配的类
class AutoTuningHcomOpsKernelInfoStore : public HcomOpsKernelInfoStore {
 public:
  AutoTuningHcomOpsKernelInfoStore();
  ~AutoTuningHcomOpsKernelInfoStore() override;
  // load task for op
  ge::Status LoadTask(ge::GETaskInfo &task) override;
  ge::Status UnloadTask(ge::GETaskInfo &task) override;

 private:
  HcclResult SetCustomKernelInfo(ge::OpInfo &opinfo, std::map<string, ge::OpInfo> &infos) const override;
  HcclResult CheckPrivateDef(const ge::GETaskInfo &task) override;
  HcclResult GetDataTypeFromTaskInfo(const ge::GETaskInfo &task, HcclDataType &dataType) const override;
  HcclResult GetRankSizeFromTaskInfo(const ge::GETaskInfo &task, uint32_t &rankSize);
  HcclResult GetHcclInfo(const ge::GETaskInfo &task, ge::GETaskKernelHcclInfo &hcclInfo);
  HcclResult HCCLOpsKernel(const ge::GETaskInfo &task, const std::string &sCollectiveType);
  HcclResult HcomBroadcastOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomAllReduceOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomAllGatherOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomAlltoAllVOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomAlltoAllVCOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomAlltoAllOpKernel(const ge::GETaskInfo &task);
  HcclResult GetOriginalGraphShapeTypeFromTaskInfo(const ge::GETaskInfo &task, u32 &shapeType) override;
  // add
  HcclResult HcomReduceOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomReduceScatterOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomSendOpKernel(const ge::GETaskInfo &task);
  HcclResult HcomReceiveOpKernel(const ge::GETaskInfo &task);
  HcclResult InitAlltoAllHostMem(HcclDataType recvType, u64 recvCount, void *hostMemEmpty);

  std::map<u64, std::map<HcclDataType, AclHostMemPtr>> mapCountTypeHostMem_;
};
}  // namespace hccl
#endif  // AUTO_TUNING_HCOM_OPS_KERNEL_INFO_STORE_H
