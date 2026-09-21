/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_KERNEL_INFO_STORE__BASEH
#define OPS_KERNEL_INFO_STORE__BASEH

#include "hcom_ops_stores.h"
#include "hccl/base.h"
#include "common/opskernel/ge_task_info.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/node.h"
#include "proto/task.pb.h"
#include "op_common_def.h"
#include "hcom_log.h"
#include "op_hcom_comm.h"

namespace hccl {
// Ge适配的类
class HCCLOpsKernelInfoStore : public ge::OpsKernelInfoStore {
 public:
  HCCLOpsKernelInfoStore();
  ~HCCLOpsKernelInfoStore() override;
  // initialize opsKernelInfoStore
  ge::Status Initialize(const map<string, string> &options) override;
  // close opsKernelInfoStore
  ge::Status Finalize() override;
  // get all opsKernelInfo
  void GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const override;
  // whether the opsKernelInfoStore is supported based on the operator attribute
  bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &unSupportedReason) const override;

 protected:
  virtual HcclResult SetCustomKernelInfo(ge::OpInfo &opinfo, std::map<string, ge::OpInfo> &infos) const = 0;
  virtual HcclResult GetSupportedOP(std::vector<std::string> &hcclSupportOp) const = 0;
  virtual HcclResult GetDataTypeFromTaskInfo(const ge::GETaskInfo &task, HcclDataType &dataType) const = 0;

  HcclResult CheckSupportedOP(const std::string &sCollectiveType) const;
  HcclResult GetCollectiveTypeFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, std::string &sCollectiveType);
  HcclResult GetCountFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, u64 &count);
  HcclResult GetStreamMainFromTaskInfo(const ge::GETaskInfo &taskDef, rtStream_t &stream);
  HcclResult GetInputAddrFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, uintptr_t &inputAddr);
  HcclResult GetOutputAddrFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, uintptr_t &outputAddr);
  HcclResult GetRootFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, u32 &root);
  HcclResult GetReduceTypeFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, HcclReduceOp &opType);
  HcclResult GetGlobalWorkSpaceAddrFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo,
                                                std::vector<void *> &globalWorkSpaceAddr);
};
}  // namespace hccl
#endif  // OPS_KERNEL_INFO_STORE__BASEH
