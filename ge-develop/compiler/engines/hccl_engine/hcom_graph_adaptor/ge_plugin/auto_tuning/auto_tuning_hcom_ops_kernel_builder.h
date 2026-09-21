/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_HCOM_OPS_KERNEL_BUILDER_H
#define AUTO_TUNING_HCOM_OPS_KERNEL_BUILDER_H

#include "ops_kernel_builder_base.h"
#include "op_hcom_comm.h"

namespace hccl {
// Ge适配的类
class AutoTuningHcomOpsKernelBuilder : public HCCLOpsKernelBuilder {
 public:
  AutoTuningHcomOpsKernelBuilder();
  ~AutoTuningHcomOpsKernelBuilder() override;
  // memory allocation requirement
  ge::Status CalcOpRunningParam(ge::Node &node) override;
  // generate taskinfo for op
  ge::Status GenerateTask(const ge::Node &node, ge::RunContext &runContext,
                          std::vector<domi::TaskDef> &taskDefList) override;

 private:
  HcclResult GetSupportedOP(std::vector<std::string> &hcclSupportOp) const override;
  HcclResult GetRankSizeFromDesc(const ge::OpDescPtr &op, uint32_t &rankSize, const std::string &sCollectiveType);
  HcclResult GetOriginalGraphShapeTypeFromDesc(const ge::OpDescPtr &op, u32 &shapeType);
};

using AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF = struct autoTuningHcclKernelInfoPrivateDef_ {
  u32 rankSize;
  u32 originalGraphShapeType;
  u64 outputBytes;
  HcclDataType dataType;
};
const std::vector<std::string> AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE = {
    HCCL_KERNEL_OP_TYPE_BROADCAST, HCCL_KERNEL_OP_TYPE_ALLREDUCE,     HCCL_KERNEL_OP_TYPE_ALLGATHER,
    HCCL_KERNEL_OP_TYPE_REDUCE,    HCCL_KERNEL_OP_TYPE_REDUCESCATTER, HCCL_KERNEL_OP_TYPE_SEND,
    HCCL_KERNEL_OP_TYPE_RECEIVE,   HCCL_KERNEL_OP_TYPE_ALLTOALLV,     HCCL_KERNEL_OP_TYPE_ALLTOALLVC,
    HCCL_KERNEL_OP_TYPE_ALLTOALL};
}  // namespace hccl
#endif  // AUTO_TUNING_HCOM_OPS_KERNEL_BUILDER_H