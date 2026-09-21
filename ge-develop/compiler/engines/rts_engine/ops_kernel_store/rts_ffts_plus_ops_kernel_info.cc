/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rts_ffts_plus_ops_kernel_info.h"
#include <string>
#include "common/util.h"

#include "common/constant/constant.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "op/op_ffts_plus_factory.h"
#include "proto/task.pb.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
using domi::TaskDef;
using std::map;
using std::string;

Status RtsFftsPlusOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  (void)options;
  RTS_LOGI("RtsOpsFftsPlusKernelInfoStore init start.");
  bool isSupportFlag = false;
  auto ret = IsSupportFftsPlus(isSupportFlag);
  if (ret != SUCCESS) {
    return FAILED;
  }

  if (!isSupportFlag) {
    return SUCCESS;
  }

  OpInfo defaultOpInfo = {};
  defaultOpInfo.engine = RTS_FFTS_PLUS_ENGINE_NAME;
  defaultOpInfo.opKernelLib = RTS_FFTS_PLUS_OP_KERNEL_LIB_NAME;
  defaultOpInfo.computeCost = 0;
  defaultOpInfo.flagPartial = false;
  defaultOpInfo.flagAsync = false;
  defaultOpInfo.isAtomic = false;

  // init op_info_map_
  auto allOps = OpFftsPlusFactory::Instance().GetAllOps();
  for (auto &op : allOps) {
    opInfoMap_[op] = defaultOpInfo;
  }
  RTS_LOGI("RtsFftsPlusOpsKernelInfoStore inited success. op num=%zu", opInfoMap_.size());
  return SUCCESS;
}

Status RtsFftsPlusOpsKernelInfoStore::Finalize() {
  opInfoMap_.clear();
  return SUCCESS;
}

void RtsFftsPlusOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const {
  infos = opInfoMap_;
}

bool RtsFftsPlusOpsKernelInfoStore::CheckSupported(const OpDescPtr &opDesc, std::string &) const {
  return opInfoMap_.count(opDesc->GetType()) > 0;
}

Status RtsFftsPlusOpsKernelInfoStore::CreateSession(const map<string, string> &sessionOptions) {
  (void)sessionOptions;
  // do nothing
  return SUCCESS;
}

Status RtsFftsPlusOpsKernelInfoStore::DestroySession(const map<string, string> &sessionOptions) {
  (void)sessionOptions;
  // do nothing
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
