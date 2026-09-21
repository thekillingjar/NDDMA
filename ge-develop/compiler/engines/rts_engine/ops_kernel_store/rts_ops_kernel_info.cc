/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "./rts_ops_kernel_info.h"
#include <string>

#include "common/constant/constant.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "op/op_factory.h"
#include "proto/task.pb.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

Status RtsOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  (void)options;
  RTS_LOGI("RtsOpsKernelInfoStore init start.");

  OpInfo defaultOpInfo = {};
  defaultOpInfo.engine = RTS_ENGINE_NAME;
  defaultOpInfo.opKernelLib = RTS_OP_KERNEL_LIB_NAME;
  defaultOpInfo.computeCost = 0;
  defaultOpInfo.flagPartial = false;
  defaultOpInfo.flagAsync = false;
  defaultOpInfo.isAtomic = false;

  // init op_info_map_
  auto allOps = OpFactory::Instance().GetAllOps();
  for (auto &op : allOps) {
    op_info_map_[op] = defaultOpInfo;
  }

  RTS_LOGI("RtsOpsKernelInfoStore inited success. op num=%zu", op_info_map_.size());

  return SUCCESS;
}

Status RtsOpsKernelInfoStore::Finalize() {
  op_info_map_.clear();
  return SUCCESS;
}

void RtsOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, OpInfo> &infos) const {
  infos = op_info_map_;
}

bool RtsOpsKernelInfoStore::CheckSupported(const OpDescPtr &opDesc, std::string &) const {
  return op_info_map_.count(opDesc->GetType()) > 0;
}

Status RtsOpsKernelInfoStore::CreateSession(const map<string, string> &session_options) {
  (void)session_options;
  // do nothing
  return SUCCESS;
}

Status RtsOpsKernelInfoStore::DestroySession(const map<string, string> &session_options) {
  (void)session_options;
  // do nothing
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce
