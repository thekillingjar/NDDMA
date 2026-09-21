/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H39E4E719_91F4_4D0F_BA4F_6BA56CB1E20D
#define H39E4E719_91F4_4D0F_BA4F_6BA56CB1E20D

#include "fake_ns.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "info_store_holder.h"

FAKE_NS_BEGIN

struct FakeOpsKernelBuilder : OpsKernelBuilder, InfoStoreHolder {
  FakeOpsKernelBuilder(const std::string &kernel_lib_name);
  FakeOpsKernelBuilder();

private:
  using CalcOpParamCall = std::function<graphStatus(const Node &node)>;
  static std::map<std::string, CalcOpParamCall> calc_op_param_call;

 protected:
  Status Initialize(const map<std::string, std::string> &options) override;
  Status Finalize() override;
  Status CalcOpRunningParam(Node &node) override;
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override;
};

FAKE_NS_END

#endif
