/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/engine/base_engine.h"
#include "graph_metadef/register/register.h"

ge::RunContext CreateContext();

void DestroyContext(ge::RunContext& context);

namespace aicpu{
class TestForGetSoPath{
public:
    TestForGetSoPath() = default;
    ~TestForGetSoPath() = default;
    static TestForGetSoPath &Instance();
};
}

namespace ge {
class HostCpuTestOp : public HostCpuOp {
public:
  HostCpuTestOp() = default;
  ~HostCpuTestOp() override = default;
  graphStatus Compute(Operator &op,
                      const std::map<std::string, const Tensor> &inputs,
                      std::map<std::string, Tensor> &outputs) override;
};
}
