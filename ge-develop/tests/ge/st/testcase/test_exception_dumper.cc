/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "macro_utils/dt_public_scope.h"
#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "framework/common/types.h"
#include "graph/ge_local_context.h"
#include "common/dump/exception_dumper.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "common/debug/ge_log.h"
#include "runtime/subscriber/global_dumper.h"

namespace ge {
class STEST_exception_dumper : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {
    remove("valid_path");
  }
};

TEST_F(STEST_exception_dumper, dump_exception_constructor) {
  ExceptionDumper exception_dumper;
  EXPECT_EQ(exception_dumper.is_registered_, true);
  EXPECT_EQ(gert::GlobalDumper::GetInstance()->GetInnerExceptionDumpers().count(&exception_dumper), 1);
  ExceptionDumper exception_dumper1{false};
  EXPECT_EQ(exception_dumper1.is_registered_, false);
  EXPECT_EQ(gert::GlobalDumper::GetInstance()->GetInnerExceptionDumpers().count(&exception_dumper1), 0);
}
}  // namespace ge
