/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dirent.h>
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "common/profiling/profiling_properties.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
using namespace std;

class UtestGeProfilingProperties : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestGeProfilingProperties, test_execute_profiling) {
  auto &profiling_properties = ge::ProfilingProperties::Instance();
  profiling_properties.SetExecuteProfiling(true);
  auto is_execute = profiling_properties.IsExecuteProfiling();
  EXPECT_EQ(is_execute, true);
}

TEST_F(UtestGeProfilingProperties, test_training_trace) {
  auto &profiling_properties = ge::ProfilingProperties::Instance();
  profiling_properties.SetTrainingTrace(true);
  auto is_train_trance = profiling_properties.ProfilingTrainingTraceOn();
  EXPECT_EQ(is_train_trance, true);
}

TEST_F(UtestGeProfilingProperties, test_fpbp_point) {
auto &profiling_properties = ge::ProfilingProperties::Instance();
  std::string fp_point = "fp";
  std::string bp_point = "bp";
  profiling_properties.SetFpBpPoint(fp_point, bp_point);
  profiling_properties.GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "fp");
  EXPECT_EQ(bp_point, "bp");
}

TEST_F(UtestGeProfilingProperties, test_profiling_on) {
  auto &profiling_properties = ge::ProfilingProperties::Instance();
  profiling_properties.SetExecuteProfiling(true);
  profiling_properties.SetLoadProfiling(true);
  auto profiling_on = profiling_properties.ProfilingOn();
  EXPECT_EQ(profiling_on, true);
}

TEST_F(UtestGeProfilingProperties, test_IsDynamicShapeProfiling) {
  auto &profiling_properties = ge::ProfilingProperties::Instance();
  bool is_dynamic = profiling_properties.IsDynamicShapeProfiling();
  EXPECT_EQ(is_dynamic, false);
}
