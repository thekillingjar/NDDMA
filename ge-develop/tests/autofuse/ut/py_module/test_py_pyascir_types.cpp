/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"

#include "pyascir.h"
#include "pyascir_types.h"
#include "pyascir_common_utils.h"

class TestPyAscirTypes : public testing::Test {
 protected:
  virtual void SetUp() {
  }

  virtual void TearDown() {
  }
};

TEST_F(TestPyAscirTypes, DISABLED_HintComputeGraph_GetName) {
  PyObject* graph_obj = pyascir::HintComputeGraph::New(&pyascir::HintComputeGraph::type, nullptr, nullptr);
  EXPECT_NE(graph_obj, nullptr);
  pyascir::HintComputeGraph::Init(graph_obj, nullptr, nullptr);
  PyObject* graph_name = pyascir::HintComputeGraph::GetName(graph_obj);
  EXPECT_STREQ(PyUnicode_AsUTF8(graph_name), "fused_graph");
  pyascir::HintComputeGraph::Dealloc(graph_obj);
}

TEST_F(TestPyAscirTypes, DISABLED_FusedScheduledResult_IsCubeType) {
  PyObject *obj = pyascir::FusedScheduledResult::New(&pyascir::FusedScheduledResult::type, nullptr, nullptr);
  EXPECT_NE(obj, nullptr);
  pyascir::FusedScheduledResult::Init(obj, nullptr, nullptr);

  auto self = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(obj);

  std::vector<ascir::ScheduledResult> results1;
  ascir::ScheduledResult result1;
  result1.cube_type = ascir::CubeTemplateType::kFixpip;
  results1.push_back(result1);
  self->fused_schedule_result.node_idx_to_scheduled_results.push_back(results1);
  PyObject *res = pyascir::FusedScheduledResult::IsCubeType(obj);
  EXPECT_EQ(res, Py_True);
  pyascir::FusedScheduledResult::Dealloc(obj);
}

TEST_F(TestPyAscirTypes, DISABLED_FusedScheduledResult_IsCubeType_False) {
  PyObject *obj = pyascir::FusedScheduledResult::New(&pyascir::FusedScheduledResult::type, nullptr, nullptr);
  EXPECT_NE(obj, nullptr);
  pyascir::FusedScheduledResult::Init(obj, nullptr, nullptr);

  auto self = reinterpret_cast<pyascir::FusedScheduledResult::Object *>(obj);

  std::vector<ascir::ScheduledResult> results1;
  ascir::ScheduledResult result1;
  result1.cube_type = ascir::CubeTemplateType::kDefault;
  results1.push_back(result1);
  self->fused_schedule_result.node_idx_to_scheduled_results.push_back(results1);
  PyObject *res = pyascir::FusedScheduledResult::IsCubeType(obj);
  EXPECT_EQ(res, Py_False);
  pyascir::FusedScheduledResult::Dealloc(obj);
}

TEST_F(TestPyAscirTypes, DISABLED_FusedScheduledResult_IsCubeType_Null) {
  PyObject *obj = pyascir::FusedScheduledResult::New(&pyascir::FusedScheduledResult::type, nullptr, nullptr);
  EXPECT_NE(obj, nullptr);
  pyascir::FusedScheduledResult::Init(obj, nullptr, nullptr);
  PyObject *res = pyascir::FusedScheduledResult::IsCubeType(nullptr);
  EXPECT_EQ(res, Py_False);
  pyascir::FusedScheduledResult::Dealloc(obj);
}
