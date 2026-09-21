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
#include <string>
#include "easy_graph/graph/box.h"
#include "easy_graph/graph/node.h"
#include "easy_graph/builder/graph_dsl.h"
#include "easy_graph/builder/box_builder.h"
#include "easy_graph/layout/graph_layout.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_executor.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "rt_error_codes.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/error_tracking/error_tracking.h"

namespace ge {
class UTEST_error_tracking : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() override {
    remove("valid_path");
  }
};

TEST_F(UTEST_error_tracking, SaveSingleOpTaskOpdescInfo_test) {
  auto add = std::make_shared<OpDesc>("Add", ADD);
AttrUtils::SetListStr(add, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add, 1, 0);

  auto add2 = std::make_shared<OpDesc>("Add2", ADD);
  AttrUtils::SetListStr(add2, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add2, 2, 0);
  rtExceptionInfo rt_exception_info;
  rt_exception_info.streamid = 0;
  rt_exception_info.taskid = 1;
  ErrorTrackingCallback(&rt_exception_info);
  EXPECT_EQ(ErrorTracking::GetInstance().single_op_task_to_op_info_.size(), 2UL);
}

TEST_F(UTEST_error_tracking, SaveOpTaskOpdescInfo_full_test) {
  ErrorTracking::GetInstance().single_op_max_count_ = 2;
  auto add = std::make_shared<OpDesc>("Add", ADD);
  AttrUtils::SetListStr(add, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add, 1, 0);

  auto add2 = std::make_shared<OpDesc>("Add2", ADD);
  AttrUtils::SetListStr(add2, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add2, 2, 0);

  auto add3 = std::make_shared<OpDesc>("Add3", ADD);
  AttrUtils::SetListStr(add2, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add3, 3, 0);

  EXPECT_EQ(ErrorTracking::GetInstance().single_op_task_to_op_info_.size(), 2UL);
}

TEST_F(UTEST_error_tracking, SaveSingleOpTaskOpdescInfo_key_test) {
  auto add = std::make_shared<OpDesc>("Add", ADD);
  AttrUtils::SetListStr(add, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
  const TaskKey key1(1, 1, 9, 0);
  ErrorTracking::GetInstance().SaveGraphTaskOpdescInfo(add, key1, 1);

  auto add2 = std::make_shared<OpDesc>("Add2", ADD);
  AttrUtils::SetListStr(add2, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1"});
  const TaskKey key2(1, 1, 10, 1);
  ErrorTracking::GetInstance().SaveGraphTaskOpdescInfo(add2, key2, 1);
  ErrorTrackingOpInfo op_info;
  EXPECT_TRUE(ErrorTracking::GetInstance().GetGraphTaskOpdescInfo(key1, op_info));
  rtExceptionInfo rt_exception_info;
  rt_exception_info.streamid = 1;
  rt_exception_info.taskid = 1;
  rt_exception_info.expandInfo.type = RT_EXCEPTION_FFTS_PLUS;
  rt_exception_info.expandInfo.u.fftsPlusInfo.contextId = 10;
  rt_exception_info.expandInfo.u.fftsPlusInfo.threadId = 1;
  ErrorTrackingCallback(&rt_exception_info);
  EXPECT_EQ(ErrorTracking::GetInstance().graph_task_to_op_info_[1].size(), 2UL);
  ErrorTracking::GetInstance().ClearUnloadedModelOpdescInfo(1);
  EXPECT_EQ(ErrorTracking::GetInstance().graph_task_to_op_info_.size(), 0UL);
}

TEST_F(UTEST_error_tracking, overflow_case_test) {
  auto add = std::make_shared<OpDesc>("Add", ADD);
  AttrUtils::SetListStr(add, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add, 1, 0);

  auto add2 = std::make_shared<OpDesc>("Add2", ADD);
  AttrUtils::SetListStr(add2, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1"});
  ErrorTracking::GetInstance().SaveSingleOpTaskOpdescInfo(add2, 2, 0);
  rtExceptionInfo rt_exception_info;
  rt_exception_info.retcode = ACL_ERROR_RT_AICORE_OVER_FLOW;
  rt_exception_info.streamid = 0;
  rt_exception_info.taskid = 1;
  ErrorTrackingCallback(&rt_exception_info);
  EXPECT_EQ(ErrorTracking::GetInstance().single_op_task_to_op_info_.size(), 2UL);
}

TEST_F(UTEST_error_tracking, update_task_id_success) {
  auto add_op = std::make_shared<OpDesc>("Add", "Add");
  uint32_t model_id = 0;
  uint32_t stream_id = 1;
  uint32_t old_task_id = 100;
  uint32_t new_task_id = 200;

  ErrorTracking::GetInstance().SaveGraphTaskOpdescInfo(add_op, old_task_id, stream_id, model_id);

  auto &task_map = ErrorTracking::GetInstance().graph_task_to_op_info_[model_id];
  TaskKey old_key(old_task_id, stream_id);
  EXPECT_NE(task_map.find(old_key), task_map.end());
  EXPECT_EQ(task_map[old_key].op_name, "Add");

  ErrorTracking::GetInstance().UpdateTaskId(old_task_id, new_task_id, stream_id, model_id);

  TaskKey new_key(new_task_id, stream_id);
  EXPECT_EQ(task_map.find(old_key), task_map.end());
  EXPECT_NE(task_map.find(new_key), task_map.end());
  EXPECT_EQ(task_map[new_key].op_name, "Add");
}

TEST_F(UTEST_error_tracking, update_task_id_not_found) {
  uint32_t model_id = 0;
  uint32_t stream_id = 1;
  uint32_t old_task_id = 999;
  uint32_t new_task_id = 200;

  auto &task_map = ErrorTracking::GetInstance().graph_task_to_op_info_[model_id];
  size_t initial_size = task_map.size();

  ErrorTracking::GetInstance().UpdateTaskId(old_task_id, new_task_id, stream_id, model_id);

  EXPECT_EQ(task_map.size(), initial_size);
}

TEST_F(UTEST_error_tracking, update_task_id_model_not_found) {
  uint32_t model_id = 0;
  uint32_t null_model_id = 999;
  uint32_t stream_id = 1;
  uint32_t old_task_id = 999;
  uint32_t new_task_id = 200;

  auto &task_map = ErrorTracking::GetInstance().graph_task_to_op_info_[model_id];
  size_t initial_size = task_map.size();


  ErrorTracking::GetInstance().UpdateTaskId(old_task_id, new_task_id, stream_id, null_model_id);

  EXPECT_EQ(task_map.size(), initial_size);
}
}
