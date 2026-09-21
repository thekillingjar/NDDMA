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

#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/rts/stream_switch_task_info.h"
#include "graph/load/model_manager/memory_app_type_classifier.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
class UtestStreamSwitchTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestStreamSwitchTaskInfo, stream_switch_task_success) {
  const PisToArgs args = {};
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix},
                                 {0x311c, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 0x40000;

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  GeTensorDesc desc;
  auto op_desc = std::make_shared<OpDesc>("stream_switch", STREAMSWITCH);
  AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, 0);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, {0});
  op_desc->SetStreamId(0);
  op_desc->SetId(2);
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  op_desc->SetInputOffset({8, 16});
  model.op_list_[op_desc->GetId()] = op_desc;
  model.SetFeatureBaseRefreshable(true);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::StreamSwitchDef *stream_switch_def = task_def.mutable_stream_switch();
  stream_switch_def->set_op_index(op_desc->GetId());

  StreamSwitchTaskInfo task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  int64_t op_index = task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, op_desc->GetId());
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(PtrToValue(task_info.input_ptr_), 0x1a23);
  EXPECT_EQ(PtrToValue(task_info.value_ptr_), 0x311c);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  task_def.clear_stream_switch();
}

TEST_F(UtestStreamSwitchTaskInfo, stream_switch_task_fail) {
  const PisToArgs args = {};
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix},
                                 {0x311c, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};
  DavinciModel model(0, nullptr);
  model.runtime_param_.mem_size = 0x40000;

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  GeTensorDesc desc;
  auto op_desc = std::make_shared<OpDesc>("stream_switch", STREAMSWITCH);
  op_desc->SetStreamId(0);
  op_desc->SetId(2);
  op_desc->SetInputOffset({8, 16});
  model.op_list_[op_desc->GetId()] = op_desc;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::StreamSwitchDef *stream_switch_def = task_def.mutable_stream_switch();
  stream_switch_def->set_op_index(op_desc->GetId());

  StreamSwitchTaskInfo task_info;
  // fail for STREAM_SWITCH_COND
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), FAILED);

  // ATTR_NAME_SWITCH_DATA_TYPE not Int
  AttrUtils::SetBool(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, RT_SWITCH_INT32);
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), FAILED);
  op_desc->DelAttr(ATTR_NAME_SWITCH_DATA_TYPE);
  AttrUtils::SetInt(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, RT_SWITCH_INT64);

  // fail for input_size
  AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, 0);
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  // fail for ACTIVE_STREAM_LIST
  op_desc->AddInputDesc(desc);
  op_desc->AddInputDesc(desc);
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  // active_stream_list.size() != kTrueBranchStreamNum_1
  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, {});
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  // true_stream_index >= davinci_model_->GetStreamList().size()
  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, {1});
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, {0});
  model.SetFeatureBaseRefreshable(true);
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  // check task_run_param
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 2);
  EXPECT_NE(task_run_param.parsed_input_addrs[0].logic_addr, 0);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].support_refresh, false);
  EXPECT_NE(task_run_param.parsed_input_addrs[1].logic_addr, 0);
  EXPECT_EQ(task_run_param.parsed_input_addrs[1].support_refresh, false);
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
}
}  // end of namespace ge
