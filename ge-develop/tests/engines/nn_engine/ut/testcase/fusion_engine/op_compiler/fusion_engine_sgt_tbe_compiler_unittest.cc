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
#include <nlohmann/json.hpp>
#include "all_ops_stub.h"

#define protected public
#define private public

#include "adapter/adapter_itf/task_builder_adapter.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/plugin_manager.h"
#include "graph/utils/attr_utils.h"
#include "common/util/op_info_util.h"
#include "ops_store/op_kernel_info.h"
#include "common/fe_op_info_common.h"
#include "common/fe_log.h"
#include "common/sgt_slice_type.h"
#include "common/aicore_util_types.h"
#include "common/scope_allocator.h"
#include "ops_store/ops_kernel_manager.h"
#include "platform/platform_info.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;
using namespace te;


using TbeInfoAssemblerPtr = std::shared_ptr<TbeInfoAssembler>;
te::OpBuildResCode TeFusionStubOnlySingleNode(std::vector<Node*> teGraphNode,
                                              OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>&to_be_del,
                                              uint64_t taskid, uint64_t tid, uint64_t slice_id,
                                              const std::string op_compile_strategy)
{
  string json_file_path = "./kernel_meta/";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_SUCC;
}


te::OpBuildResCode TeFusionStubNew(std::vector<Node*> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr> &to_be_del,
                                   uint64_t taskid, uint64_t tid, const std::string op_compile_strategy)
{
  string json_file_path = "";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_SUCC;
}
// The first time ub fused compilation is failed and the
// second time single-node compilation is passed.
bool WaitAllFinishedStub2(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "path_failed1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "path_failed2");
    fin_task.push_back(fin_com_task);
    count++;
  } else {
    te::FinComTask fin_com_task1;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task1.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task1.taskId = curr_atomic_id - 1;
    fin_com_task1.status = 0;
    ge::AttrUtils::SetStr(fin_com_task1.teNodeOpDesc, "json_file_path", "path_succ1");
    fin_task.push_back(fin_com_task1);

    te::FinComTask fin_com_task2;
    fin_com_task2.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task2.taskId = curr_atomic_id - 2;
    fin_com_task2.status = 0;
    ge::AttrUtils::SetStr(fin_com_task2.teNodeOpDesc, "json_file_path", "path_succ2");
    fin_task.push_back(fin_com_task2);

    te::FinComTask fin_com_task3;
    fin_com_task3.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task3.taskId = curr_atomic_id - 3;
    fin_com_task3.status = 0;
    ge::AttrUtils::SetStr(fin_com_task3.teNodeOpDesc, "json_file_path", "path_succ3");
    fin_task.push_back(fin_com_task3);

    te::FinComTask fin_com_task4;
    fin_com_task4.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task4.taskId = curr_atomic_id - 4;
    fin_com_task4.status = 0;
    ge::AttrUtils::SetStr(fin_com_task4.teNodeOpDesc, "json_file_path", "path_succ4");
    fin_task.push_back(fin_com_task4);
    count++;
  }

  return true;
}

// The first time two single node compilation: one successful and one failed.
// second time single-node compilation is successfully.
bool WaitAllFinishedStub3(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 3;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 4;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task);
    count++;
  } else {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu2");
    fin_task.push_back(fin_com_task);
  }

  return true;
}

// The first time two single node compilation: both successful
bool WaitAllFinishedStubBothSucc(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  te::FinComTask fin_com_task;
  uint64_t curr_atomic_id = GetAtomicId();

  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
  fin_com_task.taskId = curr_atomic_id - 1;
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu1");
  fin_task.push_back(fin_com_task);

  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
  fin_com_task.taskId = curr_atomic_id - 2;
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu2");
  fin_task.push_back(fin_com_task);

  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
  fin_com_task.taskId = curr_atomic_id - 3;
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn1");
  fin_task.push_back(fin_com_task);

  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
  fin_com_task.taskId = curr_atomic_id - 4;
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
  fin_task.push_back(fin_com_task);
  return true;
}

// The first time ub fused compilation is failed and the
// second time single-node compilation still faield.
bool WaitAllFinishedSecondTimeStillFailed(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);
    count++;
  } else {
    te::FinComTask fin_com_task1;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task1.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task1.taskId = curr_atomic_id - 1;
    fin_com_task1.status = 0;
    ge::AttrUtils::SetStr(fin_com_task1.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task1);

    te::FinComTask fin_com_task2;
    fin_com_task2.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task2.taskId = curr_atomic_id - 2;
    fin_com_task2.status = 1;
    ge::AttrUtils::SetStr(fin_com_task2.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task2);

    te::FinComTask fin_com_task3;
    fin_com_task3.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task3.taskId = curr_atomic_id - 3;
    fin_com_task3.status = 1;
    ge::AttrUtils::SetStr(fin_com_task3.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task3);

    te::FinComTask fin_com_task4;
    fin_com_task4.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task4.taskId = curr_atomic_id - 4;
    fin_com_task4.status = 1;
    ge::AttrUtils::SetStr(fin_com_task4.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task4);
    count++;
  }

  return true;
}

// One task is failed and another is successful
// Re-compile them as single node.
// Second time compilation is successful
bool WaitAllFinishedOneTaskFailedAnotherSucc(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task);
    count++;
  } else {
    te::FinComTask fin_com_task1;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task1.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task1.taskId = curr_atomic_id - 1;
    fin_com_task1.status = 0;
    ge::AttrUtils::SetStr(fin_com_task1.teNodeOpDesc, "json_file_path", "relu1");
    fin_task.push_back(fin_com_task1);

    te::FinComTask fin_com_task2;
    fin_com_task2.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task2.taskId = curr_atomic_id - 2;
    fin_com_task2.status = 0;
    ge::AttrUtils::SetStr(fin_com_task2.teNodeOpDesc, "json_file_path", "relu2");
    fin_task.push_back(fin_com_task2);

    te::FinComTask fin_com_task3;
    fin_com_task3.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task3.taskId = curr_atomic_id - 3;
    fin_com_task3.status = 0;
    ge::AttrUtils::SetStr(fin_com_task3.teNodeOpDesc, "json_file_path", "bn1");
    fin_task.push_back(fin_com_task3);

    te::FinComTask fin_com_task4;
    fin_com_task4.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task4.taskId = curr_atomic_id - 4;
    fin_com_task4.status = 0;
    ge::AttrUtils::SetStr(fin_com_task4.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task4);
    count++;
  }

  return true;
}

// One task is failed and another is successful for scope 0. Single node compiling is
// successful for scope 1
// Re-compile them as single node.
// Second time compilation is successful
bool WaitAllFinishedThreeNode(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "conv1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "conv2");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 3;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 4;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task);

    count++;
  } else {
    te::FinComTask fin_com_task1;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task1.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task1.taskId = curr_atomic_id - 1;
    fin_com_task1.status = 0;
    ge::AttrUtils::SetStr(fin_com_task1.teNodeOpDesc, "json_file_path", "relu1");
    fin_task.push_back(fin_com_task1);

    te::FinComTask fin_com_task2;
    fin_com_task2.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task2.taskId = curr_atomic_id - 2;
    fin_com_task2.status = 0;
    ge::AttrUtils::SetStr(fin_com_task2.teNodeOpDesc, "json_file_path", "relu2");
    fin_task.push_back(fin_com_task2);

    te::FinComTask fin_com_task3;
    fin_com_task3.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task3.taskId = curr_atomic_id - 3;
    fin_com_task3.status = 0;
    ge::AttrUtils::SetStr(fin_com_task3.teNodeOpDesc, "json_file_path", "bn1");
    fin_task.push_back(fin_com_task3);

    te::FinComTask fin_com_task4;
    fin_com_task4.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task4.taskId = curr_atomic_id - 4;
    fin_com_task4.status = 0;
    ge::AttrUtils::SetStr(fin_com_task4.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task4);
    count++;
  }

  return true;
}

// One task is failed and another is successful for scope 0. Single node conv
// compiling is failed for scope 1.
// Re-compile them as single node.
// Second time the single node conv still fails.
bool WaitAllFinishedThreeNodeAllFailed(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();


    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 3;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 4;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task);

    count++;
  } else {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();


    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "conv1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "failed");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 3;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 4;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu2");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 5;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 6;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task);
    count++;
  }

  return true;
}

/* Compile three node.
 * First time one of the fused tasks(bn+relu) fails and we re-compile them.
 * The single node also fails to compile.
 * Second time they are all successfully compiled.
 * The single node is not a sgt sliced node. */
bool WaitAllFinishedThreeNode2(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  static int count = 0;
  if (count == 0) {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv_not_sliced", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "fail");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 1;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "path_failed1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 3;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "path_failed2");
    fin_task.push_back(fin_com_task);


    count++;
  } else {
    te::FinComTask fin_com_task;
    uint64_t curr_atomic_id = GetAtomicId();

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("conv_not_sliced", "Conv2D");
    fin_com_task.taskId = curr_atomic_id - 1;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "conv_not_sliced");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 2;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("relu", "Activation");
    fin_com_task.taskId = curr_atomic_id - 3;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "relu2");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 4;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn1");
    fin_task.push_back(fin_com_task);

    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("bn", "BatchNorm");
    fin_com_task.taskId = curr_atomic_id - 5;
    fin_com_task.status = 0;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "bn2");
    fin_task.push_back(fin_com_task);


    count++;
  }

  return true;
}

te::LX_QUERY_STATUS GetTbeOpinfoStubSucc(const te::TbeOpInfo &info, std::string &op_info) {
  return te::LX_QUERY_SUCC;
}

namespace {
CubeVecStateNew current_cv_state = CubeVecStateNew::CUBE_VEC_FUSE;
AICoreMode current_ffts_mode = FFTS_MODE_FFTS_PLUS;
}

class UTEST_FE_SGT_TBE_COMPILER : public testing::Test {
 protected:

  static void SetUpTestCase() {
    current_cv_state = PlatformUtils::Instance().GetCubeVecState();
    current_ffts_mode = PlatformUtils::Instance().GetFftsMode();
    PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeVecState)] = static_cast<int64_t>(CubeVecStateNew::CUBE_VEC_SPLIT);
    PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::FftsMode)] = static_cast<int64_t>(FFTS_MODE_FFTS_PLUS);
  }

  static void TearDownTestCase() {
    PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::CubeVecState)] = static_cast<int64_t>(current_cv_state);
    PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::FftsMode)] = static_cast<int64_t>(current_ffts_mode);
  }

  void SetSliceinfo(const vector<int64_t> &shape, size_t tensor_num, size_t slice_num,
                    vector<vector<vector<ffts::DimRange>>> &tensorSlice) {
    for (size_t i = 0; i < slice_num; i++) {
      vector<vector<ffts::DimRange>> dim_range_vec_each_tensor;
      for (size_t j = 0; j < tensor_num; j++) {
        vector<ffts::DimRange> dim_range_vec;
        for (size_t dim_idx = 0; dim_idx < shape.size(); dim_idx++) {
          if (dim_idx == 0) {
            // only the highest dimension needs to be sliced.
            int64_t dim_before_slicing = shape.at(dim_idx);
            int64_t dim_non_tail = dim_before_slicing / slice_num;
            int64_t dim_tail = dim_before_slicing / slice_num + dim_before_slicing % slice_num;
            int64_t low, high;
            if (i == slice_num - 1) {
              low = i * dim_non_tail;
              high = dim_before_slicing;
            } else {
              low = i * dim_non_tail;
              high = (i + 1) * dim_non_tail;
            }
            ffts::DimRange dr = {low, high};
            dim_range_vec.emplace_back(dr);
          } else {
            ffts::DimRange dr = {0, shape.at(dim_idx)};
            dim_range_vec.emplace_back(dr);
          }
        }
        dim_range_vec_each_tensor.emplace_back(dim_range_vec);
      }
      tensorSlice.emplace_back(dim_range_vec_each_tensor);
    }
  }

  void SetOneTensorDesc(ge::GeTensorDescPtr &tensor, size_t tensor_num, size_t slice_num,
                        vector<vector<vector<ffts::DimRange>>> &tensorSlice,
                        vector<vector<vector<ffts::DimRange>>> &oriTensorSlice) {
    auto shape = tensor->GetShape().GetDims();
    auto ori_shape = tensor->GetOriginShape().GetDims();

    SetSliceinfo(shape, tensor_num, slice_num, tensorSlice);
    SetSliceinfo(ori_shape, tensor_num, slice_num, oriTensorSlice);
  }

  void SetSgtSliceInfo(ge::ComputeGraphPtr &graph, int32_t slice_num) {
    for (auto &node : graph->GetDirectNode()) {
      auto op_desc_ptr = node->GetOpDesc();

      if (op_desc_ptr->GetName() == "relu" ||
          op_desc_ptr->GetName() == "bn" ||
          op_desc_ptr->GetName() == "conv") {
        AttrUtils::SetInt(node->GetOpDesc(), kThreadScopeId, 1);
        ffts::ThreadSliceMap subgraphInfo;
        //thread->tensor->dim->range
        vector<vector<vector<ffts::DimRange>>> inputTensorSlice;
        vector<vector<vector<ffts::DimRange>>> oriInputTensorSlice;
        auto input0 = op_desc_ptr->MutableInputDesc(0);
        SetOneTensorDesc(input0, 1, slice_num, inputTensorSlice, oriInputTensorSlice);

        vector<vector<vector<ffts::DimRange>>> outputTensorSlice;
        vector<vector<vector<ffts::DimRange>>> oriOutputTensorSlice;
        auto output0 = op_desc_ptr->MutableOutputDesc(0);
        SetOneTensorDesc(output0, 1, slice_num, outputTensorSlice, oriOutputTensorSlice);

        subgraphInfo.input_tensor_slice = inputTensorSlice;
        subgraphInfo.ori_input_tensor_slice = oriInputTensorSlice;
        subgraphInfo.output_tensor_slice = outputTensorSlice;
        subgraphInfo.ori_output_tensor_slice = oriOutputTensorSlice;
        subgraphInfo.slice_instance_num = slice_num;
        subgraphInfo.thread_mode = 1;
        ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(subgraphInfo);
        node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
      }
    }
  }
  /*
   * batchnorm
   *    |
   *   relu
   */
  static void CreateGraphSgtSlice(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("bn", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dims = {288, 32, 48, 65};
    GeShape shape(dims);
    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    in_desc1.SetOriginShape(shape);
    bn_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    out_desc1.SetOriginShape(shape);
    bn_op->AddOutputDesc("y", out_desc1);

    vector<int64_t> dims2 = {65, 2, 3, 4};
    GeShape shape2(dims2);
    GeTensorDesc in_desc2(shape2);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    in_desc2.SetOriginShape(shape2);
    relu_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape2);
    out_desc2.SetFormat(FORMAT_NCHW);
    out_desc2.SetDataType(DT_FLOAT16);
    out_desc2.SetOriginShape(shape2);
    relu_op->AddOutputDesc("y", out_desc2);

    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(bn_op, kThreadMode, 1);
    ge::AttrUtils::SetInt(relu_op, kThreadMode, 1);
    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    ge::AttrUtils::SetStr(bn_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }

  static void CreateGraphConv(ComputeGraphPtr graph, string name = "conv") {
    OpDescPtr conv_op = std::make_shared<OpDesc>(name, "Conv2D");

    // add descriptor
    vector<int64_t> dims = {288, 32, 48, 65};
    GeShape shape(dims);
    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_NCHW);
    in_desc1.SetDataType(DT_FLOAT16);
    in_desc1.SetOriginShape(shape);
    conv_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_NCHW);
    out_desc1.SetDataType(DT_FLOAT16);
    out_desc1.SetOriginShape(shape);
    conv_op->AddOutputDesc("y", out_desc1);

    ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(conv_op);
    ge::AttrUtils::SetStr(bn_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  }
};

/* All tasks in one group are failed. Need to re-compile them all.
 * First time and second time are both single op. */
void StubPlatFormInfo() {
  std::string soc_version = "Ascend910B2";
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = soc_version;
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion(soc_version);
  PlatformUtils::Instance().soc_version_ = soc_version;
}

void CheckOneTensor(const vector<vector<ffts::DimRange>> &slice,
                    const vector<ffts::DimRange>& expect_range,
                    const vector<int64_t>& expect,
                    const ge::GeTensorDescPtr &tensor,
                    int32_t index_in_attr) {
  EXPECT_EQ(slice.size(), 1);

  auto slice_tensor_0 = slice[0];
  EXPECT_EQ(slice_tensor_0.size(), 4);
  EXPECT_EQ(slice_tensor_0[0], expect_range[0]);
  EXPECT_EQ(slice_tensor_0[1], expect_range[1]);
  EXPECT_EQ(slice_tensor_0[2], expect_range[2]);
  EXPECT_EQ(slice_tensor_0[3], expect_range[3]);

  vector<vector<int64_t>> slice_dims_head_tail;
  (void)ge::AttrUtils::GetListListInt(tensor, ATTR_NAME_SGT_SLICE_SHAPE, slice_dims_head_tail);
  ASSERT_EQ(slice_dims_head_tail.size(), 2);
  EXPECT_EQ(slice_dims_head_tail[index_in_attr][0], expect[0]);
  EXPECT_EQ(slice_dims_head_tail[index_in_attr][1], expect[1]);
  EXPECT_EQ(slice_dims_head_tail[index_in_attr][2], expect[2]);
  EXPECT_EQ(slice_dims_head_tail[index_in_attr][3], expect[3]);

}

void CheckSliceInfo(const ge::NodePtr &node, const vector<int64_t> &non_tail,
                    const vector<int64_t> &tail, vector<ffts::DimRange> &first_slice_range,
                    vector<ffts::DimRange> &last_slice_range, uint32_t slice_num) {
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);

  EXPECT_EQ(slice_info_ptr->slice_instance_num, slice_num);
  auto &input_slices = slice_info_ptr->input_tensor_slice;
  EXPECT_EQ(input_slices.size(), slice_num);

  // non-tail slice of input
  auto input_first_slice = input_slices.at(0);

  FE_LOGD("Check non-tail slice for node %s.", node->GetName().c_str());
  CheckOneTensor(input_first_slice, first_slice_range, non_tail,
                 node->GetOpDesc()->MutableInputDesc(0), 0);

  // tail slice of input
  auto input_tail_slice = input_slices.at(slice_num - 1);
  CheckOneTensor(input_tail_slice, last_slice_range, tail,
                 node->GetOpDesc()->MutableInputDesc(0), 1);


  auto &output_slices = slice_info_ptr->output_tensor_slice;
  EXPECT_EQ(output_slices.size(), slice_num);

  // non-tail slice of output
  auto output_first_slice = output_slices.at(0);
  CheckOneTensor(output_first_slice, first_slice_range, non_tail,
                 node->GetOpDesc()->MutableOutputDesc(0), 0);


  // tail slice of output
  auto output_tail_slice = output_slices.at(slice_num - 1);
  CheckOneTensor(output_tail_slice, last_slice_range, tail,
                 node->GetOpDesc()->MutableOutputDesc(0), 1);
}

int64_t GetN(bool is_tail, uint32_t slice_num, int64_t init_dim) {
  if (is_tail) {
    return init_dim / slice_num + init_dim % slice_num;
  } else {
    return init_dim / slice_num;
  }
}
/* Failed to compile two nodes(relu and bn) as a fused node.
 * Retry single node compilation.
 * Two single nodes are both successfully compiled. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_first_time_failed_and_rolled_back_to_single_op)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub2;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 7;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr;
  for (auto node: graph->GetDirectNode()) {
    vector_node_ptr.emplace_back(node.get());
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  EXPECT_EQ(compile_info.compile_ret_map.size(), 2);
  uint32_t index = 1;
  for (auto ele : compile_info.compile_ret_map) {
    string path_name1 = "path_succ" + std::to_string(index * 2);
    string path_name2 = "path_succ" + std::to_string(index * 2 - 1);
    EXPECT_EQ(ele.second.size(), 2);
    EXPECT_EQ(ele.second.at(0).json_file_path, path_name1);
    EXPECT_EQ(ele.second.at(1).json_file_path, path_name2);
    index++;
  }

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Failed to compile relu as a single node and successfully compile bn.
 * Retry single node relu compilation.
 * Second time compiliation for relu is successful. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_one_node_failed_another_succ)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub3;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 8;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  std::vector<ge::Node*> vector_node_ptr1;
  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      vector_node_ptr0.emplace_back(node.get());
    }
    if (node->GetName() == "relu") {
      vector_node_ptr1.emplace_back(node.get());
    }
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr1));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  ScopeAllocator::Instance().neg_scope_id_ = 0;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  EXPECT_EQ(compile_info.compile_ret_map.size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[0][0].json_file_path, "bn2");
  EXPECT_EQ(compile_info.compile_ret_map[-1][0].json_file_path, "relu2");
  EXPECT_EQ(compile_info.compile_ret_map[0][1].json_file_path, "bn1");
  EXPECT_EQ(compile_info.compile_ret_map[-1][1].json_file_path, "relu1");

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Compile two node as single node, both of them are successfully compiled. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_both_succ)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStubBothSucc;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 9;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  std::vector<ge::Node*> vector_node_ptr1;
  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      vector_node_ptr0.emplace_back(node.get());
    }
    if (node->GetName() == "relu") {
      vector_node_ptr1.emplace_back(node.get());
    }
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr1));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  EXPECT_EQ(compile_info.compile_ret_map.size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[0].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[1].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[0][0].json_file_path, "bn2");
  EXPECT_EQ(compile_info.compile_ret_map[1][0].json_file_path, "relu2");
  EXPECT_EQ(compile_info.compile_ret_map[0][1].json_file_path, "bn1");
  EXPECT_EQ(compile_info.compile_ret_map[1][1].json_file_path, "relu1");

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Compile two node as a fused node.
 * First time compilation is failed and we re-compile them as single node.
 * One task of relu still fails. Return FAILED. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_second_time_still_failed)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedSecondTimeStillFailed;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 10;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  for (auto node: graph->GetDirectNode()) {
    vector_node_ptr0.emplace_back(node.get());
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  EXPECT_EQ(compile_info.compile_ret_map.size(), 0);

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }

  EXPECT_EQ(fe::FAILED, ret);
}

/* Compile two node as sa fused node, one task failed.
 * Re-Compile them as single node, both successful. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_one_task_succ_another_failed)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedOneTaskFailedAnotherSucc;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 10;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  for (auto node: graph->GetDirectNode()) {
    vector_node_ptr0.emplace_back(node.get());
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  ScopeAllocator::Instance().neg_scope_id_ = 0;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  EXPECT_EQ(compile_info.compile_ret_map.size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[-1].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[-2].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[-1][0].json_file_path, "bn2");
  EXPECT_EQ(compile_info.compile_ret_map[-2][0].json_file_path, "relu2");
  EXPECT_EQ(compile_info.compile_ret_map[-1][1].json_file_path, "bn1");
  EXPECT_EQ(compile_info.compile_ret_map[-2][1].json_file_path, "relu1");

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Compile three node.
 * First time one of the fused tasks(bn+relu) fails and we re-compile them.
 * The single node compiles successfully.
 * Second time they(bn+relu) both succeed. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_one_task_succ_another_failed_three_node)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedThreeNode;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 10;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  CreateGraphConv(graph);

  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  std::vector<ge::Node*> vector_node_ptr1;
  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "relu" || node->GetName() == "bn") {
      vector_node_ptr0.emplace_back(node.get());
    } else {
      vector_node_ptr1.emplace_back(node.get());
    }
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr1));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  ScopeAllocator::Instance().neg_scope_id_ = 0;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  ASSERT_EQ(compile_info.compile_ret_map.size(), 3);
  ASSERT_EQ(compile_info.compile_ret_map[1].size(), 2);
  ASSERT_EQ(compile_info.compile_ret_map[-1].size(), 2);
  ASSERT_EQ(compile_info.compile_ret_map[-2].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[1][0].json_file_path, "conv2");
  EXPECT_EQ(compile_info.compile_ret_map[-1][0].json_file_path, "bn2");
  EXPECT_EQ(compile_info.compile_ret_map[-2][0].json_file_path, "relu2");
  EXPECT_EQ(compile_info.compile_ret_map[1][1].json_file_path, "conv1");
  EXPECT_EQ(compile_info.compile_ret_map[-1][1].json_file_path, "bn1");
  EXPECT_EQ(compile_info.compile_ret_map[-2][1].json_file_path, "relu1");

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn" || node->GetName() == "conv") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Compile three node.
 * First time one of the fused tasks(bn+relu) fails and we re-compile them.
 * The single node also fails to compile.
 * Second time they(bn+relu) both succeed but the single
 * node(conv) still fails. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_one_task_succ_another_failed_three_node_2)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedThreeNodeAllFailed;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 10;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  CreateGraphConv(graph);

  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  std::vector<ge::Node*> vector_node_ptr1;
  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "relu" || node->GetName() == "bn") {
      vector_node_ptr0.emplace_back(node.get());
    } else {
      vector_node_ptr1.emplace_back(node.get());
    }
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr1));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);
  EXPECT_EQ(compile_info.compile_ret_map.size(), 0);

  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "bn" || node->GetName() == "conv") {
      int64_t non_tail_n = GetN(false, slice_num, 288);
      int64_t tail_n = GetN(true, slice_num, 288);
      vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
      vector<int64_t> tail = {tail_n, 32, 48, 65};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
      vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }

    if (node->GetName() == "relu") {
      int64_t non_tail_n = GetN(false, slice_num, 65);
      int64_t tail_n = GetN(true, slice_num, 65);
      vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
      vector<int64_t> tail = {tail_n, 2, 3, 4};
      vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
      vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
      CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
    }
  }
  EXPECT_EQ(fe::FAILED, ret);
}

/* Compile three node.
 * First time one of the fused tasks(bn+relu) fails and we re-compile them.
 * The single node also fails to compile.
 * Second time they are all successfully compiled.
 * The single node is not a sgt sliced node. */
TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_one_task_succ_another_failed_three_node_3)
{
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.TeFusionV = TeFusionStubOnlySingleNode;
  compile_tbe_op.TeFusion = TeFusionStubNew;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedThreeNode2;
  compile_tbe_op.GetOpInfo = GetTbeOpinfoStubSucc;
  uint32_t slice_num = 10;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphSgtSlice(graph);
  CreateGraphConv(graph, "conv_not_sliced");

  SetSgtSliceInfo(graph, slice_num);
  std::vector<ge::Node*> vector_node_ptr0;
  std::vector<ge::Node*> vector_node_ptr1;
  for (auto node: graph->GetDirectNode()) {
    if (node->GetName() == "relu" || node->GetName() == "bn") {
      vector_node_ptr0.emplace_back(node.get());
    } else {
      vector_node_ptr1.emplace_back(node.get());
    }
  }

  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr0));
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr1));

  CompileResultMap compile_ret_map;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  StubPlatFormInfo();
  ScopeAllocator::Instance().neg_scope_id_ = 0;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  EXPECT_EQ(compile_info.compile_ret_map.size(), 3);

  ASSERT_EQ(compile_info.compile_ret_map[-1].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[-1][0].json_file_path, "bn2");
  EXPECT_EQ(compile_info.compile_ret_map[-1][1].json_file_path, "bn1");
  ASSERT_EQ(compile_info.compile_ret_map[-2].size(), 2);
  EXPECT_EQ(compile_info.compile_ret_map[-2][0].json_file_path, "relu2");
  EXPECT_EQ(compile_info.compile_ret_map[-2][1].json_file_path, "relu1");
  EXPECT_EQ(compile_info.compile_ret_map[-3][0].json_file_path, "conv_not_sliced");
  for (auto node: graph->GetDirectNode()) {
    for (auto node: graph->GetDirectNode()) {
      if (node->GetName() == "bn") {
        int64_t non_tail_n = GetN(false, slice_num, 288);
        int64_t tail_n = GetN(true, slice_num, 288);
        vector<int64_t> non_tail = {non_tail_n, 32, 48, 65};
        vector<int64_t> tail = {tail_n, 32, 48, 65};
        vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 32}, {0, 48}, {0, 65}};
        vector<ffts::DimRange> last_slice_range = {{288 - tail_n, 288}, {0, 32}, {0, 48}, {0, 65}};
        CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
      }

      if (node->GetName() == "relu") {
        int64_t non_tail_n = GetN(false, slice_num, 65);
        int64_t tail_n = GetN(true, slice_num, 65);
        vector<int64_t> non_tail = {non_tail_n, 2, 3, 4};
        vector<int64_t> tail = {tail_n, 2, 3, 4};
        vector<ffts::DimRange> first_slice_range = {{0, non_tail_n}, {0, 2}, {0, 3}, {0, 4}};
        vector<ffts::DimRange> last_slice_range = {{65 - tail_n, 65}, {0, 2}, {0, 3}, {0, 4}};
        CheckSliceInfo(node, non_tail, tail, first_slice_range, last_slice_range, slice_num);
      }
    }

    if (node->GetName() == "conv_not_sliced") {
      ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
      slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
      EXPECT_EQ(slice_info_ptr, nullptr);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_FE_SGT_TBE_COMPILER, case_compile_op_multi_slice) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());
  ge::AttrUtils::SetInt(Node1->GetOpDesc(), kThreadMode, 1);

  ScopeNodeIdMap fusion_nodes;
  fusion_nodes.insert(std::make_pair(0, vector_node_ptr));
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node2);
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes;

  auto origin_soc_version = PlatformUtils::Instance().GetSocVersion();
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";

  Status ret = compile_tbe_op.CompileOp(compile_info);
  EXPECT_EQ(fe::FAILED, ret);
  PlatformUtils::Instance().soc_version_ = origin_soc_version;
}
