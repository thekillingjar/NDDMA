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
#include <memory>

#include "common/ge_inner_error_codes.h"
#include "common/util.h"
#include "runtime/mem.h"
#include "omg/omg_inner_types.h"

#include "macro_utils/dt_public_scope.h"
#include "executor/ge_executor.h"
#include "common/helper/file_saver.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/davinci_model.h"
#include "hybrid/hybrid_davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/execute/graph_executor.h"

using namespace std;

namespace ge {
class UtestGraphLoader : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
  }
};

TEST_F(UtestGraphLoader, LoadDataFromFile) {
  GeExecutor ge_executor;
  ge_executor.is_inited_ = true;

  string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);

  string path = test_smap;
  int32_t priority = 0;
  ModelData model_data;
  auto ret = GraphLoader::LoadDataFromFile(path, priority, model_data);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_NE(model_data.model_data, nullptr);
  delete[] static_cast<char *>(model_data.model_data);
  model_data.model_data = nullptr;
  ge_executor.is_inited_ = false;
}

TEST_F(UtestGraphLoader, CommandHandle) {
  Command command;
  EXPECT_EQ(ModelManager::GetInstance().HandleCommand(command), PARAM_INVALID);
}

TEST_F(UtestGraphLoader, LoadModelFromData) {
  uint32_t model_id = 0;
  ModelData model_data;
  const ModelParam model_param{0, 0U, 1024U, 0U, 0U};
  auto ret = GraphLoader::LoadModelFromData(model_data, model_param, model_id);
  EXPECT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestGraphLoader, LoadModelWithQ) {
  uint32_t model_id = 0;
  ModelData model_data;
  const std::vector<uint32_t> input_queue_ids;
  const std::vector<uint32_t> output_queue_ids;
  auto ret = GraphLoader::LoadModelWithQ(model_id, model_data, {input_queue_ids, output_queue_ids, {}});
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(UtestGraphLoader, LoadModelWithQueueParam) {
  uint32_t model_id = 0;
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  const GeRootModelPtr root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);
  const std::vector<uint32_t> input_queue_ids;
  const std::vector<uint32_t> output_queue_ids;
  const std::vector<QueueAttrs> input_queue_attrs;
  const std::vector<QueueAttrs> output_queues_attrs;
  ModelQueueParam model_queue_param;
  model_queue_param.input_queues = input_queue_ids;
  model_queue_param.output_queues = output_queue_ids;
  model_queue_param.input_queues_attrs = input_queue_attrs;
  model_queue_param.output_queues_attrs = output_queues_attrs;
  auto ret = GraphLoader::LoadModelWithQueueParam(model_id, root_model, model_queue_param, false);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestGraphLoader, ExecuteModel) {
  uint32_t model_id = 0;
  rtStream_t stream = nullptr;
  bool async_mode = false;
  InputData input_data;
  std::vector<GeTensorDesc> input_desc;
  OutputData output_data;
  std::vector<GeTensorDesc> output_desc;
  auto ret = GraphLoader::ExecuteModel(model_id, stream, async_mode, input_data, input_desc,
                                       output_data, output_desc);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestGraphLoader, DestroyAicpuKernel) {
  uint64_t session_id = 0U;
  uint32_t model_id = 0U;
  uint32_t sub_model_id = 0U;
  auto ret = ModelManager::GetInstance().DestroyAicpuKernel(session_id, model_id, sub_model_id);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphLoader, DestroyAicpuSessionForInfer) {
  uint32_t model_id = 0;
  auto ret = ModelManager::GetInstance().DestroyAicpuSessionForInfer(model_id);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestGraphLoader, LoadModelOnline_Invalid) {
  uint32_t model_id = 0U;
  GeRootModelPtr ge_root_model = nullptr;
  GraphNodePtr graph_node;
  uint32_t device_id = 0U;
  error_message::ErrorManagerContext error_context;

  auto ret = GraphLoader::LoadModelOnline(model_id, ge_root_model, graph_node, device_id, error_context);
  EXPECT_EQ(ret, GE_GRAPH_PARAM_NULLPTR);

  device_id = UINT32_MAX;
  ge_root_model = make_shared<ge::GeRootModel>();
  ret = GraphLoader::LoadModelOnline(model_id, ge_root_model, graph_node, device_id, error_context);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphLoader, GetMaxUsedMemory_Invalid) {
  uint32_t model_id = 1;
  uint64_t max_size;

  auto ret = ModelManager::GetInstance().GetMaxUsedMemory(model_id, max_size);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphLoader, LoadDataFromFile_Invalid) {
  int32_t priority = 0;
  ModelData model_data;
  string path = "/abc/model_file";

  auto ret = GraphLoader::LoadDataFromFile(path, priority, model_data);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
}
}