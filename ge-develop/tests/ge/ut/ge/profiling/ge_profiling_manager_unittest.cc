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
#include <securec.h>
#include "depends/runtime/src/runtime_stub.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "common/global_variables/diagnose_switch.h"
#include "macro_utils/dt_public_scope.h"

#undef  PROF_MODEL_EXECUTE_MASK
#undef  PROF_OP_DETAIL_MASK
#undef  PROF_MODEL_LOAD_MASK

constexpr uint64_t PROF_TRAINING_TRACE = 0x00000040ULL;
constexpr uint64_t PROF_MODEL_EXECUTE_MASK        = 0x0000001000000ULL;
constexpr uint64_t PROF_OP_DETAIL_MASK            = 0x0000080000000ULL;
constexpr uint64_t PROF_MODEL_LOAD_MASK           = 0x8000000000000000ULL;

#include "common/profiling/profiling_init.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_properties.h"
#include "common/profiling_definitions.h"
#include "common/profiling/command_handle.h"
#include "graph/ge_local_context.h"
#include "framework/common/profiling/ge_profiling.h"
#include "graph/manager/graph_manager.h"
#include "graph/ops_stub.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "depends/profiler/src/profiling_test_util.h"

using namespace ge;
using namespace std;

namespace {
enum ProfCommandHandleType {
    kProfCommandhandleInit = 0,
    kProfCommandhandleStart,
    kProfCommandhandleStop,
    kProfCommandhandleFinalize,
    kProfCommandhandleModelSubscribe,
    kProfCommandhandleModelUnsubscribe
  };
  constexpr int32_t TMP_ERROR = -1;       // RT_ERROR


}

class UtestGeProfilingManager : public testing::Test {
 protected:
  void SetUp() {
    const auto davinci_model = MakeShared<DavinciModel>(2005U, MakeShared<RunAsyncListener>());
    ModelManager::GetInstance().InsertModel(2005U, davinci_model);
  }

  void TearDown() override {
    EXPECT_EQ(ModelManager::GetInstance().DeleteModel(2005U), SUCCESS);
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
  }
};

int32_t ReporterCallback(uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
  return -1;
}

void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}


TEST_F(UtestGeProfilingManager, Prof_finalize_) {
  Status ret = ProfilingManager::Instance().ProfFinalize();
  EXPECT_EQ(ret, ge::SUCCESS);
}


TEST_F(UtestGeProfilingManager, get_fp_bp_point_) {
  map<std::string, string> options_map = {
    {OPTION_EXEC_PROFILING_OPTIONS,
    R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})"}};
  GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);

  std::string fp_point;
  std::string bp_point;
  ProfilingProperties::Instance().GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "Data_0");
  EXPECT_EQ(bp_point, "addn");
}

TEST_F(UtestGeProfilingManager, get_fp_bp_point_empty) {
  // fp bp empty
  map<std::string, string> options_map = {
    { OPTION_EXEC_PROFILING_OPTIONS,
      R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","ai_core_metrics":"ResourceConflictRatio"})"}};
  GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);
  std::string fp_point = "fp";
  std::string bp_point = "bp";
  ProfilingProperties::Instance().bp_point_ = "";
  ProfilingProperties::Instance().fp_point_ = "";
  ProfilingProperties::Instance().GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "");
  EXPECT_EQ(bp_point, "");
}

TEST_F(UtestGeProfilingManager, handle_subscribe_info) {
  uint32_t prof_type = RT_PROF_CTRL_SWITCH;
  MsprofCommandHandle prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  prof_data.type = 0;
  domi::GetContext().train_flag = true;
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(false);
  auto prof_ptr = std::make_shared<MsprofCommandHandle>(prof_data);
  Status ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilingManager, need_handle_start_end) {
  Status ret;
  uint32_t prof_type = RT_PROF_CTRL_SWITCH;

  MsprofCommandHandle prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  domi::GetContext().train_flag = true;
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(false);
  auto prof_ptr = std::make_shared<MsprofCommandHandle>(prof_data);

  prof_ptr->type = kProfCommandhandleModelUnsubscribe + 2;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, TMP_ERROR);

  prof_ptr->type = kProfCommandhandleStart;
  prof_ptr->devNums = 0;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, TMP_ERROR);

  prof_ptr->devNums = 2;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, TMP_ERROR);

  prof_ptr->devNums = 1;
  prof_ptr->devIdList[0] = 0;
  prof_ptr->devIdList[1] = 1;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, RT_ERROR_NONE);
}

TEST_F(UtestGeProfilingManager, handle_ctrl_switch) {
  Status ret;
  uint32_t prof_type = RT_PROF_CTRL_SWITCH;

  std::map<std::string, std::string> options_map;
  options_map[OPTION_GRAPH_RUN_MODE] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options_map);
  MsprofCommandHandle prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  domi::GetContext().train_flag = true;
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(false);
  auto prof_ptr = std::make_shared<MsprofCommandHandle>(prof_data);

  prof_ptr->devNums = 1;
  prof_ptr->devIdList[0] = 0;
  prof_ptr->devIdList[1] = 1;

  prof_ptr->type = 4;   // kProfCommandHandleModelSubscribe;
  prof_ptr->profSwitch = -1;
  prof_ptr->modelId = -1;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, SUCCESS);

  prof_ptr->type = 5;
  prof_ptr->profSwitch = 0;
  prof_ptr->modelId = 2005U;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, FAILED);
  ProfilingProperties::Instance().SetSubscribeInfo(0, 2005U, false);
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, SUCCESS);

  domi::GetContext().train_flag = false;
  std::map<std::string, std::string> options_tmp;
  ge::GetThreadLocalContext().SetGraphOption(options_tmp);
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(true);
  prof_ptr->type = 4;
  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, SUCCESS);

  ret = ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeProfilingManager, handle_subscribe_info2) {
  uint32_t prof_type = RT_PROF_CTRL_REPORTER;
  MsprofCommandHandle prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  prof_data.type = 0;
  domi::GetContext().train_flag = true;
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(false);
  auto prof_ptr = std::make_shared<MsprofCommandHandle>(prof_data);

  Status ret = ProfCtrlHandle(prof_type, nullptr, sizeof(prof_data));
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestGeProfilingManager, handle_unsubscribe_info) {
  uint32_t prof_type = kProfCommandhandleModelUnsubscribe;
  MsprofCommandHandle prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  domi::GetContext().train_flag = true;
  ProfilingProperties::Instance().SetProfilingLoadOfflineFlag(false);
  ProfilingProperties::Instance().SetSubscribeInfo(0, 1, true);
  auto prof_ptr = std::make_shared<MsprofCommandHandle>(prof_data);
  (void)ProfCtrlHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_NO_THROW(ProfilingProperties::Instance().CleanSubscribeInfo());
}

TEST_F(UtestGeProfilingManager, set_subscribe_info) {
  ProfilingProperties::Instance().SetSubscribeInfo(0, 1, true);
  const auto &subInfo = ProfilingProperties::Instance().GetSubscribeInfo();
  EXPECT_EQ(subInfo.prof_switch, 0);
  EXPECT_EQ(subInfo.graph_id, 1);
  EXPECT_EQ(subInfo.is_subscribe, true);
}

TEST_F(UtestGeProfilingManager, clean_subscribe_info) {
  ProfilingProperties::Instance().CleanSubscribeInfo();
  const auto &subInfo = ProfilingProperties::Instance().GetSubscribeInfo();
  EXPECT_EQ(subInfo.prof_switch, 0);
  EXPECT_EQ(subInfo.graph_id, 0);
  EXPECT_EQ(subInfo.is_subscribe, false);
}

TEST_F(UtestGeProfilingManager, get_model_id_success) {
  ProfilingManager::Instance().SetGraphIdToModelMap(0, 1);
  uint32_t model_id = 0;
  Status ret = ProfilingManager::Instance().GetModelIdFromGraph(0, model_id);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilingManager, get_model_id_failed) {
  ProfilingManager::Instance().SetGraphIdToModelMap(0, 1);
  uint32_t model_id = 0;
  Status ret = ProfilingManager::Instance().GetModelIdFromGraph(10, model_id);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestGeProfilingManager, get_device_id_success) {
  ProfilingManager::Instance().SetGraphIdToDeviceMap(0, 1);
  uint32_t device_id = 0;
  Status ret = ProfilingManager::Instance().GetDeviceIdFromGraph(0, device_id);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilingManager, get_device_id_failed) {
  ProfilingManager::Instance().SetGraphIdToDeviceMap(0, 1);
  uint32_t device_id = 0;
  Status ret = ProfilingManager::Instance().GetDeviceIdFromGraph(10, device_id);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestGeProfilingManager, start_profiling_failed) {
  const uint64_t module = PROF_OP_DETAIL_MASK | PROF_TRAINING_TRACE;
  std::map<std::string, std::string> cmd_params_map;
  cmd_params_map["key"] = "value";
  Status ret = ProfilingManager::Instance().ProfStartProfiling(module, cmd_params_map);
  EXPECT_EQ(ret, ge::FAILED);
}


void BuildTaskDescInfo(TaskDescInfo &task_info) {
  task_info.op_name = "test";
  std::string op_type (80,'x');
  task_info.op_type = op_type;
  task_info.shape_type = "static";
  task_info.block_dim = 1;
  task_info.task_id = 1;
  task_info.cur_iter_num = 1;
  task_info.task_type = "AI_CPU";
}




class DModelListener : public ge::ModelListener {
 public:
  DModelListener() {
  };
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode,
                       std::vector<gert::Tensor> &outputs) {
    GELOGI("In Call back. OnComputeDone");
    return SUCCESS;
  }
};
shared_ptr<ge::ModelListener> g_label_call_back(new DModelListener());

namespace {


class UtestGeProfilingManager1 : public testing::Test {
 public:
    ComputeGraphPtr graph;
    shared_ptr<DavinciModel> model;
    uint32_t model_id;

 protected:
  void SetUp() {
    graph = make_shared<ComputeGraph>("default");

    model_id = 1;
    model = MakeShared<DavinciModel>(1, g_label_call_back);
    model->SetId(model_id);
    model->om_name_ = "testom";
    model->name_ = "test";
    ModelManager::GetInstance().InsertModel(model_id, model);

    model->ge_model_ = make_shared<GeModel>();
    model->runtime_param_.mem_base = 0x08000000;
    model->runtime_param_.mem_size = 5120000;
  }
  void TearDown() {
    ModelManager::GetInstance().DeleteModel(model_id);
  }
};
}

TEST_F(UtestGeProfilingManager1, prof_model_unsubscribe) {
  Status retStatus;

  retStatus = ProfilingManager::Instance().ProfModelUnsubscribe(model->GetDeviceId(), 1);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ModelManager::GetInstance().ModelSubscribe(1);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ProfilingManager::Instance().ProfInit(PROF_MODEL_LOAD_MASK);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeProfilingManager, prof_start_profiling) {
  Status retStatus;
  const uint64_t module = 1;
  std::map<std::string, std::string> config_para;

  config_para["devNums"] = "ab";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "12356890666666";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "-1";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "1";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devIdList"] = "ab";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devIdList"] = "12356890666666";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ProfilingManager::Instance().ProfStartProfiling(PROF_MODEL_EXECUTE_MASK, config_para);
  EXPECT_EQ(retStatus, SUCCESS);
  ProfilingManager::Instance().RecordLoadedModelId(1);
  retStatus = ProfilingManager::Instance().ProfStartProfiling(PROF_MODEL_LOAD_MASK, config_para);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeProfilingManager, StartTaskEventProfiling_Ok) {
  const uint64_t module = 0xffffffffUL;
  std::map<std::string, std::string> config_para;
  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  auto retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_TRUE(ProfilingProperties::Instance().IsTaskEventProfiling());
}

TEST_F(UtestGeProfilingManager, ProfOpDetailProfiling_Ok) {
  const uint64_t module = PROF_OP_DETAIL_MASK;
  std::map<std::string, std::string> config_para;
  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  ProfilingManager::Instance().RecordLoadedModelId(2005U);
  Status ret = ProfilingManager::Instance().ProfStartProfiling(module, config_para, 1);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(ProfilingProperties::Instance().IsOpDetailProfiling());
  ret = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_FALSE(ProfilingProperties::Instance().IsOpDetailProfiling());
}

TEST_F(UtestGeProfilingManager, StopTaskEventProfiling_Ok) {
  const uint64_t module = 0xffffffffUL;
  std::map<std::string, std::string> config_para;
  ProfilingProperties::Instance().SetTaskEventProfiling(true);
  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  auto retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_FALSE(ProfilingProperties::Instance().IsTaskEventProfiling());
}

TEST_F(UtestGeProfilingManager, prof_stop_profiling) {
  Status retStatus;
  const uint64_t module = 1;
  std::map<std::string, std::string> config_para;

  config_para["devNums"] = "ab";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "12356890666666";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "-1";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "1";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devIdList"] = "ab";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devIdList"] = "12356890666666";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, FAILED);

  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  retStatus = ProfilingManager::Instance().ProfStopProfiling(module, config_para);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ProfilingManager::Instance().ProfStopProfiling(PROF_MODEL_EXECUTE_MASK, config_para);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeProfilingManager, registe_element) {
  int64_t first_idx = -1;
  int64_t second_idx = -1;
  ProfilingProperties::Instance().SetLoadProfiling(true);
  ProfilingProperties::Instance().SetOpDetailProfiling(true);
  ProfilingProperties::Instance().SetTaskEventProfiling(true);
  ProfilingManager::Instance().RegisterElement(first_idx, "tiling1");
  ProfilingManager::Instance().RegisterElement(second_idx, "tiling2");
  ProfilingProperties::Instance().SetLoadProfiling(false);
  ProfilingProperties::Instance().SetOpDetailProfiling(false);
  EXPECT_EQ(first_idx + 1, second_idx);
}

TEST_F(UtestGeProfilingManager, HandleCtrlSetStepInfo_ReportOk) {
  ProfStepInfoCmd_t info{1, 1, (void *)0x01};
  uint32_t prof_type = PROF_CTRL_STEPINFO;
  Status ret = ProfCtrlHandle(prof_type, &info, sizeof(info));
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilingManager, StartProfilingWithDeviceNum_Ok) {
  const uint64_t module = 0xffffffffUL;
  std::map<std::string, std::string> config_para;
  config_para["devNums"] = "4";
  config_para["devIdList"] = "0,1,2,8";
  config_para["heterogeneous_host"] = "1";
  auto retStatus = ProfilingManager::Instance().ProfStartProfiling(module, config_para);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_TRUE(ProfilingProperties::Instance().IsTaskEventProfiling());
  ProfilingProperties::Instance().ClearProperties();
}

TEST_F(UtestGeProfilingManager, ProfilerCollector_Ok_RecordStart) {
  size_t records = 0UL;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == ge::InfoType::kEvent) {
      ++records;
      auto info = reinterpret_cast<MsprofEvent *>(data);
      EXPECT_EQ(info->level, MSPROF_REPORT_MODEL_LEVEL);
      EXPECT_EQ(info->itemId, 10);
      EXPECT_EQ(info->requestId, 1);
      EXPECT_EQ(info->type, static_cast<uint32_t>(gert::GeProfInfoType::kModelExecute));
      return 0;
    }

    if (type == ge::InfoType::kApi) {
      ++records;
      auto info = reinterpret_cast<MsprofApi*>(data);
      EXPECT_EQ(info->level, MSPROF_REPORT_NODE_LEVEL);
      EXPECT_EQ(info->itemId, 0);
      EXPECT_EQ(info->type, static_cast<uint32_t>(gert::GeProfInfoType::kStepInfo));
    }
    return 0;
  };
  ge::diagnoseSwitch::EnableDeviceProfiling();
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  ProfilerCollector profiler_collector{10, 10};
  EXPECT_EQ(profiler_collector.RecordStart((void *)0x01), ge::SUCCESS);
  EXPECT_EQ(records, 2);
  ge::diagnoseSwitch::DisableProfiling();
}

TEST_F(UtestGeProfilingManager, ProfilerCollector_Ok_RecordEnd) {
  size_t records = 0UL;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == ge::InfoType::kEvent) {
      ++records;
      auto info = reinterpret_cast<MsprofEvent *>(data);
      EXPECT_EQ(info->level, MSPROF_REPORT_MODEL_LEVEL);
      EXPECT_EQ(info->itemId, 10);
      EXPECT_EQ(info->requestId, 1);
      EXPECT_EQ(info->type, static_cast<uint32_t>(gert::GeProfInfoType::kModelExecute));
      return 0;
    }

    if (type == ge::InfoType::kApi) {
      ++records;
      auto info = reinterpret_cast<MsprofApi*>(data);
      EXPECT_EQ(info->level, MSPROF_REPORT_NODE_LEVEL);
      EXPECT_EQ(info->itemId, 1);
      EXPECT_EQ(info->type, static_cast<uint32_t>(gert::GeProfInfoType::kStepInfo));
    }

    if (type == ge::InfoType::kInfo) {
      ++records;
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_EQ(info->type, MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE);
      auto graph_id_info = reinterpret_cast<MsprofGraphIdInfo *>(info->data);
      EXPECT_EQ(graph_id_info->graphId, 99);
      EXPECT_EQ(graph_id_info->modelId, 10);
      return 0;
    }
    return 0;
  };
  ge::diagnoseSwitch::EnableDeviceProfiling();
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  ProfilerCollector profiler_collector{10, 99};
  EXPECT_EQ(profiler_collector.RecordEnd((void *)0x01), ge::SUCCESS);
  EXPECT_EQ(records, 3);
  ge::diagnoseSwitch::DisableProfiling();
}
