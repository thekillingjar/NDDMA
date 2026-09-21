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

#include "ge/ge_api.h"
#include "framework/executor/ge_executor.h"
#include "session/session_manager.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_op.h"
#include "common/dump/dump_properties.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/dump/data_dumper.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "graph/load/model_manager/model_manager.h"
#include "depends/runtime/src/runtime_stub.h"
#include "framework/ge_runtime_stub/include/common/dump_checker.h"

using namespace std;
namespace ge {
namespace {
Status LoadTask(const DumpProperties &dump_properties) {
  return ge::SUCCESS;
}

Status UnloadTask(const DumpProperties &dump_properties) {
  return ge::SUCCESS;
}
}  // namespace

class DumpTest : public testing::Test {
 protected:
  void SetUp() {
    DumpManager::GetInstance().RemoveDumpProperties(kInferSessionId);
    RTS_STUB_SETUP();
  }
  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
  static void BuildGraphWithDumpOptions(const std::map<std::string, std::string> &dump_options) {
    std::map<std::string, std::string> options;
    //  OPTION_EXEC_DUMP_STEP
    //OPTION_EXEC_DUMP_MODE
    EXPECT_EQ(GEInitialize(options), SUCCESS);

    std::map<std::string, std::string> session_options;
    session_options = dump_options;
    Session session(session_options);

    GraphId graph_id = 1;
    const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
    Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

    EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

    vector<Tensor> inputs;
    vector<InputTensorInfo> tensors;
    EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);

    EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
    EXPECT_EQ(GEFinalize(), SUCCESS);
  }
};

TEST_F(DumpTest, TestSetDumpStatusByOption) {
  std::map<std::string, std::string> dump_options;
  dump_options.emplace(OPTION_EXEC_ENABLE_DUMP, "1");
  dump_options.emplace(OPTION_EXEC_ENABLE_DUMP_DEBUG, "0");
  dump_options.emplace(OPTION_EXEC_DUMP_PATH, "./");
  dump_options.emplace(OPTION_EXEC_DUMP_STEP, "0|5|10-20");
  dump_options.emplace(OPTION_EXEC_DUMP_MODE, "all");
  BuildGraphWithDumpOptions(dump_options);
}

TEST_F(DumpTest, TestSetDumpDebugByOption) {
  std::map<std::string, std::string> dump_options;
  dump_options.emplace(OPTION_EXEC_ENABLE_DUMP_DEBUG, "1");
  dump_options.emplace(OPTION_EXEC_DUMP_PATH, "./");
  dump_options.emplace(OPTION_EXEC_DUMP_DEBUG_MODE, OP_DEBUG_AICORE); // OP_DEBUG_ATOMIC /  OP_DEBUG_ALL
  BuildGraphWithDumpOptions(dump_options);

  dump_options[OPTION_EXEC_DUMP_DEBUG_MODE] = OP_DEBUG_ATOMIC;
  BuildGraphWithDumpOptions(dump_options);

  dump_options[OPTION_EXEC_DUMP_DEBUG_MODE] = OP_DEBUG_ALL;
  BuildGraphWithDumpOptions(dump_options);
}

TEST_F(DumpTest, TestSetDumpStatusByCmd) {
  Command dump_command;
  dump_command.cmd_type = "dump";
  dump_command.cmd_params = {DUMP_STATUS};
  GeExecutor ge_executor;
  EXPECT_EQ(ge_executor.CommandHandle(dump_command), ACL_ERROR_GE_COMMAND_HANDLE);

  // cmd params saved as: { key1, val1, key2, val2, key3, val3 }
  dump_command.cmd_params = {DUMP_FILE_PATH, "./",
                             DUMP_STATUS, "on",
                             DUMP_MODEL, "om",
                             DUMP_MODE, "all",
                             DUMP_LAYER, "relu"};
  EXPECT_EQ(ge_executor.CommandHandle(dump_command), SUCCESS);

//  DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsSingleOpNeedDump();

  dump_command.cmd_params = {DUMP_STATUS, "off"};
  EXPECT_EQ(ge_executor.CommandHandle(dump_command), SUCCESS);
}

TEST_F(DumpTest, TestSetDumpStatusByApi) {
  GeExecutor ge_executor;
  {
    ge::DumpConfig dump_cfg;
    dump_cfg.dump_path = "./dump/";
    dump_cfg.dump_mode = "all";
    dump_cfg.dump_status = "on";
    dump_cfg.dump_op_switch = "on";
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
    EXPECT_TRUE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen());
    EXPECT_FALSE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsOpDebugOpen());
  }

  {
    ge::DumpConfig dump_cfg;
    dump_cfg.dump_status = "off";
    dump_cfg.dump_debug = "off";
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
    EXPECT_FALSE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen());
  }
}

TEST_F(DumpTest, TestSetDumpDebugByApi) {
  GeExecutor ge_executor;
  {
    ge::DumpConfig dump_cfg;
    dump_cfg.dump_path = "./dump/";
    dump_cfg.dump_mode = "all";
    dump_cfg.dump_debug = "on";
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
    EXPECT_FALSE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen());
    EXPECT_TRUE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsOpDebugOpen());
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).GetOpDebugMode(), 3);
  }

  {
    ge::DumpConfig dump_cfg;
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
    EXPECT_FALSE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen());
  }
}

TEST_F(DumpTest, TestSetDumpDataByApi) {
  GeExecutor ge_executor;
  {
    ge::DumpConfig dump_cfg;
    dump_cfg.dump_path = "./dump/";
    dump_cfg.dump_mode = "all";
    dump_cfg.dump_status = "on";
    dump_cfg.dump_data = "stats";
    EXPECT_EQ(ge_executor.SetDump(dump_cfg), SUCCESS);
    EXPECT_TRUE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen());
    EXPECT_FALSE(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsOpDebugOpen());
    EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).GetOpDebugMode(), 0);
    EXPECT_EQ(strcmp(DumpManager::GetInstance().GetDumpProperties(kInferSessionId).GetDumpData().c_str(), "stats"), 0);
  }
}

TEST_F(DumpTest, TestCheckDumpDataByApi) {
  GeExecutor ge_executor;
  {
    ge::DumpConfig dump_cfg;
    dump_cfg.dump_path = "./dump/";
    dump_cfg.dump_mode = "all";
    dump_cfg.dump_status = "on";
    dump_cfg.dump_data = "test";
    EXPECT_NE(ge_executor.SetDump(dump_cfg), SUCCESS);
  }
}

TEST_F(DumpTest, TestGetDumpCfgByApi) {
  {
    DumpConfig dump_config;
    std::map<std::string, std::string> options;
    options[OPTION_EXEC_ENABLE_DUMP] = "1";
    options[OPTION_EXEC_DUMP_MODE] = "all";
    bool ret = DumpManager::GetCfgFromOption(options, dump_config);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(dump_config.dump_status, "device_on");
    EXPECT_EQ(dump_config.dump_mode, "all");
  }
}

TEST_F(DumpTest, EnableDumpAgain_WithRegisFunc_OK) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  DumpManager::GetInstance().RegisterCallBackFunc("Load", LoadTask);
  DumpManager::GetInstance().RegisterCallBackFunc("Unload", UnloadTask);
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(DumpTest, SetDumpOffAfterOn_WithRegisFunc_OK) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  DumpManager::GetInstance().RegisterCallBackFunc("Load", LoadTask);
  DumpManager::GetInstance().RegisterCallBackFunc("Unload", UnloadTask);
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  dump_config.dump_status = "off";
  dump_config.dump_debug = "off";
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(DumpTest, LoadDumpInfo_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  std::shared_ptr<OpDesc> op_desc_1(new OpDesc());
  op_desc_1->AddOutputDesc("test", GeTensorDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc_1, 0);
  string dump_mode = "output";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.ReLoadDumpInfo();
  data_dumper.UnloadDumpInfo();
  data_dumper.ReLoadDumpInfo();
}

TEST_F(DumpTest, LoadDumpInfoForPreprocessTask_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test1");
  data_dumper.SetModelId(233);
  std::shared_ptr<OpDesc> op_desc(new OpDesc());
  op_desc->AddOutputDesc("test1", GeTensorDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0, {}, {}, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL);
  string dump_mode = "output";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.ReLoadDumpInfo();
  data_dumper.UnloadDumpInfo();
  data_dumper.ReLoadDumpInfo();
}

TEST_F(DumpTest, DumpWorkspaceForPreprocessTask_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  data_dumper.is_op_debug_ = true;

  toolkit::aicpu::dump::Task task;
  auto op_desc = std::make_shared<ge::OpDesc>("conv", "conv");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});

  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);

  DataDumper::InnerDumpInfo inner_dump_info;
  inner_dump_info.op = op_desc;
  (void)data_dumper.DumpOutputWithTask(inner_dump_info, task);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0, {}, {}, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL);
  std::vector<uint64_t> space_addr = {23333U};
  data_dumper.SetWorkSpaceAddr(op_desc, space_addr);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(DumpTest, LoadDumpInfo_fail) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  RuntimeParam rts_param;

  DataDumper data_dumper(&rts_param);
  data_dumper.DumpShrink();
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(11);
  data_dumper.SaveEndGraphId(0U, 0U);
  data_dumper.SetComputeGraph(graph);
  data_dumper.need_generate_op_buffer_ = true;

  toolkit::aicpu::dump::Task task;
  DataDumper::InnerDumpInfo inner_dump_info{};

  auto op_desc = std::make_shared<ge::OpDesc>("conv", "conv");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(tensor, ATTR_DATA_DUMP_REF, "conv:input:1");
  op_desc->AddInputDesc(tensor);
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  graph->AddNode(op_desc);
  inner_dump_info.op = op_desc;
  data_dumper.dump_properties_.SetDumpMode(std::string("all"));
  data_dumper.is_op_debug_ = true;
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
  data_dumper.dump_properties_.SetDumpMode(std::string("input"));
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(DumpTest, BuildTaskInfoForPrint_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  auto op_desc = std::make_shared<ge::OpDesc>("conv", "conv");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});
  (void)AttrUtils::SetInt(op_desc, "current_context_id", 1);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  data_dumper.SavePrintDumpTask({0, 0, 0, 0}, nullptr, 0);
  data_dumper.SavePrintDumpTask({0, 0, 0, 0}, op_desc, 0);

  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  std::string dump_path = "/tmp/";
  DumpProperties dump_properties;
  dump_properties.SetDumpPath(dump_path);
  data_dumper.SetDumpProperties(dump_properties);

  int64_t buffer_size = 23;
  const std::string kOpDfxBufferSize = "_op_dfx_buffer_size";
  ge::AttrUtils::SetInt(op_desc, kOpDfxBufferSize, buffer_size);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  std::vector<uint64_t> space_addr = {23333U};
  data_dumper.SetWorkSpaceAddrForPrint(op_desc, space_addr);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();

  op_desc->SetWorkspaceBytes(vector<int64_t>{10});
  EXPECT_NE(data_dumper.LoadDumpInfo(), SUCCESS);
  data_dumper.UnloadDumpInfo();
}

TEST_F(DumpTest, DataDump_ReloadForStaticGraph) {
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 0U;
  shared_model->SetDeviceId(0);
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  ge::DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  (void)DumpManager::GetInstance().SetDumpConf(dump_config);
  (void)DumpManager::GetInstance().SetDumpConf(dump_config);
  const auto dump_properties = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(ModelManager::GetInstance().LoadTaskForDavinciModel(dump_properties), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x78000001);
  EXPECT_EQ(ModelManager::GetInstance().UnloadTaskForDavinciModel(dump_properties), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(davinci_model_id), SUCCESS);
  ge::DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(DumpTest, Test_LoadDumpInfo_st_stub_success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  auto dump_checker_stub = std::make_shared<DumpCheckRuntimeStub>();
  RuntimeStub::SetInstance(dump_checker_stub);
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("dt");
  data_dumper.SetModelId(1111);
  std::shared_ptr<OpDesc> op_desc(new OpDesc());
  op_desc->AddOutputDesc("test", GeTensorDesc());
  data_dumper.SaveDumpTask({0, 0, 0, 0}, op_desc, 0);
  string dump_mode = "output";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  EXPECT_EQ(dump_checker_stub->GetDumpChecker().GetLoadOpMappingInfoSize(), 1U);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  RuntimeStub::Reset();
}

TEST_F(DumpTest, DataDump_dumpoutput_opblacklist_success) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["conv"].output_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].output_indices = {1};
  new_blacklist_map["test"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor_0(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_1(GeShape(), FORMAT_NCHW, DT_FLOAT);

  auto op_desc = std::make_shared<ge::OpDesc>("conv", "conv");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});

  int32_t calc_type = 1;
  ge::AttrUtils::SetInt(tensor_1, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddOutputDesc(tensor_0);
  op_desc->AddOutputDesc(tensor_1);
  inner_dump_info.op = op_desc;
  EXPECT_EQ(data_dumper.DumpOutputWithTask(inner_dump_info, task), SUCCESS);
}

TEST_F(DumpTest, DataDump_dumpinput_opblacklist_success) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("input"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["conv"].input_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].input_indices = {1};
  new_blacklist_map["test"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);
  const auto blacklistMap = data_dumper.dump_properties_.GetModelDumpBlacklistMap();

  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor_0(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_1(GeShape(), FORMAT_NCHW, DT_FLOAT);

  auto op_desc = std::make_shared<ge::OpDesc>("conv", "conv");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});

  int32_t calc_type = 1;
  ge::AttrUtils::SetInt(tensor_1, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddInputDesc(tensor_0);
  op_desc->AddInputDesc(tensor_1);
  inner_dump_info.op = op_desc;
  EXPECT_EQ(data_dumper.DumpInput(inner_dump_info, task), SUCCESS);
}

TEST_F(DumpTest, DataDump_DataType_ModelInputNode_optypeblacklist_success) {
  RuntimeParam rts_param;
  rts_param.mem_size = 1024;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetOmName("test1");
  data_dumper.SetModelId(2333);
  data_dumper.dump_properties_.SetDumpMode(std::string("output"));
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_opname_blacklist["data"].output_indices = {1};
  model1_bl.dump_optype_blacklist["data"].output_indices = {0};
  new_blacklist_map["test1"] = model1_bl;
  data_dumper.dump_properties_.SetModelDumpBlacklistMap(new_blacklist_map);

  Status retStatus;
  toolkit::aicpu::dump::Task task;
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  auto op_desc = std::make_shared<ge::OpDesc>("data", "data");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});

  DataDumper::InnerDumpInfo inner_dump_info;
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  inner_dump_info.op = op_desc;
  inner_dump_info.is_task = false;
  inner_dump_info.input_anchor_index = 1;
  inner_dump_info.output_anchor_index = 0;
  inner_dump_info.cust_to_relevant_offset_ = {{0, 1}, {1, 2}};
  retStatus = data_dumper.DumpOutput(inner_dump_info, task);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(DumpTest, DataDump_dump_op_opblacklist_success) {
  DumpOp dump_op;
  DumpProperties dump_properties;

  OpDescPtr op_desc = std::make_shared<ge::OpDesc>("conv", "conv");
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(tensor, ATTR_DATA_DUMP_REF, "conv:input:1");
  ge::AttrUtils::SetBool(tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->AddOutputDesc(tensor);

  std::set<std::string> temp;
  dump_properties.model_dump_properties_map_.emplace("model1", temp);
  dump_properties.enable_dump_ = "1";
  dump_properties.dump_mode_ = "all";
  std::map<std::string, ModelOpBlacklist> new_blacklist_map;
  ModelOpBlacklist model1_bl;
  model1_bl.dump_optype_blacklist["conv"].input_indices = {0};
  model1_bl.dump_opname_blacklist["conv"].input_indices = {1};
  model1_bl.dump_opname_blacklist["conv"].output_indices = {0};
  model1_bl.dump_optype_blacklist["conv"].output_indices = {1};
  new_blacklist_map["model1"] = model1_bl;
  dump_properties.SetModelDumpBlacklistMap(new_blacklist_map);
  EXPECT_EQ(dump_op.global_step_, 0U);
  EXPECT_EQ(dump_op.loop_per_iter_, 0U);
  EXPECT_EQ(dump_op.loop_cond_, 0U);
  dump_op.SetLoopAddr(2U, 2U, 2U);
  EXPECT_EQ(dump_op.global_step_, 2U);
  EXPECT_EQ(dump_op.loop_per_iter_, 2U);
  EXPECT_EQ(dump_op.loop_cond_, 2U);
  dump_op.SetDynamicModelInfo("model1", "model2", 1);

  std::vector<uintptr_t> input_addrs;
  std::vector<uintptr_t> output_addrs;
  int64_t *addr_in = (int64_t *)malloc(1024);
  int64_t *addr_out = (int64_t *)malloc(1024);
  input_addrs.push_back(reinterpret_cast<uintptr_t>(addr_in));
  output_addrs.push_back(reinterpret_cast<uintptr_t>(addr_out));
  dump_op.SetDumpInfo(dump_properties, op_desc, input_addrs, output_addrs, nullptr);
  auto ret = dump_op.LaunchDumpOp(false);
  EXPECT_EQ(ret, ge::SUCCESS);

  toolkit::aicpu::dump::Task task;
  EXPECT_EQ(dump_op.DumpInput(task, op_desc, input_addrs), ge::SUCCESS);
  EXPECT_EQ(dump_op.DumpOutput(task, op_desc, output_addrs), ge::SUCCESS);

  free(addr_in);
  free(addr_out);
}

}  // namespace ge


