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
#include <memory>

#include "runtime/fast_v2/common/fake_node_helper.h"
#include "framework/common/string_util.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "common/global_variables/diagnose_switch.h"
#include "core/executor_error_code.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "macro_utils/dt_public_scope.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
class GlobalProfilingUT : public testing::Test {
 public:
  void assertRecord(const std::string line, const std::string &name, const std::string &type,
                    const std::string &event) {
    auto elements = ge::StringUtils::Split(line, ' ');
    ASSERT_EQ(elements.size(), 5);
    EXPECT_EQ(elements[2], name);
    EXPECT_EQ(elements[3], type);
    EXPECT_EQ(elements[4], event);
  }

 protected:
  void TearDown() {
    ge::diagnoseSwitch::DisableProfiling();
    ge::ProfilingTestUtil::Instance().Clear();
  }
};

TEST_F(GlobalProfilingUT, GetInstanceOk) {
  auto ins = GlobalProfilingWrapper::GetInstance();
  EXPECT_NE(ins, nullptr);
  EXPECT_EQ(ins->GetRecordCount(), 0UL);
  EXPECT_EQ(ins->GetGlobalProfiler(), nullptr);
}

TEST_F(GlobalProfilingUT, InitAndFreeOk) {
  auto ins = GlobalProfilingWrapper::GetInstance();
  EXPECT_EQ(ins->GetGlobalProfiler(), nullptr);
  ins->Init(4Ul);
  EXPECT_NE(ins->GetGlobalProfiler(), nullptr);
  EXPECT_EQ(ins->GetEnableFlags(), 4UL);
  ins->Free();
  EXPECT_EQ(ins->GetGlobalProfiler(), nullptr);
}

TEST_F(GlobalProfilingUT, RecordOk) {
  auto ins = GlobalProfilingWrapper::GetInstance();
  EXPECT_EQ(ins->GetGlobalProfiler(), nullptr);
  ins->Init(4UL);
  EXPECT_NE(ins->GetGlobalProfiler(), nullptr);
  ins->Record(0UL, 1UL, kModelStart, std::chrono::system_clock::now());
  EXPECT_EQ(ins->GetRecordCount(), 1);
  ins->Free();
  EXPECT_EQ(ins->GetGlobalProfiler(), nullptr);
}

TEST_F(GlobalProfilingUT, RegisterStringOk) {
  auto ins = GlobalProfilingWrapper::GetInstance();
  const auto name_idx = ins->RegisterString("NodeName1");
  const auto type_idx = ins->RegisterString("InferShape");
  EXPECT_TRUE(ins->RegisterString("NodeName1") == name_idx);
  EXPECT_TRUE(ins->RegisterString("InferShape") == type_idx);
}
TEST_F(GlobalProfilingUT, RegisterCallbackWorkOk) {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  mmSetEnv("ASCEND_WORK_PATH", current_path, 1);
  // register cb to diagnose_switch in constructor
  auto ins = GlobalProfilingWrapper::GetInstance();
  ins->Free();
  ASSERT_EQ(ins->GetGlobalProfiler(), nullptr);
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0UL);
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  ASSERT_NE(ins->GetGlobalProfiler(), nullptr);
  EXPECT_EQ(ins->GetEnableFlags(), BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kDevice}));
  ASSERT_NE(ins->GetGlobalProfiler(), nullptr);
  EXPECT_EQ(ins->GetEnableFlags(),
            BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost, ProfilingType::kDevice}));
  ge::diagnoseSwitch::MutableProfiling().SetEnableFlag(0UL);

  std::string ge_profiling_path = current_path;
  ge_profiling_path += "/ge_profiling_" + std::to_string(mmGetPid()) + ".txt";
  EXPECT_EQ(mmAccess(ge_profiling_path.c_str()), EN_OK);

  EXPECT_EQ(ins->GetGlobalProfiler(), nullptr);
  EXPECT_EQ(ins->GetEnableFlags(), 0UL);
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(GlobalProfilingUT, DumpAndClearOk) {
  const auto name_idx1 = GlobalProfilingWrapper::GetInstance()->RegisterString("NodeName1");
  const auto name_idx2 = GlobalProfilingWrapper::GetInstance()->RegisterString("NodeName2");
  const auto type_idx = GlobalProfilingWrapper::GetInstance()->RegisterString("InferShape");
  auto node_holder1 = FakeNodeHelper::FakeNode("NodeName1", "InferShape", 0);
  auto node_holder2 = FakeNodeHelper::FakeNode("NodeName2", "InferShape", 1);
  GlobalProfilingWrapper::GetInstance()->Init(
      BuiltInSubscriberUtil::BuildEnableFlags<ProfilingType>({ProfilingType::kGeHost}));

  auto profiler = std::make_unique<GeHostProfiler>(nullptr);
  ASSERT_NE(profiler, nullptr);
  profiler->is_inited_ = true;
  profiler->prof_extend_infos_.emplace_back(ProfExtendInfo{name_idx1, name_idx1, type_idx, 0});
  profiler->prof_extend_infos_.emplace_back(ProfExtendInfo{name_idx2, name_idx2, type_idx, 0});
  profiler->Record(nullptr, kModelStart);
  profiler->Record(nullptr, kModelEnd);
  profiler->Record(&node_holder1.node, kExecuteStart);
  profiler->Record(&node_holder1.node, kExecuteEnd);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kModelStart, nullptr, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kModelEnd, nullptr, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kExecuteStart, &node_holder2.node, kStatusSuccess);
  GeHostProfiler::OnExecuteEvent(1, profiler.get(), kExecuteEnd, &node_holder2.node, kStatusSuccess);

  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetRecordCount(), 8);

  std::stringstream ss;
  GlobalProfilingWrapper::GetInstance()->DumpAndFree(ss);

  auto lines = ge::StringUtils::Split(ss.str(), '\n');
  ASSERT_EQ(lines.size(), 11);
  EXPECT_EQ(lines[0], "ExecutorProfiler version: 2.0-SingleThread, dump start, records num: 8");
  EXPECT_EQ(lines[9], "Profiling dump end");
  assertRecord(lines[1], "[Model]", "[Execute]", "Start");
  assertRecord(lines[2], "[Model]", "[Execute]", "End");
  assertRecord(lines[3], "[NodeName1]", "[InferShape]", "Start");
  assertRecord(lines[4], "[NodeName1]", "[InferShape]", "End");
  assertRecord(lines[7], "[NodeName2]", "[InferShape]", "Start");
  assertRecord(lines[8], "[NodeName2]", "[InferShape]", "End");

  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler(), nullptr);
}

TEST_F(GlobalProfilingUT, ConstructGlobalProfiler_NonRegisterBuiltInString_WithGeProfilingOff) {
  auto gloabl_prof_ins = GlobalProfilingWrapper::GetInstance();
  gloabl_prof_ins->str_idx_ = 0UL;
  gloabl_prof_ins->idx_to_str_.clear();
  ge::diagnoseSwitch::DisableProfiling();
  gloabl_prof_ins = GlobalProfilingWrapper::GetInstance();
  EXPECT_EQ(gloabl_prof_ins->GetIdxToStr().size(), 0UL);
}

TEST_F(GlobalProfilingUT, ConstructGlobalProfiler_RegisterBuiltInString_WithGeProfilingOn) {
  auto gloabl_prof_ins = GlobalProfilingWrapper::GetInstance();
  gloabl_prof_ins->str_idx_ = 0UL;
  gloabl_prof_ins->idx_to_str_.clear();
  gloabl_prof_ins->is_builtin_string_registered_ = false;
  ge::diagnoseSwitch::EnableGeHostProfiling();
  gloabl_prof_ins = GlobalProfilingWrapper::GetInstance();
  EXPECT_EQ(gloabl_prof_ins->str_idx_, profiling::kProfilingIndexEnd);
  EXPECT_EQ(gloabl_prof_ins->GetIdxToStr().size(), kInitSize);
}

TEST_F(GlobalProfilingUT, RegisterString_TriggerBuiltInStringRegistry_WithGeProfilingOff) {
  auto gloabl_prof_ins = GlobalProfilingWrapper::GetInstance();
  gloabl_prof_ins->str_idx_ = 0UL;
  gloabl_prof_ins->idx_to_str_.clear();
  gloabl_prof_ins->is_builtin_string_registered_ = false;
  ge::diagnoseSwitch::DisableProfiling();
  const auto idx = GlobalProfilingWrapper::GetInstance()->RegisterString("test");
  EXPECT_EQ(idx, profiling::kProfilingIndexEnd);
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetIdxToStr()[profiling::kProfilingIndexEnd], "test");
}

TEST_F(GlobalProfilingUT, RegisterString_TriggerBuiltInStringRegistry_WithCannHostProfilingOn) {
  auto gloabl_prof_ins = GlobalProfilingWrapper::GetInstance();
  gloabl_prof_ins->str_idx_ = 0UL;
  gloabl_prof_ins->idx_to_str_.clear();
  gloabl_prof_ins->is_builtin_string_registered_ = false;
  ge::diagnoseSwitch::EnableCannHostProfiling();
  EXPECT_NE(gloabl_prof_ins->GetIdxToStr().size(), 0);
  EXPECT_EQ(gloabl_prof_ins->GetIdxToStr()[gert::profiling::kUnknownName], "UNKNOWNNAME");
}

TEST_F(GlobalProfilingUT, ProfilerUtil_BuildNodeBasic_Ok) {
  uint64_t prof_time = 1;
  uint32_t block_dim = 3;
  uint64_t type_hash_id = 100;
  uint64_t name_hash_id = 120;
  uint32_t op_flag = 0x1;
  auto task_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_AI_CORE);
  MsprofCompactInfo node_basic_info{};
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  ge::AttrUtils::SetInt(op_desc, "_op_impl_mode_enum", 0x40);
  GlobalProfilingWrapper::BuildNodeBasicInfo(op_desc, block_dim, {name_hash_id, type_hash_id}, task_type,
                                             node_basic_info);
  GlobalProfilingWrapper::BuildCompactInfo(prof_time, node_basic_info);
  EXPECT_NE(node_basic_info.threadId, 0);
  EXPECT_EQ(node_basic_info.timeStamp, prof_time);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(&node_basic_info.data.info[0]), name_hash_id);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&node_basic_info.data.info[8]), task_type);
  EXPECT_EQ(*reinterpret_cast<uint64_t *>(&node_basic_info.data.info[12]), type_hash_id);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&node_basic_info.data.info[20]), block_dim);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&node_basic_info.data.info[24]), op_flag);
}

TEST_F(GlobalProfilingUT, ProfilerReport_ReportApi_Ok) {
  uint64_t begin_time = 1;
  uint64_t end_time = 2;
  uint32_t item_id = 3;
  uint32_t api_type = 4;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    EXPECT_EQ(type, ge::InfoType::kApi);
    auto api = reinterpret_cast<MsprofApi *>(data);
    EXPECT_EQ(api->beginTime, begin_time);
    EXPECT_EQ(api->endTime, end_time);
    EXPECT_EQ(api->itemId, item_id);
    EXPECT_EQ(api->type, api_type);
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  EXPECT_EQ(GlobalProfilingWrapper::ReportApiInfo(begin_time, end_time, item_id, api_type), ge::SUCCESS);
}

TEST_F(GlobalProfilingUT, ProfilerReport_BuildContextIdInfo_Ok) {
  std::vector<uint32_t> context_ids(100, 6);
  uint64_t prof_time = 1;
  std::vector<ContextIdInfoWrapper> infos{};
  GlobalProfilingWrapper::BuildContextIdInfo(prof_time, context_ids, "test", infos);
  auto info = infos[0];
  EXPECT_EQ(info.context_id_info.dataLen, context_ids.size() * (sizeof(uint32_t) / sizeof(uint8_t)));
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info.context_id_info.data[8]), kMaxContextIdNum);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info.context_id_info.data[12]), context_ids[0]);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info.context_id_info.data[16]), context_ids[1]);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info.context_id_info.data[20]), context_ids[2]);
  EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info.context_id_info.data[24]), context_ids[3]);
}

TEST_F(GlobalProfilingUT, ProfilerReport_ReportTensorInfo_Ok) {
  ge::TaskDescInfo task_desc_info{};
  task_desc_info.op_name = "test";
  task_desc_info.task_type = MSPROF_GE_TASK_TYPE_AI_CORE;
  task_desc_info.input_shape = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
  task_desc_info.output_shape = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
  task_desc_info.input_format = {ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN};
  task_desc_info.output_format = {ge::FORMAT_CN, ge::FORMAT_CN, ge::FORMAT_CN};
  task_desc_info.input_data_type = {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT};
  task_desc_info.output_data_type = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT16};
  size_t record_num = 0;
  std::hash<std::string> hs;
  const auto name_hash = hs(task_desc_info.op_name);
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    ++record_num;
    if (record_num == 1) {
      EXPECT_EQ(type, ge::InfoType::kInfo);
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_EQ(info->dataLen, 232);
      EXPECT_EQ(*reinterpret_cast<uint64_t *>(&info->data[0]), name_hash);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[8]), 5);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[12]), MsprofGeTensorType::MSPROF_GE_TENSOR_TYPE_INPUT);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[16]), ge::FORMAT_DHWCN);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[20]), ge::DataType::DT_FLOAT);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[24]), 1);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[28]), 1);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[32]), 1);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[36]), 0);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[56]), MsprofGeTensorType::MSPROF_GE_TENSOR_TYPE_INPUT);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[68]), 1);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[100]), MsprofGeTensorType::MSPROF_GE_TENSOR_TYPE_INPUT);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[112]), 1);
    }
    if (record_num == 2) {
      auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      EXPECT_EQ(type, ge::InfoType::kInfo);
      EXPECT_EQ(info->dataLen, 56);
      EXPECT_EQ(*reinterpret_cast<uint64_t *>(&info->data[0]), name_hash);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[8]), 1);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[12]), MsprofGeTensorType::MSPROF_GE_TENSOR_TYPE_OUTPUT);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[16]), ge::FORMAT_CN);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[20]), ge::DataType::DT_FLOAT16);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[24]), 2);
      EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[36]), 0);
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  GlobalProfilingWrapper::ReportTensorInfo(1, true, task_desc_info);
}

TEST_F(GlobalProfilingUT, ProfilerReport_ReportGraphIdInfo_Ok) {
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    EXPECT_EQ(type, ge::InfoType::kInfo);
    auto info = reinterpret_cast<MsprofAdditionalInfo *>(data);
    EXPECT_EQ(info->level, MSPROF_REPORT_MODEL_LEVEL);
    EXPECT_EQ(*reinterpret_cast<uint64_t *>(&info->data[0]), 0);
    EXPECT_EQ(*reinterpret_cast<uint32_t *>(&info->data[8]), 5);
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  GlobalProfilingWrapper::ReportGraphIdMap(1, 1, {5,5}, false, 0);
}

TEST_F(GlobalProfilingUT, ProfilerReport_ProfileStepTrace_Ok) {
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::ProfilingType>(gert::ProfilingType::kTaskTime));
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    EXPECT_EQ(type, ge::InfoType::kApi);
    auto api = reinterpret_cast<MsprofApi *>(data);
    EXPECT_EQ(api->itemId, 2);
    EXPECT_EQ(api->type, static_cast<uint32_t>(gert::GeProfInfoType::kStepInfo));
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  EXPECT_EQ(GlobalProfilingWrapper::ProfileStepTrace(1, 3, 2, (void *)0x10), ge::SUCCESS);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
}

TEST_F(GlobalProfilingUT, ProfilerReport_RegisterType_WhenEnableProfiling) {
  ge::ProfilingTestUtil::Instance().Clear();
  GlobalProfilingWrapper::GetInstance()->RegisterProfType();
  EXPECT_EQ(ge::ProfilingTestUtil::Instance().GetRegisterType(MSPROF_REPORT_NODE_LEVEL,
                                                              static_cast<uint32_t>(GeProfInfoType::kStepInfo)),
            1);
}

TEST_F(GlobalProfilingUT, ProfilerReport_ReportEvent_Ok) {
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    EXPECT_EQ(type, ge::InfoType::kEvent);
    auto info = reinterpret_cast<MsprofEvent *>(data);
    EXPECT_EQ(info->level, MSPROF_REPORT_MODEL_LEVEL);
    EXPECT_EQ(info->itemId, 1);
    EXPECT_EQ(info->requestId, 2);
    EXPECT_EQ(info->type, static_cast<uint32_t>(GeProfInfoType::kModelLoad));
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  MsprofEvent event{};
  GlobalProfilingWrapper::ReportEvent(1, 2, GeProfInfoType::kModelLoad, event);
}

TEST_F(GlobalProfilingUT, ProfilerReport_GetProfModelId_MutipleTimes) {
  auto cur_model_id = GlobalProfilingWrapper::GetInstance()->GetProfModelId();
  GlobalProfilingWrapper::GetInstance()->IncProfModelId();
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetProfModelId(), ++cur_model_id);
  GlobalProfilingWrapper::GetInstance()->IncProfModelId();
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetProfModelId(), ++cur_model_id);
  GlobalProfilingWrapper::GetInstance()->IncProfModelId();
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetProfModelId(), ++cur_model_id);
  GlobalProfilingWrapper::GetInstance()->IncProfModelId();
  EXPECT_EQ(GlobalProfilingWrapper::GetInstance()->GetProfModelId(), ++cur_model_id);
}

TEST_F(GlobalProfilingUT, ProfilerReport_Ok_ReportLogicStreamInfo) {
  std::unordered_map<uint32_t, std::set<uint32_t>> logic_ids_to_physic_stream_ids{};
  logic_ids_to_physic_stream_ids[1] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 80; i++) {
    logic_ids_to_physic_stream_ids[2].insert(i);
  }
  size_t records = 0UL;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    EXPECT_EQ(type, ge::InfoType::kInfo);
    auto logic_info = reinterpret_cast<MsprofAdditionalInfo *>(data);
    auto logic_data = reinterpret_cast<MsprofLogicStreamInfo *>(logic_info->data);
    if (logic_data->logicStreamId == 1) {
      ++records;
      EXPECT_EQ(logic_data->physicStreamNum, 6);
      EXPECT_EQ(logic_data->physicStreamId[0], 1);
      EXPECT_EQ(logic_data->physicStreamId[1], 2);
      EXPECT_EQ(logic_data->physicStreamId[2], 3);
      EXPECT_EQ(logic_data->physicStreamId[3], 4);
      EXPECT_EQ(logic_data->physicStreamId[4], 5);
      EXPECT_EQ(logic_data->physicStreamId[5], 6);
      EXPECT_EQ(logic_data->physicStreamId[6], 0);
    }
    if (logic_data->logicStreamId == 2) {
      if (logic_data->physicStreamNum == MSPROF_PHYSIC_STREAM_ID_MAX_NUM) {
        ++records;
        EXPECT_EQ(logic_data->physicStreamId[0], 0);
        EXPECT_EQ(logic_data->physicStreamId[MSPROF_PHYSIC_STREAM_ID_MAX_NUM - 1], MSPROF_PHYSIC_STREAM_ID_MAX_NUM - 1);
      }
      if (logic_data->physicStreamNum == (80 % MSPROF_PHYSIC_STREAM_ID_MAX_NUM)) {
        ++records;
        EXPECT_EQ(logic_data->physicStreamId[0], MSPROF_PHYSIC_STREAM_ID_MAX_NUM);
        EXPECT_EQ(logic_data->physicStreamId[(80 % MSPROF_PHYSIC_STREAM_ID_MAX_NUM) - 1], 79);
        EXPECT_EQ(logic_data->physicStreamId[80 % MSPROF_PHYSIC_STREAM_ID_MAX_NUM], 0);
      }
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);
  gert::GlobalProfilingWrapper::ReportLogicStreamInfo(100, 10, logic_ids_to_physic_stream_ids, false);
  EXPECT_EQ(records, 3);
}

TEST_F(GlobalProfilingUT, ProfilerReport_ReportStaticOpMemInfo) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_prof");
  const uint32_t graph_mem_size = 0x1234;

  CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(graph, nullptr, graph_mem_size, 1U, std::numeric_limits<uint32_t>::max());

  const auto prof_flags = gert::GlobalProfilingWrapper::GetInstance()->GetEnableFlags();

  // test for kMemory is enabled
  EXPECT_NO_THROW(gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::ProfilingType>(gert::ProfilingType::kMemory)));
  CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(graph, nullptr, graph_mem_size, 1U, std::numeric_limits<uint32_t>::max());
  EXPECT_NO_THROW(gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(prof_flags));
}
}  // namespace gert
