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
#include "common/dump/dump_manager.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_local_context.h"
#include "depends/profiler/src/dump_stub.h"
#include "session/inner_session.h"
#include "common/global_variables/diagnose_switch.h"
#include "runtime_stub.h"
#include "rt_error_codes.h"

namespace ge {
namespace {
Status LoadTask(const DumpProperties &dump_properties) {
  return ge::SUCCESS;
}

Status UnloadTask(const DumpProperties &dump_properties) {
  return ge::SUCCESS;
}
}  // namespace

class UTEST_dump_manager : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(UTEST_dump_manager, is_dump_open_success) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  (void)DumpManager::GetInstance().SetDumpConf(dump_config);
  const auto dump = DumpManager::GetInstance().GetDumpProperties(0);
  bool result = dump.IsDumpOpen();
  EXPECT_EQ(result, true);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UTEST_dump_manager, is_dump_op_success) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "off";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_dump_manager, is_dump_single_op_close_success) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "off";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UTEST_dump_manager, dump_status_empty) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_op_switch = "off";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
}

// dump_debug and debug_status are on
TEST_F(UTEST_dump_manager, dump_op_debug_on) {
  DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_status = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
}

// just dump_status is on
TEST_F(UTEST_dump_manager, dump_status_without_dump_list) {
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

// dump_status is on with dump_list
TEST_F(UTEST_dump_manager, dump_status_with_dump_list) {
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.model_name = "test";
  dump_list.layers.push_back("first");
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UTEST_dump_manager, add_dump_list_with_op_range) {
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.model_name = "test";
  dump_list.dump_op_ranges.push_back(std::make_pair("a","b"));
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UTEST_dump_manager, add_valid_dump_opblacklist_success) {
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.model_name = "test";
  dump_list.layers.push_back("first");
  DumpBlacklist blacklist;
  blacklist.name = "conv";
  blacklist.pos = {"input0", "input1", "output0"};
  dump_list.opname_blacklist.push_back(blacklist);
  dump_list.optype_blacklist.push_back(blacklist);
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UTEST_dump_manager, add_invalid_dump_opblacklist_failed) {
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.model_name = "test";
  dump_list.layers.push_back("first");
  DumpBlacklist blacklist;
  blacklist.name = "conv";
  blacklist.pos = {"inputx", "ouput"};
  dump_list.opname_blacklist.push_back(blacklist);
  dump_list.optype_blacklist.push_back(blacklist);
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UTEST_dump_manager, not_need_do_dump) {
  DumpConfig dump_config;
  dump_config.dump_status = "off";
  dump_config.dump_debug = "off";
  DumpProperties dump_properties;
  bool ret = DumpManager::GetInstance().NeedDoDump(dump_config, dump_properties);
  EXPECT_EQ(ret, false);
}

TEST_F(UTEST_dump_manager, get_cfg_from_option) {
  DumpConfig dump_config;
  std::map<std::string, std::string> options;
  options[OPTION_EXEC_ENABLE_DUMP] = "1";
  options[OPTION_EXEC_DUMP_MODE] = "all";
  bool ret = DumpManager::GetInstance().GetCfgFromOption(options, dump_config);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(dump_config.dump_status, "device_on");
  EXPECT_EQ(dump_config.dump_mode, "all");
}

TEST_F(UTEST_dump_manager, EnableExceptionDump_Ok_ByDumpConfig) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  dump_config.dump_exception = "1";
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::SUCCESS);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, EnableLiteException_Ok_ByDumpConfig) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  dump_config.dump_exception = "2";
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::SUCCESS);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, Enable_lite_exception_Exception_Ok_ByDumpConfig) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  dump_config.dump_exception = "lite_exception";
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::SUCCESS);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, Enable_aic_err_brief_dump_Exception_Ok_ByDumpConfig) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  dump_config.dump_exception = "aic_err_brief_dump";
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::SUCCESS);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, Enable_aic_err_norm_dump_Exception_Ok_ByDumpConfig) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  dump_config.dump_exception = "aic_err_norm_dump";
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::SUCCESS);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, Enable_aic_err_detail_dump_Exception_Ok_ByDumpConfig) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  dump_config.dump_exception = "aic_err_detail_dump";
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::SUCCESS);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kDataDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kOverflowDump));
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, test_is_exception_dump_open_log) {
  const char_t *const kEnvRecordPath = "NPU_COLLECT_PATH";
  DumpConfig dump_config;
  dump_config.dump_status = "on";
  char_t npu_collect_path[MMPA_MAX_PATH] = "valid_path";
  mmSetEnv(kEnvRecordPath, &npu_collect_path[0U], MMPA_MAX_PATH);
  EXPECT_EQ(DumpManager::GetInstance().SetDumpConf(dump_config), ge::PARAM_INVALID);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  unsetenv(kEnvRecordPath);
}

TEST_F(UTEST_dump_manager, EnableDumpAgain_WithRegisFunc_OK) {
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

TEST_F(UTEST_dump_manager, SetDumpOffAfterOn_WithRegisFunc_OK) {
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

TEST_F(UTEST_dump_manager, dump_open_by_both_acl_and_options) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto dump = DumpManager::GetInstance().GetDumpProperties(1);
  EXPECT_EQ(dump.GetDumpMode(), "all");

  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_MODE, "input"},
                                              {OPTION_EXEC_DUMP_DATA, "tegnsor"},
                                              {OPTION_EXEC_DUMP_LAYER, "Conv2D"}};
  GetThreadLocalContext().SetGlobalOption(options);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 1;
  InnerSession inner_session(session_id, option);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  inner_session.AddDumpProperties(dp);
  dump = DumpManager::GetInstance().GetDumpProperties(1);
  EXPECT_EQ(dump.GetDumpMode(), "input");
  EXPECT_EQ(DumpManager::GetInstance().CheckIfAclDumpSet(), false);

  dump_config.dump_path = "/test1/";
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  dump = DumpManager::GetInstance().GetDumpProperties(1);
  EXPECT_EQ(dump.GetDumpMode(), "all");
  DumpManager::GetInstance().RemoveDumpProperties(1);
}

TEST_F(UTEST_dump_manager, acl_off_and_option_on_and_acl_off) {
  DumpConfig dump_config;
  dump_config.dump_status = "off";
  dump_config.dump_debug = "off";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), false);

  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_MODE, "all"},
                                              {OPTION_EXEC_DUMP_DATA, "tegnsor"},
                                              {OPTION_EXEC_DUMP_LAYER, "Conv2D"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, option);
  inner_session.AddDumpProperties(dp);
  dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), true);
  EXPECT_EQ(dump.GetDumpMode(), "all");

  dump_config.dump_status = "off";
  dump_config.dump_debug = "off";
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), false);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UTEST_dump_manager, acl_on_and_option_off_and_acl_on) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), true);

  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP, "0"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, option);
  inner_session.AddDumpProperties(dp);
  dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), false);

  dump_config.dump_mode = "output";
  ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), true);
  EXPECT_EQ(dump.GetDumpMode(), "output");
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UTEST_dump_manager, acl_on_and_option_none) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), true);

  DumpProperties dp;
  std::map<std::string, std::string> options {};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, option);
  inner_session.AddDumpProperties(dp);
  dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), true);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UTEST_dump_manager, acl_on_and_option_overflow_on) {
  DumpConfig dump_config;
  dump_config.dump_path = "/test";
  dump_config.dump_mode = "all";
  dump_config.dump_status = "on";
  dump_config.dump_op_switch = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), true);

  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_DEBUG_MODE, "aicore_overflow"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 0;
  InnerSession inner_session(session_id, option);
  inner_session.AddDumpProperties(dp);
  dump = DumpManager::GetInstance().GetDumpProperties(0);
  EXPECT_EQ(dump.IsDumpOpen(), false);
  DumpManager::GetInstance().RemoveDumpProperties(0);
}

TEST_F(UTEST_dump_manager, dump_data_valid) {
   DumpConfig dump_config;
   dump_config.dump_path = "/test";
   dump_config.dump_mode = "all";
   dump_config.dump_status = "on";
   dump_config.dump_data = "stats";
   auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
   EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_dump_manager, dump_data_invalid) {
   DumpConfig dump_config;
   dump_config.dump_path = "/test";
   dump_config.dump_mode = "all";
   dump_config.dump_status = "on";
   dump_config.dump_data = "test";
   auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
   EXPECT_EQ(ret, ge::PARAM_INVALID);
}

// dump specified op without model_name
TEST_F(UTEST_dump_manager, dump_specified_op_without_model_name) {
  DumpConfig dump_config;
  DumpProperties dump_properties;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.layers.push_back("first");
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetNormalDumpConf(dump_config, dump_properties);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(dump_properties.IsLayerNeedDump("model_test", "om_test", "first"));
}

TEST_F(UTEST_dump_manager, dump_with_watcher_model) {
  DumpConfig dump_config;
  DumpProperties dump_properties;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.layers.push_back("first");
  dump_list.watcher_nodes.push_back("second");
  dump_list.watcher_nodes.push_back("three");
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetNormalDumpConf(dump_config, dump_properties);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(dump_properties.IsDumpWatcherModelEnable());
  EXPECT_TRUE(dump_properties.IsLayerOpOnWatcherMode("first"));
  EXPECT_TRUE(dump_properties.IsWatcherNode("second"));
  EXPECT_TRUE(dump_properties.IsWatcherNode("three"));
}

TEST_F(UTEST_dump_manager, dump_without_watcher_model) {
  DumpConfig dump_config;
  DumpProperties dump_properties;
  dump_config.dump_status = "on";
  ModelDumpConfig dump_list;
  dump_list.layers.push_back("first");
  dump_config.dump_list.push_back(dump_list);
  auto ret = DumpManager::GetInstance().SetNormalDumpConf(dump_config, dump_properties);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_FALSE(dump_properties.IsDumpWatcherModelEnable());
  EXPECT_FALSE(dump_properties.IsLayerOpOnWatcherMode("first"));
  EXPECT_FALSE(dump_properties.IsWatcherNode("second"));
}

TEST_F(UTEST_dump_manager, dump_op_debug_on_and_set_op_switch) {
  DumpConfig dump_config;
  dump_config.dump_debug = "on";
  dump_config.dump_status = "on";
  auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsSingleOpNeedDump(), false);
  DumpManager::GetInstance().SetDumpOpSwitch(0, "on");
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(0).IsSingleOpNeedDump(), true);

  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_DEBUG_MODE, "aicore_overflow"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
  std::map<std::string, std::string> option = {};
  uint64_t session_id = 1;
  InnerSession inner_session(session_id, option);
  inner_session.AddDumpProperties(dp);
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(1).IsSingleOpNeedDump(), false);
  DumpManager::GetInstance().SetDumpOpSwitch(1, "on");
  EXPECT_EQ(DumpManager::GetInstance().GetDumpProperties(1).IsSingleOpNeedDump(), true);
}

TEST_F(UTEST_dump_manager, dump_op_debug_enable) {
    DumpConfig dump_config;
    dump_config.dump_debug = "on";
    dump_config.dump_path = "/test";
    auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
    EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_dump_manager, InitL0ExceptionDump_Ok_ByOption) {
  ge::diagnoseSwitch::DisableDumper();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);

  // enable l0 exception dump
  std::map<std::string, std::string> run_options = {{"ge.exec.enable_exception_dump", "2"}, {"ge.exec.dumpPath", "./"}};
  DumpManager::GetInstance().Init(run_options);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(DumpManager::GetInstance().IsDumpExceptionOpen());
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();

  // set dump path to env ascend work path
  const char_t *const path = "/home/ascend/work/path";
  mmSetEnv("ASCEND_WORK_PATH", path, 1);
  DumpManager::GetInstance().Init(run_options);

  std::string dump_path;
  EXPECT_TRUE(DumpStub::GetInstance().GetDumpPath(Adx::DumpType::ARGS_EXCEPTION, dump_path));
  EXPECT_EQ(dump_path, path);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(UTEST_dump_manager, InitL1ExceptionDump_Ok_ByOption) {
  ge::diagnoseSwitch::DisableDumper();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);

  // enable l1 exception dump
  std::map<std::string, std::string> run_options = {{"ge.exec.enable_exception_dump", "1"}};
  DumpManager::GetInstance().Init(run_options);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_TRUE(DumpManager::GetInstance().IsDumpExceptionOpen());
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();

  // set dump path to env ascend work path
  const char_t *const path = "/home/ascend/work/path";
  mmSetEnv("ASCEND_WORK_PATH", path, 1);
  DumpManager::GetInstance().Init(run_options);

  std::string dump_path;
  EXPECT_TRUE(DumpStub::GetInstance().GetDumpPath(Adx::DumpType::EXCEPTION, dump_path));
  EXPECT_EQ(dump_path, path);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(UTEST_dump_manager, InitException_Fail_ByInvalidOption) {
  ge::diagnoseSwitch::DisableDumper();
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);

  // invalie option
  std::map<std::string, std::string> run_options = {{"ge.exec.enable_exception_dump", "3"}};
  DumpManager::GetInstance().Init(run_options);
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_FALSE(DumpManager::GetInstance().IsDumpExceptionOpen());
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
}

TEST_F(UTEST_dump_manager, InitL1ExceptionDump_By_NPU_COLLECT_PATH) {
  ge::diagnoseSwitch::DisableDumper();
  mmSetEnv("NPU_COLLECT_PATH", "valid_path", 1);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  std::map<std::string, std::string> run_options;
  DumpManager::GetInstance().Init(run_options);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_TRUE(DumpManager::GetInstance().IsDumpExceptionOpen());
  std::string dump_path;
  EXPECT_TRUE(DumpStub::GetInstance().GetDumpPath(Adx::DumpType::EXCEPTION, dump_path));
  EXPECT_EQ(dump_path, "valid_path");
  unsetenv("NPU_COLLECT_PATH");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();
}

TEST_F(UTEST_dump_manager, InitL1ExceptionDump_By_NPU_COLLECT_PATH_EXE) {
  ge::diagnoseSwitch::DisableDumper();
  mmSetEnv("NPU_COLLECT_PATH_EXE", "valid_path", 1);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  std::map<std::string, std::string> run_options;
  DumpManager::GetInstance().Init(run_options);
  EXPECT_TRUE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump));
  EXPECT_FALSE(gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump));
  EXPECT_TRUE(DumpManager::GetInstance().IsDumpExceptionOpen());
  std::string dump_path;
  EXPECT_TRUE(DumpStub::GetInstance().GetDumpPath(Adx::DumpType::EXCEPTION, dump_path));
  EXPECT_EQ(dump_path, "valid_path");
  unsetenv("NPU_COLLECT_PATH_EXE");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();
}

TEST_F(UTEST_dump_manager, AddDumpProperties_With_DatadumpEnable) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  std::map<std::string, std::string> run_options;
  DumpManager::GetInstance().Init(run_options);

  GetThreadLocalContext().SetGlobalOption(
      {{"ge.exec.enableDump", "1"}, {"ge.exec.dumpPath", "./"}, {"ge.exec.dumpData", "stats"}});

  DumpProperties dump_properties;
  EXPECT_EQ(dump_properties.InitByOptions(), ge::SUCCESS);
  EXPECT_EQ(DumpManager::GetInstance().AddDumpProperties(0, dump_properties), ge::SUCCESS);
  EXPECT_TRUE(DumpStub::GetInstance().IsDumpEnable(Adx::DumpType::OPERATOR));
  DumpManager::GetInstance().RemoveDumpProperties(0);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();
}

TEST_F(UTEST_dump_manager, AddDumpProperties_With_enableDump_On_enableDumpDebug_On) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  std::map<std::string, std::string> run_options;
  DumpManager::GetInstance().Init(run_options);

  GetThreadLocalContext().SetGlobalOption({{"ge.exec.enableDump", "1"},
                                           {"ge.exec.enableDumpDebug", "1"},
                                           {"ge.exec.dumpPath", "./"},
                                           {"ge.exec.dumpData", "stats"}});

  DumpProperties dump_properties;
  EXPECT_EQ(dump_properties.InitByOptions(), ge::FAILED);
  DumpManager::GetInstance().RemoveDumpProperties(0);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
  DumpManager::GetInstance().Finalize();
  DumpStub::GetInstance().Reset();
}
}  // namespace ge
