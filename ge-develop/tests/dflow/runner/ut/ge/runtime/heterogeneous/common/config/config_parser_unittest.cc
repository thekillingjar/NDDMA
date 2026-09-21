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
#include <string>
#include <map>
#include <stdlib.h>

#include "framework/common/debug/ge_log.h"
#include "graph/ge_local_context.h"
#include "common/env_path.h"
#include "macro_utils/dt_public_scope.h"
#include "common/config/config_parser.h"
#include "macro_utils/dt_public_unscope.h"
#include "dflow/deployer/common/utils/deploy_location.h"
#include "depends/mmpa/src/mmpa_stub.h"

using namespace std;
namespace ge {
class MockMmpaRealPath : public ge::MmpaStubApiGe {
  public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }
};

class UtConfigParser : public testing::Test {
 public:
  UtConfigParser() {}
 protected:
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
  }
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
  }
  const std::string data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data"});
};

TEST_F(UtConfigParser, parse_server_info_success) {
  ge::DeployerConfig information;
  std::string config_file = PathUtils::Join({data_path, "valid/server/numa_config.json"});
  std::map<std::string, std::string> options;
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  ge::GetThreadLocalContext().SetGlobalOption(options);
  auto ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(information.remote_node_config_list.size(), 0);
  ASSERT_EQ(information.node_config.device_list.size(), 9);
  ASSERT_EQ(information.node_config.node_type, "TestNodeType1");
  ASSERT_EQ(information.node_config.resource_type, DeployLocation::IsX86() ? "X86" : "Aarch");
  for (const auto &device_config : information.node_config.device_list) {
    if (device_config.device_type == CPU) {
      ASSERT_EQ(device_config.resource_type, DeployLocation::IsX86() ? "X86" : "Aarch");
    } else {
      ASSERT_EQ(device_config.resource_type, "Ascend");
    }
  }
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}

TEST_F(UtConfigParser, parse_server_info_success_with_remote_no_device) {
  ge::DeployerConfig information;
  std::string config_file = PathUtils::Join({data_path, "valid/server/numa_config_with_remote_no_device.json"});
  auto ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(information.remote_node_config_list.size(), 1);
  ASSERT_EQ(information.remote_node_config_list[0].device_list.size(), 1);
  ASSERT_EQ(information.remote_node_config_list[0].device_list[0].device_type, CPU);
  ASSERT_EQ(information.remote_node_config_list[0].device_list[0].support_flowgw, false);
}

TEST_F(UtConfigParser, parse_config_without_resource_success) {
  ge::DeployerConfig information;
  std::string config_file = PathUtils::Join({data_path, "valid/server/numa_config2_without_resource.json"});
  auto ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(information.remote_node_config_list.size(), 1);
  ASSERT_EQ(information.node_config.device_list.size(), 3);
  ASSERT_EQ(information.node_config.node_type, "TestNodeType1");
  ASSERT_EQ(information.node_config.resource_type, DeployLocation::IsX86() ? "X86" : "Aarch");
  for (const auto &device_config : information.node_config.device_list) {
    if (device_config.device_type == CPU) {
      ASSERT_EQ(device_config.resource_type, DeployLocation::IsX86() ? "X86" : "Aarch");
    } else {
      ASSERT_EQ(device_config.resource_type, "Ascend");
    }
  }
}

TEST_F(UtConfigParser, parse_config_invalid_resource) {
  ge::DeployerConfig information;
  std::string config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config2_invalid_resource.json"});
  auto ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config2_invalid_resource2.json"});
  ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtConfigParser, parse_server_shuffle_node_success) {
  ge::DeployerConfig information;
  std::string config_file = PathUtils::Join({data_path, "valid/server/numa_config2.json"});
  auto ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(information.remote_node_config_list.size(), 1);
  ASSERT_EQ(information.node_config.node_id, 0);
  ASSERT_EQ(information.node_config.ipaddr, "10.216.56.14");
  ASSERT_EQ(information.node_config.node_mesh_index.size(), 2);
  ASSERT_EQ(information.node_config.node_mesh_index[0], 0);
  ASSERT_EQ(information.node_config.node_mesh_index[1], 0);
}

TEST_F(UtConfigParser, parse_server_info_failed) {
  ge::DeployerConfig information;
  std::string config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config_2p.json"});
  auto ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);

  // invalid json format
  config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config_invalid_cluster.json"});
  ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  NumaConfig numa_config;
  ret = ge::ConfigParser::InitNumaConfig(config_file, numa_config);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);

  // invalid node_def
  config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config_invalid_node_def.json"});
  ret = ge::ConfigParser::InitNumaConfig(config_file, numa_config);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);

  // invalid item_def
  config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config_invalid_item_def.json"});
  ret = ge::ConfigParser::InitNumaConfig(config_file, numa_config);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);

  // invalid num
  config_file = PathUtils::Join({data_path, "wrong_value/server/numa_config_invalid_num.json"});
  ret = ge::ConfigParser::ParseServerInfo(config_file, information);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  ret = ge::ConfigParser::InitNumaConfig(config_file, numa_config);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}
}

