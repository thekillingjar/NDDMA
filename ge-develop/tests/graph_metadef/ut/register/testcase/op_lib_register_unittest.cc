/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_lib_register_impl.h"
#include "register/register_base.h"

#include <gtest/gtest.h>
#include "framework/common/debug/ge_log.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
size_t g_lib_register_cnt = 1;
std::vector<size_t> init_func_vec;
const std::string custom_op_name = "libcust_opapi.so";
const std::string tmp_test_lib_dir = "./test_op_lib_register/";

uint32_t FakeFunc(ge::AscendString& path) {
  init_func_vec.emplace_back(g_lib_register_cnt);
  path = AscendString(to_string(g_lib_register_cnt).c_str());
  ++g_lib_register_cnt;
  return 0;
}

void ClearCache() {
  g_lib_register_cnt = 1;
  init_func_vec.clear();
  OpLibRegistry::GetInstance().vendor_funcs_.clear();
  OpLibRegistry::GetInstance().vendor_names_set_.clear();
  OpLibRegistry::GetInstance().op_lib_paths_ = "";
  OpLibRegistry::GetInstance().ClearHandles();
  OpLibRegistry::GetInstance().is_init_ = false;
}

void CreateVendorSoPath(const std::string &vendor_dir) {
  system(("mkdir -p " + vendor_dir).c_str());
  system(("touch " + vendor_dir + custom_op_name).c_str());
}

void CreateVendorOldRunbagDir(const std::string &vendor_dir) {
  system(("mkdir -p " + vendor_dir + "/op_proto/").c_str());
}

void DelVendorSoDir(const std::string &vendor_dir) {
  system(("rm -rf " + vendor_dir).c_str());
}

class MockMmpaForOpLib : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *fileName, int32_t mode) override {
    auto tmp_register = ge::OpLibRegister(fileName).RegOpLibInit(FakeFunc);
    return (void *) fileName;
  }
  int32_t DlClose(void *handle) override {
    return 0L;
  }
};
}

class OpLibRegisterUT : public testing::Test {
 protected:
  void SetUp() {
    system("pwd");
    system(("mkdir -p " + tmp_test_lib_dir).c_str());
  }

  void TearDown() {
    system(("rm -rf " + tmp_test_lib_dir).c_str());
    unsetenv("ASCEND_CUSTOM_OPP_PATH");
    ClearCache();
    ge::MmpaStub::GetInstance().Reset();
  }
};

TEST_F(OpLibRegisterUT, register_construct) {
  OpLibRegister tmp1("vendor1");
  auto tmp2 = tmp1;
  auto tmp3 = OpLibRegister(std::move(tmp1));
  EXPECT_NE(tmp1.impl_.get(), nullptr);
  EXPECT_EQ(tmp2.impl_.get(), nullptr);
  EXPECT_EQ(tmp3.impl_.get(), nullptr);
}

TEST_F(OpLibRegisterUT, register_same_vendor) {
  ClearCache();
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 0);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_names_set_.size(), 0);

  REGISTER_OP_LIB(vendor_1).RegOpLibInit(FakeFunc);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 1);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_names_set_.size(), 1);

  REGISTER_OP_LIB(vendor_1).RegOpLibInit(FakeFunc);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 1);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_names_set_.size(), 1);
}

TEST_F(OpLibRegisterUT, register_direct_link) {
  ClearCache();
  REGISTER_OP_LIB(vendor_1).RegOpLibInit(FakeFunc);
  REGISTER_OP_LIB(vendor_2).RegOpLibInit(FakeFunc);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 2);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_names_set_.size(), 2);
  unsetenv("ASCEND_CUSTOM_OPP_PATH");
  std::string custom_path = aclGetCustomOpLibPath();
  std::vector<size_t> expect_vec{1, 2};
  EXPECT_EQ(init_func_vec, expect_vec);
  EXPECT_EQ("1:2", custom_path);
}

TEST_F(OpLibRegisterUT, register_direct_link_and_env_priority) {
  ClearCache();
  REGISTER_OP_LIB(vendor_1).RegOpLibInit(FakeFunc);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 1);

  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForOpLib>());
  std::string vendor_2_dir = tmp_test_lib_dir + "/vendor_2/";
  CreateVendorSoPath(vendor_2_dir);
  std::string vendor_3_dir = tmp_test_lib_dir + "/vendor_3/";
  CreateVendorSoPath(vendor_3_dir);
  std::string env_val = vendor_2_dir + ":" + vendor_3_dir;
  mmSetEnv("ASCEND_CUSTOM_OPP_PATH", env_val.c_str(), 1);

  auto ret = OpLibRegistry::GetInstance().PreProcessForCustomOp();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 3);
  std::vector<size_t> expect_vec{1, 2, 3};
  EXPECT_EQ(init_func_vec, expect_vec);
  EXPECT_EQ(OpLibRegistry::GetInstance().handles_.size(), 2);
  EXPECT_EQ(OpLibRegistry::GetInstance().is_init_, true);
  ret = OpLibRegistry::GetInstance().PreProcessForCustomOp();
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  DelVendorSoDir(vendor_2_dir);
  DelVendorSoDir(vendor_3_dir);
}

TEST_F(OpLibRegisterUT, register_coexistence_direct_link) {
  ClearCache();
  REGISTER_OP_LIB(vendor_1).RegOpLibInit(FakeFunc);
  EXPECT_EQ(OpLibRegistry::GetInstance().vendor_funcs_.size(), 1);
  std::string old_dir = tmp_test_lib_dir + "/vendor_2/";
  mmSetEnv("ASCEND_CUSTOM_OPP_PATH", old_dir.c_str(), 1);
  CreateVendorOldRunbagDir(old_dir);
  auto ret = OpLibRegistry::GetInstance().PreProcessForCustomOp();
  EXPECT_EQ(ret, SUCCESS);
  std::string custom_path = aclGetCustomOpLibPath();
  std::string expect_path = "1:" + old_dir;
  EXPECT_EQ(expect_path, custom_path);
  DelVendorSoDir(old_dir);
}

TEST_F(OpLibRegisterUT, register_coexistence_env) {
  ClearCache();
  std::string old_dir = tmp_test_lib_dir + "/vendor_old/";
  CreateVendorOldRunbagDir(old_dir);

  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForOpLib>());
  std::string vendor_2_dir = tmp_test_lib_dir + "/vendor_new/";
  CreateVendorSoPath(vendor_2_dir);
  std::string env_val = old_dir + ":" + vendor_2_dir + ":";
  mmSetEnv("ASCEND_CUSTOM_OPP_PATH", env_val.c_str(), 1);

  auto ret = OpLibRegistry::GetInstance().PreProcessForCustomOp();
  EXPECT_EQ(ret, SUCCESS);
  std::string custom_path = aclGetCustomOpLibPath();
  std::string expect_path = "1:" + env_val;
  EXPECT_EQ(expect_path, custom_path);
  DelVendorSoDir(old_dir);
  DelVendorSoDir(vendor_2_dir);
}
} // namespace ge