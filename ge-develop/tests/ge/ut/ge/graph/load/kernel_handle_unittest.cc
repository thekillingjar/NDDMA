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
#include <gmock/gmock.h>

#include "framework/common/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace std;

namespace ge {
class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD5(rtKernelGetAddrAndPrefCntV2, int32_t(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                                    const uint32_t flag, rtKernelDetailInfo_t *kernelInfo));
};

class UtestTBEKernelHandle : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    MockRuntime::Reset();
  }
};


// test InitTbeHandleWithFfts
TEST_F(UtestTBEKernelHandle, init_tbe_handle_with_ffts) {
  TBEKernelHandle kernel_handle;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  // without tbe_kernel
  EXPECT_EQ(kernel_handle.RegisterAutoThreadHandle(op_desc), INTERNAL_ERROR);

  std::vector<OpKernelBinPtr> tbe_kernel;
  vector<char> buffer;
  string key = op_desc->GetName();
  OpKernelBinPtr tbe_kernel_ptr0 = std::make_shared<ge::OpKernelBin>(key + "_0", std::move(buffer));
  OpKernelBinPtr tbe_kernel_ptr1 = std::make_shared<ge::OpKernelBin>(key + "_1", std::move(buffer));
  tbe_kernel.push_back(tbe_kernel_ptr0);
  tbe_kernel.push_back(tbe_kernel_ptr1);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel);
  // without _register_stub_func
  EXPECT_EQ(kernel_handle.RegisterAutoThreadHandle(op_desc), INTERNAL_ERROR);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc,
                 const uint32_t flag, rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  vector<string> bin_file_keys;
  bin_file_keys.emplace_back(op_desc->GetName() + "_0");
  bin_file_keys.emplace_back(op_desc->GetName() + "_1");
  AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  vector<string> json_list;
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetListStr(op_desc, "_thread_" + TVM_ATTR_NAME_MAGIC, json_list);
  vector<string> meta_data_list;
  meta_data_list.emplace_back("meta_data_0");
  meta_data_list.emplace_back("meta_data_1");
  AttrUtils::SetListStr(op_desc, "_thread_" +TVM_ATTR_NAME_METADATA, meta_data_list);
  vector<string> thread_kernel_names = {tbe_kernel_ptr0->GetName(), tbe_kernel_ptr1->GetName()};
  AttrUtils::SetListStr(op_desc, "_thread_kernelname", thread_kernel_names);
  EXPECT_EQ(kernel_handle.RegisterAutoThreadHandle(op_desc), SUCCESS);
  // rtQueryFunctionRegistered(bin_file_key) failed
  EXPECT_EQ(kernel_handle.used_tbe_handle_map_.size(), 2);
}

TEST_F(UtestTBEKernelHandle, init_tbe_handle_fail_with_ffts) {
  TBEKernelHandle kernel_handle;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  std::vector<OpKernelBinPtr> tbe_kernel;
  vector<char> buffer;
  string key = op_desc->GetName();
  OpKernelBinPtr tbe_kernel_ptr0 = std::make_shared<ge::OpKernelBin>(key + "_0", std::move(buffer));
  OpKernelBinPtr tbe_kernel_ptr1 = std::make_shared<ge::OpKernelBin>(key + "_1", std::move(buffer));
  tbe_kernel.push_back(tbe_kernel_ptr0);
  tbe_kernel.push_back(tbe_kernel_ptr1);
  op_desc->SetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc,
                 const uint32_t flag, rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 3;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));

  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x1);
  vector<string> bin_file_keys;
  bin_file_keys.emplace_back(op_desc->GetName() + "_0");
  bin_file_keys.emplace_back(op_desc->GetName() + "_1");
  AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  vector<string> json_list;
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetListStr(op_desc, "_thread_" + TVM_ATTR_NAME_MAGIC, json_list);
  vector<string> meta_data_list;
  meta_data_list.emplace_back("meta_data_0");
  meta_data_list.emplace_back("meta_data_1");
  AttrUtils::SetListStr(op_desc, "_thread_" +TVM_ATTR_NAME_METADATA, meta_data_list);
  vector<string> thread_kernel_names = {tbe_kernel_ptr0->GetName(), tbe_kernel_ptr1->GetName()};
  AttrUtils::SetListStr(op_desc, "_thread_kernelname", thread_kernel_names);
  EXPECT_EQ(kernel_handle.RegisterAutoThreadHandle(op_desc), INTERNAL_ERROR);
}

// test InitTbeHandleWithFfts
TEST_F(UtestTBEKernelHandle, init_thread_kernel_from_store) {
  TBEKernelHandle kernel_handle;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);

  vector<char> buffer;
  string key = op_desc->GetName();
  OpKernelBinPtr tbe_kernel_ptr0 = std::make_shared<ge::OpKernelBin>(key + "_0", std::move(buffer));
  OpKernelBinPtr tbe_kernel_ptr1 = std::make_shared<ge::OpKernelBin>(key + "_1", std::move(buffer));
  TBEKernelStore store;
  store.AddKernel(tbe_kernel_ptr0);
  store.AddKernel(tbe_kernel_ptr1);

  std::vector<OpKernelBinPtr> tbe_kernel{tbe_kernel_ptr0, tbe_kernel_ptr1};

  vector<string> bin_file_keys;
  bin_file_keys.emplace_back(op_desc->GetName() + "_0");
  bin_file_keys.emplace_back(op_desc->GetName() + "_1");
  AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  vector<string> json_list;
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetListStr(op_desc, "_thread_" + TVM_ATTR_NAME_MAGIC, json_list);
  vector<string> meta_data_list;
  meta_data_list.emplace_back("meta_data_0");
  meta_data_list.emplace_back("meta_data_1");
  AttrUtils::SetListStr(op_desc, "_thread_" +TVM_ATTR_NAME_METADATA, meta_data_list);
  vector<string> thread_kernel_names = {tbe_kernel_ptr0->GetName(), tbe_kernel_ptr1->GetName()};
  AttrUtils::SetListStr(op_desc, "_thread_kernelname", thread_kernel_names);
  EXPECT_EQ(kernel_handle.RegisterAutoThreadHandle(op_desc, store), SUCCESS);
  // rtQueryFunctionRegistered(bin_file_key) failed
  EXPECT_EQ(kernel_handle.used_tbe_handle_map_.size(), 2);
}

// test InitBinaryMagic
TEST_F(UtestTBEKernelHandle, init_binary_magic) {
  TBEKernelHandle kernel_handle;
  rtDevBinary_t binary;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  vector<string> json_list;
  AttrUtils::SetListStr(op_desc, TVM_ATTR_NAME_THREAD_MAGIC, json_list);
  // without tvm_magic
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, 0, binary, ""), INTERNAL_ERROR);
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF_AICPU");
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF");
  op_desc->DelAttr(TVM_ATTR_NAME_MAGIC);
  AttrUtils::SetListStr(op_desc, TVM_ATTR_NAME_MAGIC, json_list);
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, 0, binary, ""), SUCCESS);
  EXPECT_EQ(binary.magic, RT_DEV_BINARY_MAGIC_ELF_AICPU);
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, 1, binary, ""), SUCCESS);
  EXPECT_EQ(binary.magic, RT_DEV_BINARY_MAGIC_ELF);

  json_list.clear();
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  op_desc->DelAttr(TVM_ATTR_NAME_MAGIC);
  AttrUtils::SetListStr(op_desc, TVM_ATTR_NAME_MAGIC, json_list);
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, 0, binary, ""), SUCCESS);
  EXPECT_EQ(binary.magic, RT_DEV_BINARY_MAGIC_ELF_AIVEC);
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, 1, binary, ""), SUCCESS);
  EXPECT_EQ(binary.magic, RT_DEV_BINARY_MAGIC_ELF_AICUBE);

  // with invalid json type
  json_list.clear();
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF_INVALID");
  json_list.emplace_back("RT_DEV_BINARY_MAGIC_ELF_INVALID");
  op_desc->DelAttr(TVM_ATTR_NAME_MAGIC);
  AttrUtils::SetListStr(op_desc, TVM_ATTR_NAME_MAGIC, json_list);
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, 0, binary, ""), PARAM_INVALID);

  // test unffts
  string json_string = "RT_DEV_BINARY_MAGIC_ELF_AIVEC";
  op_desc->DelAttr(TVM_ATTR_NAME_MAGIC);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, json_string);
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, UINT32_MAX, binary, ""), SUCCESS);
  EXPECT_EQ(binary.magic, RT_DEV_BINARY_MAGIC_ELF_AIVEC);
}

// test InitMetaData
TEST_F(UtestTBEKernelHandle, init_meta_data) {
  TBEKernelHandle kernel_handle;
  void *bin_handle = nullptr;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  vector<string> meta_data_list;
  // with empty meta_data
  EXPECT_EQ(kernel_handle.InitMetaData(op_desc, 0, bin_handle, ""), INTERNAL_ERROR);
  meta_data_list.emplace_back("meta_data_0");
  meta_data_list.emplace_back("meta_data_1");
  AttrUtils::SetListStr(op_desc, TVM_ATTR_NAME_METADATA, meta_data_list);
  EXPECT_EQ(kernel_handle.InitMetaData(op_desc, 0, bin_handle, ""), SUCCESS);

  string meta_data = "meta_data";
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_METADATA, meta_data);
  EXPECT_EQ(kernel_handle.InitMetaData(op_desc, UINT32_MAX, bin_handle, ""), SUCCESS);
}

// test InitKernelName
TEST_F(UtestTBEKernelHandle, init_kernel_name) {
  TBEKernelHandle kernel_handle;
  string kernel_name;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  // failed when name is invalid
  EXPECT_EQ(kernel_handle.InitKernelName(op_desc, 0, kernel_name, ""), INTERNAL_ERROR);
  OpDescPtr op_desc1 = CreateOpDesc("sgt_graph_nodes/loss_scale", SCALE);
  string attr_kernel_name = "_thread_kernelname";
  vector<string> kernel_name_list;
  AttrUtils::SetListStr(op_desc, attr_kernel_name, kernel_name_list);
  // failed without kernel_name
  EXPECT_EQ(kernel_handle.InitKernelName(op_desc, 0, kernel_name, ""), INTERNAL_ERROR);
  kernel_name_list.emplace_back("kernel_name_0");
  kernel_name_list.emplace_back("kernel_name_1");
  AttrUtils::SetListStr(op_desc1, attr_kernel_name, kernel_name_list);
  EXPECT_EQ(kernel_handle.InitKernelName(op_desc1, 0, kernel_name, ""), SUCCESS);

  // without ffts
  auto pos = op_desc->GetName().find("/");
  attr_kernel_name = op_desc->GetName().substr(pos + 1) + "_thread_kernelname";
  kernel_name = "kernel_name";
  AttrUtils::SetStr(op_desc, attr_kernel_name, kernel_name);
  EXPECT_EQ(kernel_handle.InitKernelName(op_desc, UINT32_MAX, kernel_name, ""), SUCCESS);
}

TEST_F(UtestTBEKernelHandle, init_kernel_name_prefix_and_binary_magic) {
  TBEKernelHandle kernel_handle;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  // failed when name is invalid
  string prefix("test");
  string kernel_name_stub("kernel_stub");
  string attr_kernel_name = prefix + op_desc->GetName() + "_kernelname";
  (void)AttrUtils::SetStr(op_desc, attr_kernel_name, kernel_name_stub);

  string kernel_name;
  EXPECT_EQ(kernel_handle.InitKernelName(op_desc, UINT32_MAX, kernel_name, prefix), SUCCESS);
  EXPECT_EQ(kernel_name, kernel_name_stub);

  rtDevBinary_t binary;
  // without tvm_magic
  EXPECT_EQ(kernel_handle.InitBinaryMagic(op_desc, UINT32_MAX, binary, "_mix_aic"), SUCCESS);
}

TEST_F(UtestTBEKernelHandle, GetAddrAndPrefCnt_Fail_StaticShape_Binary) {
  TBEKernelHandle kernel_handle;
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aic_kernel");
  TBEHandleStore::GetInstance().StoreTBEHandle("mix_aic_kernel", nullptr, nullptr);
  (void)ge::AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "_te_kernel");
  std::vector<std::string> name_prefix = {"_mix_aic"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  std::string kernel_handle_name = kernel_handle.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, nullptr, nullptr);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 3;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  std::vector<std::pair<void *, uint32_t>> addr_and_pref_cnt;
  EXPECT_EQ(kernel_handle.GetAddrAndPrefCnt(op_desc, "_te_kernel_mix_aic", "_mix_aic", addr_and_pref_cnt),
            INTERNAL_ERROR);
  EXPECT_EQ(addr_and_pref_cnt.empty(), true);
}

TEST_F(UtestTBEKernelHandle, GetAddrAndPrefCnt_allkernel) {
  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  std::vector<std::string> names_prefix;
  names_prefix.emplace_back(string("aaa"));
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  auto tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  tiling_info->SetTilingKey(666U);
  op_desc->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);

  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
  std::string tbe_name1("tbe_kernel_key1");
  void *tbe_handle1 = (void *)0x12345678;
  std::shared_ptr<OpKernelBin> tbe_kernel = std::shared_ptr<OpKernelBin>();
  kernel_store.StoreTBEHandle(tbe_name1, tbe_handle1, tbe_kernel);

  TBEKernelHandle kernel_handle;
  std::vector<std::pair<void *, uint32_t>> addr_and_pref_cnt;
  EXPECT_EQ(kernel_handle.GetAddrAndPrefCnt(op_desc, tbe_name1, "aaa", addr_and_pref_cnt), SUCCESS);
}
}  // namespace ge
