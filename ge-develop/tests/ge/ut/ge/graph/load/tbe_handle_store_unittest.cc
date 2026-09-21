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
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "runtime/kernel.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestTBEHandleStore : public testing::Test {
 protected:
  void SetUp() {
    TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
    kernel_store.bin_key_to_handle_.clear();
    kernel_store.handle_to_kernel_to_unique_id_.clear();
  }

  void TearDown() {
    TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
    kernel_store.bin_key_to_handle_.clear();
    kernel_store.handle_to_kernel_to_unique_id_.clear();
  }
};

TEST_F(UtestTBEHandleStore, test_store_tbe_handle) {
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
  kernel_store.bin_key_to_handle_.clear();

  // not in store, can`t find.
  void *handle = nullptr;
  std::string tbe_name0("tbe_kernel_key0");
  EXPECT_FALSE(kernel_store.FindTBEHandle(tbe_name0, handle));
  EXPECT_EQ(handle, nullptr);

  // store first, size is 1, num is 1.
  std::string tbe_name1("tbe_kernel_key1");
  void *tbe_handle1 = (void *)0x12345678;
  std::shared_ptr<OpKernelBin> tbe_kernel = std::shared_ptr<OpKernelBin>();
  kernel_store.StoreTBEHandle(tbe_name1, tbe_handle1, tbe_kernel);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 1);

  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name1, handle));
  EXPECT_EQ(handle, tbe_handle1);

  auto it = kernel_store.bin_key_to_handle_.find(tbe_name1);
  EXPECT_NE(it, kernel_store.bin_key_to_handle_.end());
  auto &info1 = it->second;
  EXPECT_EQ(info1->handle(), tbe_handle1);
  EXPECT_EQ(info1->used_num(), 1);

  // store second, size is 1, num is 2.
  kernel_store.StoreTBEHandle(tbe_name1, tbe_handle1, tbe_kernel);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 1);

  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name1, handle));
  EXPECT_EQ(handle, tbe_handle1);

  it = kernel_store.bin_key_to_handle_.find(tbe_name1);
  EXPECT_NE(it, kernel_store.bin_key_to_handle_.end());
  auto &info2 = it->second;
  EXPECT_EQ(info2->handle(), tbe_handle1);
  EXPECT_EQ(info2->used_num(), 2);

  // store other, size is 2, num is 2, num is 1.
  std::string tbe_name2("tbe_kernel_key2");
  void *tbe_handle2 = (void *)0x22345678;
  kernel_store.StoreTBEHandle(tbe_name2, tbe_handle2, tbe_kernel);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 2);

  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name2, handle));
  EXPECT_EQ(handle, tbe_handle2);
  EXPECT_TRUE(kernel_store.FindTBEHandle(tbe_name1, handle));
  EXPECT_EQ(handle, tbe_handle1);

  it = kernel_store.bin_key_to_handle_.find(tbe_name1);
  EXPECT_NE(it, kernel_store.bin_key_to_handle_.end());
  auto &info3 = it->second;
  EXPECT_EQ(info3->handle(), tbe_handle1);
  EXPECT_EQ(info3->used_num(), 2);

  it = kernel_store.bin_key_to_handle_.find(tbe_name2);
  EXPECT_NE(it, kernel_store.bin_key_to_handle_.end());
  auto &info4 = it->second;
  EXPECT_EQ(info4->handle(), tbe_handle2);
  EXPECT_EQ(info4->used_num(), 1);

  // For Refer
  kernel_store.ReferTBEHandle(tbe_name0);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 2);

  kernel_store.ReferTBEHandle(tbe_name1);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 2);

  // For Erase.
  std::map<std::string, uint32_t> names0 = {{tbe_name0, 1}};
  kernel_store.EraseTBEHandle(names0);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 2);

  std::map<std::string, uint32_t> names1 = {{tbe_name1, 1}};
  kernel_store.EraseTBEHandle(names1);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 2);

  std::map<std::string, uint32_t> names2 = {{tbe_name1, 2}, {tbe_name2, 1}};
  kernel_store.EraseTBEHandle(names2);
  EXPECT_EQ(kernel_store.bin_key_to_handle_.size(), 0);
}

TEST_F(UtestTBEHandleStore, test_tbe_handle_info) {
  void *tbe_handle = (void *)0x12345678;
  TbeHandleInfo info(tbe_handle, nullptr);
  EXPECT_EQ(info.used_num(), 0);

  info.used_dec();
  EXPECT_EQ(info.used_num(), 0);

  info.used_inc(std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(info.used_num(), std::numeric_limits<uint32_t>::max());

  info.used_inc();
  EXPECT_EQ(info.used_num(), std::numeric_limits<uint32_t>::max());
}

TEST_F(UtestTBEHandleStore, test_get_unique_id_ptr) {
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
  kernel_store.handle_to_kernel_to_unique_id_.clear();
  void *handle1 = (void *)0x12345678;
  void *handle2 = (void *)0x123456;
  std::string kernel1 = "k1";
  std::string kernel2 = "k2";
  bool inserted = false;
  void *id1 = kernel_store.GetUniqueIdPtr(handle1, kernel1, inserted);
  EXPECT_EQ(inserted, true);

  void *id2 = kernel_store.GetUniqueIdPtr(handle1, kernel1, inserted);
  EXPECT_EQ(inserted, false);
  EXPECT_EQ(id1, id2);

  void *id3 = kernel_store.GetUniqueIdPtr(handle1, kernel2, inserted);
  EXPECT_EQ(inserted, true);
  EXPECT_NE(id3, id1);

  void *id4 = kernel_store.GetUniqueIdPtr(handle2, kernel2, inserted);
  EXPECT_EQ(inserted, true);
  EXPECT_NE(id4, id3);

  void *id5 = kernel_store.GetUniqueIdPtr(handle2, kernel2, inserted);
  EXPECT_EQ(inserted, false);
  EXPECT_EQ(id5, id4);
}
}  // namespace ge
