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
#include "graph/manager/host_mem_manager.h"
#include "framework/memory/memory_api.h"
#include "graph_metadef/graph/aligned_ptr.h"

using namespace std;
using namespace testing;
using namespace ge;

class STestMemoryApiTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(STestMemoryApiTest, MallocSharedMemory) {
  std::vector<int64_t> dims = {1};
  TensorInfo tensor_info = {"test", dims, DT_INT32};
  uint64_t dev_addr;
  uint64_t memory_size;
  Status ret = MallocSharedMemory(tensor_info, dev_addr, memory_size);
  EXPECT_EQ(ret, SUCCESS);
  // 此处LSan会误报，需要手动释放，实际上程序结束时单例析构会调用Finalize
  HostMemManager::Instance().Finalize();
}

TEST_F(STestMemoryApiTest, QueryMemInfoSuccess) {
  const char *var_name = "host_params";
  SharedMemInfo info;
  uint8_t tmp(0);
  info.device_address = &tmp;

  std::shared_ptr<AlignedPtr> aligned_ptr = std::make_shared<AlignedPtr>(100, 16);

  info.host_aligned_ptr = aligned_ptr;
  info.fd = 0;
  info.mem_size = 100;
  info.op_name = var_name;
  HostMemManager::Instance().var_memory_base_map_[var_name] = info;
  uint64_t base_addr;
  uint64_t var_size;
  Status ret = GetVarBaseAddrAndSize(var_name, base_addr, var_size);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(var_size, 100);
  HostMemManager::Instance().var_memory_base_map_.clear();
}
