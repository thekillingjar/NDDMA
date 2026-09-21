/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#include "reader_writer/data_aligner.h"
#include "common/scope_guard.h"
#include "flow_func/mbuf_flow_msg.h"
#include "mockcpp/mockcpp.hpp"
#include "common/inner_error_codes.h"

namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
  RuntimeTensorDesc tensor_desc;
  uint8_t data[1024];
};

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufFreeStub(Mbuf *mbuf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}

int halMbufGetPrivInfoStub(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}
}  // namespace
class DataAlignerUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(DataAlignerUTest, PushAndAlignData_param_check) {
  size_t queue_num = 5;
  DataAligner data_aligner(queue_num, 100, 30 * 1000, false);
  std::vector<Mbuf *> data_list;
  EXPECT_NE(data_aligner.PushAndAlignData(0, nullptr, data_list), HICAID_SUCCESS);
  MbufImpl mbuf_impl{};
  Mbuf *mbuf = reinterpret_cast<Mbuf *>(&mbuf_impl);
  EXPECT_NE(data_aligner.PushAndAlignData(queue_num, mbuf, data_list), HICAID_SUCCESS);
}

TEST_F(DataAlignerUTest, aligned_already) {
  constexpr size_t queue_num = 4;
  std::vector<Mbuf *> data_list;
  data_list.reserve(queue_num);
  ScopeGuard guard([&data_list]() {
    for (auto *&mbuf : data_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
  });

  DataAligner data_aligner(queue_num, 100, 30 * 1000, false);
  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf);
    MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
    mbuf_impl->head_msg.trans_id = 1;
    mbuf_impl->head_msg.data_label = 1;

    if (i < queue_num - 1) {
      EXPECT_EQ(data_aligner.PushAndAlignData(i, mbuf, data_list), HICAID_SUCCESS);
      EXPECT_TRUE(data_list.empty());
      EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 1);
    } else {
      EXPECT_EQ(data_aligner.PushAndAlignData(i, mbuf, data_list), HICAID_SUCCESS);
      EXPECT_FALSE(data_list.empty());
      EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 0);
    }
  }
}

TEST_F(DataAlignerUTest, not_aligned) {
  constexpr size_t queue_num = 4;
  std::vector<Mbuf *> data_list;
  data_list.reserve(queue_num);
  ScopeGuard guard([&data_list]() {
    for (auto *&mbuf : data_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
  });

  DataAligner data_aligner(queue_num, 100, 30 * 1000, false);
  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf);
    MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
    mbuf_impl->head_msg.trans_id = i;
    mbuf_impl->head_msg.data_label = 1;

    EXPECT_EQ(data_aligner.PushAndAlignData(i, mbuf, data_list), HICAID_SUCCESS);
    EXPECT_TRUE(data_list.empty());
    EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), i + 1);
  }
}

TEST_F(DataAlignerUTest, cache_aligned) {
  constexpr size_t queue_num = 3;
  std::vector<Mbuf *> data_list;
  ScopeGuard guard([&data_list]() {
    for (auto *&mbuf : data_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
  });

  DataAligner data_aligner(queue_num, 100, 30 * 1000, false);
  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf);
    MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
    mbuf_impl->head_msg.trans_id = i;
    mbuf_impl->head_msg.data_label = i + 1;
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
    EXPECT_TRUE(tmp_data_list.empty());
  }

  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf2 = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf2);
    MbufImpl *mbuf_impl2 = (MbufImpl *)mbuf2;
    mbuf_impl2->head_msg.trans_id = (i + 1) % queue_num;
    mbuf_impl2->head_msg.data_label = (i + 1) % queue_num + 1;
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(i, mbuf2, tmp_data_list), HICAID_SUCCESS);
    data_list.insert(data_list.end(), tmp_data_list.begin(), tmp_data_list.end());
  }

  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf3 = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf3);
    MbufImpl *mbuf_impl3 = (MbufImpl *)mbuf3;
    mbuf_impl3->head_msg.trans_id = (i + 2) % queue_num;
    mbuf_impl3->head_msg.data_label = (i + 2) % queue_num + 1;
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(i, mbuf3, tmp_data_list), HICAID_SUCCESS);
    data_list.insert(data_list.end(), tmp_data_list.begin(), tmp_data_list.end());
  }

  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 0);
  EXPECT_EQ(data_list.size(), 3 * queue_num);
}

TEST_F(DataAlignerUTest, align_repeat) {
  constexpr size_t queue_num = 3;
  std::vector<std::queue<Mbuf *>> queue_list;
  queue_list.resize(queue_num);
  std::vector<Mbuf *> aligned_list;
  ScopeGuard guard([&queue_list, &aligned_list]() {
    for (auto &que : queue_list) {
      while (!que.empty()) {
        halMbufFree(que.front());
        que.pop();
      }
    }
    for (auto *&mbuf : aligned_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
  });

  DataAligner data_aligner(queue_num, 100, 30 * 1000, false);
  /*
   * queue transid as follow
   *  q1  q2  q3
   *  1   0   0
   *  2   0   0
   *  3   3   3
   *  4   4   4
   *  0   1   1
   *  0   2   2
   */
  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < queue_num; ++j) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      if (j == 0) {
        mbuf_impl->head_msg.trans_id = i + 1;
      } else {
        mbuf_impl->head_msg.trans_id = 0;
      }
      mbuf_impl->head_msg.data_label = 0;
      queue_list[j].emplace(mbuf);
    }
  }

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < queue_num; ++j) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      mbuf_impl->head_msg.trans_id = 3 + i;
      mbuf_impl->head_msg.data_label = 0;
      queue_list[j].emplace(mbuf);
    }
  }

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < queue_num; ++j) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      if (j != 0) {
        mbuf_impl->head_msg.trans_id = i + 1;
      } else {
        mbuf_impl->head_msg.trans_id = 0;
      }
      mbuf_impl->head_msg.data_label = 0;
      queue_list[j].emplace(mbuf);
    }
  }

  size_t max_wait_align_num = 0;
  while (true) {
    size_t idx = data_aligner.SelectNextIndex();
    auto &que = queue_list[idx];
    if (que.empty()) {
      break;
    }
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(idx, que.front(), tmp_data_list), HICAID_SUCCESS);
    size_t wait_num = data_aligner.GetWaitAlignDataSize();
    if (wait_num > max_wait_align_num) {
      max_wait_align_num = wait_num;
    }
    aligned_list.insert(aligned_list.cend(), tmp_data_list.begin(), tmp_data_list.end());
    que.pop();
  }
  EXPECT_EQ(aligned_list.size(), 6 * queue_num);
  // wait align trans_id is 0 1 2 3 or 0 1 2 4.
  EXPECT_EQ(max_wait_align_num, 4);
  for (auto &que : queue_list) {
    EXPECT_TRUE(que.empty());
  }
}

TEST_F(DataAlignerUTest, align_over_limit_no_drop) {
  constexpr size_t queue_num = 2;
  DataAligner data_aligner(queue_num, 256, 30 * 1000, false);
  for (int count = 0; count < 128; ++count) {
    for (int32_t i = 0; i < queue_num; ++i) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      mbuf_impl->head_msg.trans_id = count;
      mbuf_impl->head_msg.data_label = i + 1;

      std::vector<Mbuf *> tmp_data_list;
      ASSERT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
      ASSERT_TRUE(tmp_data_list.empty());
    }
  }

  ASSERT_EQ(data_aligner.GetWaitAlignDataSize(), 256);
  std::vector<Mbuf *> over_limit_data_list;
  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  ASSERT_TRUE(over_limit_data_list.empty());

  auto release_func = [](std::vector<Mbuf *> &data_list) {
    for (auto *&mbuf : data_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
    data_list.clear();
  };
  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf);
    MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
    mbuf_impl->head_msg.trans_id = 128;
    mbuf_impl->head_msg.data_label = i + 1;
    std::vector<Mbuf *> tmp_data_list;
    ASSERT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
    ASSERT_TRUE(tmp_data_list.empty());
  }

  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 256 + 2);

  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  ASSERT_FALSE(over_limit_data_list.empty());
  release_func(over_limit_data_list);

  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  ASSERT_FALSE(over_limit_data_list.empty());
  release_func(over_limit_data_list);

  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  ASSERT_TRUE(over_limit_data_list.empty());
  release_func(over_limit_data_list);
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 256);
}

TEST_F(DataAlignerUTest, align_over_limit_drop) {
  constexpr size_t queue_num = 2;
  DataAligner data_aligner(queue_num, 10, 30 * 1000, true);
  for (int count = 0; count < 5; ++count) {
    for (int32_t i = 0; i < queue_num; ++i) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      mbuf_impl->head_msg.trans_id = count;
      mbuf_impl->head_msg.data_label = i + 1;

      std::vector<Mbuf *> tmp_data_list;
      ASSERT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
      ASSERT_TRUE(tmp_data_list.empty());
    }
  }

  for (int32_t i = 0; i < queue_num; ++i) {
    Mbuf *mbuf = nullptr;
    halMbufAllocExStub(1, 0, 0, 0, &mbuf);
    MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
    mbuf_impl->head_msg.trans_id = 128;
    mbuf_impl->head_msg.data_label = i + 1;
    std::vector<Mbuf *> tmp_data_list;
    ASSERT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
    ASSERT_TRUE(tmp_data_list.empty());
  }
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 10 + 2);
  std::vector<Mbuf *> over_limit_data_list;
  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  ASSERT_TRUE(over_limit_data_list.empty());
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 10);
}

TEST_F(DataAlignerUTest, align_timeout_drop) {
  constexpr size_t queue_num = 2;
  DataAligner data_aligner(queue_num, 100, 1000, true);
  for (int count = 0; count < 5; ++count) {
    for (int32_t i = 0; i < queue_num; ++i) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      mbuf_impl->head_msg.trans_id = count;
      mbuf_impl->head_msg.data_label = i + 1;

      std::vector<Mbuf *> tmp_data_list;
      ASSERT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
      ASSERT_TRUE(tmp_data_list.empty());
    }
  }

  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 10);
  std::vector<Mbuf *> over_limit_data_list;
  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  EXPECT_TRUE(over_limit_data_list.empty());

  usleep(1200 * 1000UL);

  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  EXPECT_TRUE(over_limit_data_list.empty());
  // over timeout drop out
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 0);
}

TEST_F(DataAlignerUTest, align_timeout_no_drop) {
  constexpr size_t queue_num = 2;
  DataAligner data_aligner(queue_num, 100, 1000, false);
  for (int count = 0; count < 5; ++count) {
    for (int32_t repeat = 0; repeat < 2; ++repeat) {
      for (int32_t i = 0; i < queue_num; ++i) {
        Mbuf *mbuf = nullptr;
        halMbufAllocExStub(1, 0, 0, 0, &mbuf);
        MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
        mbuf_impl->head_msg.trans_id = count;
        mbuf_impl->head_msg.data_label = i + 1;

        std::vector<Mbuf *> tmp_data_list;
        ASSERT_EQ(data_aligner.PushAndAlignData(i, mbuf, tmp_data_list), HICAID_SUCCESS);
        ASSERT_TRUE(tmp_data_list.empty());
      }
    }
  }

  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 10);
  std::vector<Mbuf *> over_limit_data_list;
  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  EXPECT_TRUE(over_limit_data_list.empty());

  auto release_func = [](std::vector<Mbuf *> &data_list) {
    for (auto *&mbuf : data_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
    data_list.clear();
  };

  usleep(1200 * 1000UL);

  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  EXPECT_FALSE(over_limit_data_list.empty());
  release_func(over_limit_data_list);
  // over timeout drop out, but not take over
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 10);

  data_aligner.TryTakeExpiredOrOverLimitData(over_limit_data_list);
  EXPECT_FALSE(over_limit_data_list.empty());
  release_func(over_limit_data_list);
  // over timeout drop out, and take over
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 9);
  data_aligner.Reset();
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 0);
  EXPECT_EQ(data_aligner.SelectNextIndex(), 0);
}

TEST_F(DataAlignerUTest, add_exception_trandId) {
  constexpr size_t queue_num = 3;
  std::vector<std::queue<Mbuf *>> queue_list;
  queue_list.resize(queue_num);
  std::vector<Mbuf *> aligned_list;
  ScopeGuard guard([&queue_list, &aligned_list]() {
    for (auto &que : queue_list) {
      while (!que.empty()) {
        halMbufFree(que.front());
        que.pop();
      }
    }
    for (auto *&mbuf : aligned_list) {
      if (mbuf != nullptr) {
        halMbufFree(mbuf);
        mbuf = nullptr;
      }
    }
  });

  DataAligner data_aligner(queue_num, 100, 30 * 1000, false);
  /*
   * queue transid as follow
   *  q1  q2  q3
   *  1   0   0
   *  2   0   0
   *  3   3   3
   *  4   4   4
   *  0   1   1
   *  0   2   2
   */
  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < queue_num; ++j) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      if (j == 0) {
        mbuf_impl->head_msg.trans_id = i + 1;
      } else {
        mbuf_impl->head_msg.trans_id = 0;
      }
      mbuf_impl->head_msg.data_label = 0;
      queue_list[j].emplace(mbuf);
    }
  }

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < queue_num; ++j) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      mbuf_impl->head_msg.trans_id = 3 + i;
      mbuf_impl->head_msg.data_label = 0;
      queue_list[j].emplace(mbuf);
    }
  }

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < queue_num; ++j) {
      Mbuf *mbuf = nullptr;
      halMbufAllocExStub(1, 0, 0, 0, &mbuf);
      MbufImpl *mbuf_impl = (MbufImpl *)mbuf;
      if (j != 0) {
        mbuf_impl->head_msg.trans_id = i + 1;
      } else {
        mbuf_impl->head_msg.trans_id = 0;
      }
      mbuf_impl->head_msg.data_label = 0;
      queue_list[j].emplace(mbuf);
    }
  }

  size_t max_wait_align_num = 0;
  /*
   * read data as follow
   *  q1  q2  q3
   *  1   0   0
   *  2   0   0
   */
  for (int32_t i = 0; i < 6; ++i) {
    size_t idx = data_aligner.SelectNextIndex();
    auto &que = queue_list[idx];
    if (que.empty()) {
      break;
    }
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(idx, que.front(), tmp_data_list), HICAID_SUCCESS);
    size_t wait_num = data_aligner.GetWaitAlignDataSize();
    if (wait_num > max_wait_align_num) {
      max_wait_align_num = wait_num;
    }
    aligned_list.insert(aligned_list.cend(), tmp_data_list.begin(), tmp_data_list.end());
    que.pop();
  }
  EXPECT_TRUE(aligned_list.empty());
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 3);

  data_aligner.AddExceptionTransId(0);
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 2);
  EXPECT_EQ(data_aligner.SelectNextIndex(), 1);
  /*
   * now queue transid as follow
   *  q1  q2  q3
   *  3   3   3
   *  4   4   4
   *  0   1   1
   *  0   2   2
   */
  for (int32_t i = 0; i < 4; ++i) {
    size_t idx = data_aligner.SelectNextIndex();
    auto &que = queue_list[idx];
    if (que.empty()) {
      break;
    }
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(idx, que.front(), tmp_data_list), HICAID_SUCCESS);
    size_t wait_num = data_aligner.GetWaitAlignDataSize();
    if (wait_num > max_wait_align_num) {
      max_wait_align_num = wait_num;
    }
    aligned_list.insert(aligned_list.cend(), tmp_data_list.begin(), tmp_data_list.end());
    que.pop();
  }
  EXPECT_TRUE(aligned_list.empty());
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 4);
  EXPECT_EQ(data_aligner.SelectNextIndex(), 0);

  /*
   * now queue transid as follow
   *  q1  q2  q3
   *  3
   *  4
   *  0   1   1
   *  0   2   2
   */
  for (int32_t i = 0; i < 8; ++i) {
    size_t idx = data_aligner.SelectNextIndex();
    auto &que = queue_list[idx];
    if (que.empty()) {
      break;
    }
    std::vector<Mbuf *> tmp_data_list;
    EXPECT_EQ(data_aligner.PushAndAlignData(idx, que.front(), tmp_data_list), HICAID_SUCCESS);
    size_t wait_num = data_aligner.GetWaitAlignDataSize();
    if (wait_num > max_wait_align_num) {
      max_wait_align_num = wait_num;
    }
    aligned_list.insert(aligned_list.cend(), tmp_data_list.begin(), tmp_data_list.end());
    que.pop();
  }
  EXPECT_EQ(data_aligner.GetWaitAlignDataSize(), 0);
  EXPECT_EQ(aligned_list.size(), 4 * queue_num);
  // wait align trans_id is 1 2 3 4.
  EXPECT_EQ(max_wait_align_num, 4);
  for (auto &que : queue_list) {
    EXPECT_TRUE(que.empty());
  }
  data_aligner.DeleteExceptionTransId(0);
}
}  // namespace FlowFunc