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
#include "hcom/hcom_topo_info.h"
namespace ge {
class UtestHcomTopoInfo : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(UtestHcomTopoInfo, SetGroupTopoInfo) {
  HcomTopoInfo::TopoInfo topo_info;
  topo_info.rank_size = 8;
  const std::string group = "group0";

  // add invalid
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo(nullptr, topo_info), GRAPH_FAILED);

  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topo_info), GRAPH_SUCCESS);
  // add repeated, over write
  topo_info.notify_handle = reinterpret_cast<void *>(0x8000);
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topo_info), GRAPH_SUCCESS);
  void *handle = nullptr;
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupNotifyHandle(group.c_str(), handle), GRAPH_SUCCESS);
  EXPECT_EQ(handle, reinterpret_cast<void *>(0x8000));
  HcomTopoInfo::TopoInfo topo_info_existed;
  EXPECT_TRUE(HcomTopoInfo::Instance().TryGetGroupTopoInfo(group.c_str(), topo_info_existed));
  EXPECT_TRUE(HcomTopoInfo::Instance().TopoInfoHasBeenSet(group.c_str()));
  EXPECT_EQ(topo_info_existed.notify_handle, reinterpret_cast<void *>(0x8000));
  EXPECT_EQ(topo_info_existed.rank_size, 8);
}

TEST_F(UtestHcomTopoInfo, GetAndUnsetGroupTopoInfo) {
  int64_t rank_size = -1;
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupRankSize("group0", rank_size), GRAPH_SUCCESS);
  EXPECT_EQ(rank_size, 8);
  // not added, get failed
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupRankSize("group1", rank_size), GRAPH_FAILED);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupTopoDesc("group1"), nullptr);
  void *handle = nullptr;
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupNotifyHandle("group1", handle), GRAPH_FAILED);
  HcomTopoInfo::Instance().UnsetGroupTopoInfo("group1");
  HcomTopoInfo::Instance().UnsetGroupTopoInfo("group0");
  // removed
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupRankSize("group0", rank_size), GRAPH_FAILED);

  // construct topo info
  HcomTopoInfo::TopoInfo topo_info;
  HcomTopoInfo::TopoLevelDesc t0 = {2, 2};
  HcomTopoInfo::TopoLevelDesc t1 = {6, 6};
  EXPECT_EQ(sizeof(topo_info.topo_level_descs) / sizeof(HcomTopoInfo::TopoLevelDesc),
            static_cast<int64_t>(HcomTopoInfo::TopoLevel::MAX));
  topo_info.rank_size = 16;
  topo_info.topo_level_descs[0] = t0;
  topo_info.topo_level_descs[1] = t1;
  // add again
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo("group0", topo_info), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupRankSize("group0", rank_size), GRAPH_SUCCESS);
  EXPECT_EQ(rank_size, 16);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupRankSize("", rank_size), GRAPH_FAILED);
  // check
  auto topo_desc = HcomTopoInfo::Instance().GetGroupTopoDesc("group0");
  EXPECT_NE(topo_desc, nullptr);
  EXPECT_EQ(sizeof(*topo_desc) / sizeof(HcomTopoInfo::TopoLevelDesc),
            static_cast<int64_t>(HcomTopoInfo::TopoLevel::MAX));

  HcomTopoInfo::TopoDescs topo_desc_to_check = {t0, t1};
  EXPECT_EQ((*topo_desc)[0].comm_sets, topo_desc_to_check[0].comm_sets);
  EXPECT_EQ((*topo_desc)[0].rank_size, topo_desc_to_check[0].rank_size);
  EXPECT_EQ((*topo_desc)[1].comm_sets, topo_desc_to_check[1].comm_sets);
  EXPECT_EQ((*topo_desc)[1].rank_size, topo_desc_to_check[1].rank_size);
}

TEST_F(UtestHcomTopoInfo, SetAndGetAndUnsetGroupOrderedStreamWithDeviceId) {
  const std::string group0 = "group0";
  const std::string group1 = "group1";
  const std::string group = "group";
  void *stream0= (void *)1;
  void *stream1= (void *)2;
  void *stream = nullptr;
  int32_t device0 = 0;
  int32_t device1 = 1;

  // set group nullptr
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(device0, nullptr, stream0), GRAPH_FAILED);

  // set and get
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(device0, group0.c_str(), stream0), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group0.c_str(), stream), GRAPH_SUCCESS);
  EXPECT_EQ(stream, stream0);

  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(device0, group1.c_str(), stream1), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group1.c_str(), stream), GRAPH_SUCCESS);
  EXPECT_EQ(stream, stream1);

  // override
  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(device0, group0.c_str(), stream1), GRAPH_SUCCESS);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group0.c_str(), stream), GRAPH_SUCCESS);
  EXPECT_EQ(stream, stream1);

  // get no device
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device1, group.c_str(), stream), GRAPH_FAILED);

  // get no group
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group.c_str(), stream), GRAPH_FAILED);

  // unset group
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(device0, group0.c_str());
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group0.c_str(), stream), GRAPH_FAILED);
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(device0, group1.c_str());
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group1.c_str(), stream), GRAPH_FAILED);

  // repeat unset
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(device0, group1.c_str());
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupOrderedStream(device0, group1.c_str(), stream), GRAPH_FAILED);
}

TEST_F(UtestHcomTopoInfo, GetGroupLocalWindowSize) {
  HcomTopoInfo::TopoInfo topo_info;
  topo_info.rank_size = 8;
  topo_info.local_window_size = 209715200;
  const std::string group = "group0";

  EXPECT_EQ(HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topo_info), GRAPH_SUCCESS);
  uint64_t local_window_size = 0;
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupLocalWindowSize("test", local_window_size), GRAPH_FAILED);
  EXPECT_EQ(HcomTopoInfo::Instance().GetGroupLocalWindowSize(group.c_str(), local_window_size), GRAPH_SUCCESS);
  EXPECT_EQ(local_window_size, static_cast<uint64_t>(209715200));
}

}
