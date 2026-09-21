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
#include "graph/cache_policy/match_policy_for_exactly_the_same.h"
#include "cache_desc_stub/runtime_cache_desc.h"
#include "graph/cache_policy/cache_state.h"

namespace ge {
namespace {
CacheDescPtr CreateRuntimeCacheDesc(const std::vector<gert::Shape> &shapes) {
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes);
  return cache_desc;
}
CacheInfo CreateCacheInfo(const uint64_t time_count, const CacheItemId item_id,
                          const std::vector<gert::Shape> &shapes) {
  auto cache_desc = CreateRuntimeCacheDesc(shapes);
  CacheInfo cache_info{time_count, item_id, cache_desc};
  return cache_info;
}
}
class MatchPolicyForExactlyTheSameUT : public testing::Test {};

TEST_F(MatchPolicyForExactlyTheSameUT, GetCacheItemId_KInvalidCacheItemId_CannotMatchHashKey) {
  gert::Shape s1{256, 256};
  gert::Shape s2{1, 256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  const std::vector<gert::Shape> shapes2{s2};

  CCStatType hash_2_cache_infos;
  auto cache_info = CreateCacheInfo(1, 1, shapes1);
  auto hash = CreateRuntimeCacheDesc(shapes1)->GetCacheDescHash();
  hash_2_cache_infos[hash] = {cache_info};

  auto find_cache_desc = CreateRuntimeCacheDesc(shapes2);
  MatchPolicyForExactlyTheSame mp;
  auto find_id = mp.GetCacheItemId(hash_2_cache_infos, find_cache_desc);
  EXPECT_EQ(find_id, KInvalidCacheItemId);
}

TEST_F(MatchPolicyForExactlyTheSameUT, GetCacheItemId_KInvalidCacheItemId_CannotMatchShapes) {
  gert::Shape s1{256, 256};
  gert::Shape s2{1, 256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  const std::vector<gert::Shape> shapes2{s2};

  CCStatType hash_2_cache_infos;
  auto cache_info = CreateCacheInfo(1, 1, shapes1);
  auto hash = CreateRuntimeCacheDesc(shapes2)->GetCacheDescHash();
  hash_2_cache_infos[hash] = {cache_info};

  auto find_cache_desc = CreateRuntimeCacheDesc(shapes2);
  MatchPolicyForExactlyTheSame mp;
  auto find_id = mp.GetCacheItemId(hash_2_cache_infos, find_cache_desc);
  EXPECT_EQ(find_id, KInvalidCacheItemId);
}

TEST_F(MatchPolicyForExactlyTheSameUT, GetCacheItemId_ShapesAndHashMatched) {
  gert::Shape s1{256, 256};
  const std::vector<gert::Shape> shapes1{s1};
  const std::vector<gert::Shape> shapes2{s1};

  CCStatType hash_2_cache_infos;
  auto cache_info = CreateCacheInfo(1, 1, shapes1);
  auto hash = CreateRuntimeCacheDesc(shapes1)->GetCacheDescHash();
  hash_2_cache_infos[hash] = {cache_info};

  auto find_cache_desc = CreateRuntimeCacheDesc(shapes1);
  MatchPolicyForExactlyTheSame mp;
  auto find_id = mp.GetCacheItemId(hash_2_cache_infos, find_cache_desc);
  EXPECT_EQ(find_id, cache_info.GetItemId());
}

TEST_F(MatchPolicyForExactlyTheSameUT, GetCacheItemId_LogHashNotExist_KeyExistButVectorEmpty) {
  dlog_setlevel(0, 0, 0);
  gert::Shape s1{256, 256};
  const std::vector<gert::Shape> shapes1{s1};

  CCStatType hash_2_cache_infos;
  auto cache_info = CreateCacheInfo(1, 1, shapes1);
  auto find_cache_desc = CreateRuntimeCacheDesc(shapes1);
  auto hash = find_cache_desc->GetCacheDescHash();
  hash_2_cache_infos[hash] = {cache_info};
  hash_2_cache_infos[hash].erase(hash_2_cache_infos[hash].begin());  // key exist but vector value empty

  MatchPolicyForExactlyTheSame mp;
  auto find_id = mp.GetCacheItemId(hash_2_cache_infos, find_cache_desc);
  EXPECT_EQ(find_id, KInvalidCacheItemId);
  dlog_setlevel(0, 3, 0);
}

}  // namespace ge
