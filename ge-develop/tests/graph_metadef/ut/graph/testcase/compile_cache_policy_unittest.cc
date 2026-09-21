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
#include <ctime>
#include "common/checker.h"
#include "exe_graph/runtime/shape.h"
#include "graph/cache_policy/cache_state.h"
#include "graph/cache_policy/match_policy_for_exactly_the_same.h"
#include "graph/cache_policy/aging_policy_lru_k.h"
#include "cache_desc_stub/runtime_cache_desc.h"
#include "graph/cache_policy/cache_policy.h"
#include "graph/cache_policy/aging_policy_lru.h"

namespace ge {
namespace {
std::vector<CacheItemId> AddCachesByDepth(std::unique_ptr<CachePolicy> &cp, uint16_t depth) {
  std::vector<CacheItemId> ids;
  for (uint16_t i = 0; i < depth; ++i) {
    int64_t dim_0 = i;
    gert::Shape s{dim_0, 256, 256};
    auto cache_desc = std::make_shared<RuntimeCacheDesc>();
    cache_desc->SetShapes({s});
    CacheItemId cache_id = cp->AddCache(cache_desc);

    if (cache_id != KInvalidCacheItemId) {
      GELOGE(ge::FAILED, "AddCachesByDepth falied.");
      return {};
    }
    cache_id = cp->AddCache(cache_desc);
    if (cache_id == KInvalidCacheItemId) {
      GELOGE(ge::FAILED, "AddCachesByDepth falied.");
      return {};
    }
    ids.emplace_back(cache_id);
  }
  return ids;
}

std::vector<CacheItemId> AddCachesByDepthForLRU(std::unique_ptr<CachePolicy> &cp, uint16_t depth) {
  std::vector<CacheItemId> ids;
  for (uint16_t i = 0; i < depth; ++i) {
    int64_t dim_0 = i;
    gert::Shape s{dim_0, 256, 256};
    auto cache_desc = std::make_shared<RuntimeCacheDesc>();
    cache_desc->SetShapes({s});
    auto cache_id = cp->AddCache(cache_desc);
    if (cache_id == KInvalidCacheItemId) {
      GELOGE(ge::FAILED, "AddCachesByDepth falied.");
      return {};
    }
    ids.emplace_back(cache_id);
  }
  return ids;
}
}  // namespace

class UtestCompileCachePolicy : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_1) {
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ASSERT_NE(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_2) {
  auto mp_ptr = PolicyRegister::GetInstance().GetMatchPolicy(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY);
  auto ap_ptr = PolicyRegister::GetInstance().GetAgingPolicy(ge::AgingPolicyType::AGING_POLICY_LRU);
  auto ccp = ge::CachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_NE(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPFailed_1) {
  auto mp_ptr = PolicyRegister::GetInstance().GetMatchPolicy(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY);
  auto ap_ptr = nullptr;
  auto ccp = ge::CachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_EQ(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPFailed_2) {
  auto mp_ptr = nullptr;
  auto ap_ptr = nullptr;
  auto ccp = ge::CachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_EQ(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, AddSameCache) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc->AddBinary(holder);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  CacheItemId cache_id_same = ccp->AddCache(cache_desc);
  ASSERT_EQ(cache_id_same, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentOptypeCache) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc->AddBinary(holder);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  cache_desc->SetOpType("another_op");
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentUniqueIdCache) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc->AddBinary(holder);
  cache_desc->SetScopeId({1, 2});
  ASSERT_EQ(cache_desc->scope_id_.size(), 2);
  ASSERT_EQ(cache_desc->scope_id_[0], 1);
  ASSERT_EQ(cache_desc->scope_id_[1], 2);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  CompileCacheDescPtr other_cache_desc = std::make_shared<CompileCacheDesc>(*cache_desc.get());
  other_cache_desc->SetScopeId({1, 3});
  ASSERT_EQ(other_cache_desc->scope_id_.size(), 2);
  ASSERT_EQ(other_cache_desc->scope_id_[0], 1);
  ASSERT_EQ(other_cache_desc->scope_id_[1], 3);
  CacheItemId cache_id_another = ccp->AddCache(other_cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentBinarySizeCache) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  CompileCacheDescPtr other_cache_desc = std::make_shared<CompileCacheDesc>(*cache_desc.get());
  other_cache_desc->AddBinary(holder);
  CacheItemId cache_id_another = ccp->AddCache(other_cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentBinaryValueCache) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc->AddBinary(holder);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  uint8_t value_another = 8;
  BinaryHolder holder1(&value_another, 1);
  CompileCacheDescPtr other_cache_desc = std::make_shared<CompileCacheDesc>(*cache_desc.get());
  other_cache_desc->other_desc_.clear();
  other_cache_desc->AddBinary(holder1);
  CacheItemId cache_id_another = ccp->AddCache(other_cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentTensorFormatCache) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc->AddBinary(holder);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  cache_desc->tensor_info_args_vec_[0].format_ = ge::FORMAT_NCHW;
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, CacheFindFailBecauseRangeFirst) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  CompileCacheDescPtr cache_desc_match = cache_desc;
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  SmallVector<int64_t, kDefaultDimsNum> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  ASSERT_EQ(tensor_info.shape_.size(), 2);
  ASSERT_EQ(tensor_info.shape_[0], -1);
  ASSERT_EQ(tensor_info.shape_[1], -1);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{0,5}; 
  tensor_info.SetShape(shape_match);
  cache_desc_match->AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id_find, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, CacheFindFailBecauseRangeSecond) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  CompileCacheDescPtr cache_desc_match = cache_desc;
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{5,11}; 
  tensor_info.SetShape(shape_match);
  cache_desc_match->AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id_find, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, CacheFindSuccessBecauseUnknownRank) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  CompileCacheDescPtr cache_desc_match = std::make_shared<CompileCacheDesc>(*cache_desc.get());
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-2}; 
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  cache_desc->AddTensorInfo(tensor_info);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{5,11}; 
  tensor_info.SetShape(shape_match);
  cache_desc_match->AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id_find, cache_id);
}


TEST_F(UtestCompileCachePolicy, CacheFindSuccessCommonTest) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  CompileCacheDescPtr cache_desc_match = std::make_shared<CompileCacheDesc>(*cache_desc.get());
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{5,5}; 
  tensor_info.SetShape(shape_match);
  cache_desc_match->AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id, cache_id_find);
  ge::CachePolicy ccp1;
  ASSERT_EQ(ccp1.FindCache(cache_desc_match), KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, CacheDelTest) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc->AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc->AddBinary(holder);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  CacheItemId cache_id_same = ccp->AddCache(cache_desc);
  ASSERT_EQ(cache_id_same, cache_id);

  cache_desc->SetOpType("another_op");
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);

  std::vector<CacheItemId> delete_item{cache_id, cache_id_another};
  std::vector<CacheItemId> delete_item_ret = ccp->DeleteCache(delete_item);
  ASSERT_EQ(delete_item_ret.size(), 2);
}

TEST_F(UtestCompileCachePolicy, AgingCacheSuccess_1) {
  CompileCacheDescPtr cache_desc = std::make_shared<CompileCacheDesc>();
  cache_desc->SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  cache_desc->AddTensorInfo(tensor_info);
  ASSERT_EQ(cache_desc->GetTensorInfoSize(), 1);
  ASSERT_EQ(cache_desc->MutableTensorInfo(1), nullptr);
  ASSERT_NE(cache_desc->MutableTensorInfo(0), nullptr);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  BinaryHolder holder_new;
  holder_new = holder;
  ASSERT_NE(holder_new.GetDataPtr(), nullptr);
  cache_desc->AddBinary(holder_new);
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<CacheItemId> del_item = ccp->DoAging();
  ASSERT_EQ(cache_id, del_item[0]);
}

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_RuntimeCachePolicy_1) {
  auto ccp = ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_FOR_EXACTLY_THE_SAME,
                                     ge::AgingPolicyType::AGING_POLICY_LRU_K);
  ASSERT_NE(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_RuntimeCachePolicy_2) {
  auto mp_ptr = PolicyRegister::GetInstance().GetMatchPolicy(ge::MatchPolicyType::MATCH_POLICY_FOR_EXACTLY_THE_SAME);
  auto ap_ptr = PolicyRegister::GetInstance().GetAgingPolicy(ge::AgingPolicyType::AGING_POLICY_LRU_K);
  auto ccp = ge::CachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_NE(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, AddCache_ReturnKInvalidCacheItemId_CacheDescNotMetAddCondition) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s{1, 3, 256, 256};
  std::vector<gert::Shape> shapes{s};

  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, AddCache_ReturnValidCacheItemId_CacheDescMetAddCondition) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s{1, 3, 256, 256};
  std::vector<gert::Shape> shapes{s};

  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, AddCache_GetSameCacheId_AddSameCache) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s{1, 3, 256, 256};
  std::vector<gert::Shape> shapes{s};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);

  // add cache 2
  CacheItemId cache_id_same = cp->AddCache(cache_desc);
  EXPECT_TRUE(cache_id_same == cache_id);
}

TEST_F(UtestCompileCachePolicy, AddCache_GetSameCacheId_AddAnotherSameCache) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s{1, 3, 256, 256};
  std::vector<gert::Shape> shapes{s};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);

  // add cache 2
  auto another_cache_desc = std::make_shared<RuntimeCacheDesc>();
  another_cache_desc->SetShapes(shapes);
  CacheItemId cache_id_same = cp->AddCache(another_cache_desc);
  EXPECT_EQ(cache_id_same, cache_id);
}


TEST_F(UtestCompileCachePolicy, AddCache_GetDiffCacheId_OneCacheSetDiffShapes) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s1{1, 3, 256, 256};
  std::vector<gert::Shape> shapes1{s1};
  gert::Shape s2{3, 256, 256};
  std::vector<gert::Shape> shapes2{s2};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes1);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);

  // add cache 2
  cache_desc->SetShapes(shapes2);
  CacheItemId another_cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(another_cache_id, KInvalidCacheItemId);
  another_cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(another_cache_id, KInvalidCacheItemId);
  EXPECT_NE(another_cache_id, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddCache_GetDiffCacheId_AddTwoDiffCache) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s1{1, 3, 256, 256};
  std::vector<gert::Shape> shapes1{s1};
  gert::Shape s2{3, 256, 256};
  std::vector<gert::Shape> shapes2{s2};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes1);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);

  // add cache 2
  auto another_cache_desc = std::make_shared<RuntimeCacheDesc>();
  another_cache_desc->SetShapes(shapes2);
  CacheItemId another_cache_id = cp->AddCache(another_cache_desc);
  EXPECT_EQ(another_cache_id, KInvalidCacheItemId);
  another_cache_id = cp->AddCache(another_cache_desc);
  EXPECT_NE(another_cache_id, KInvalidCacheItemId);
  EXPECT_NE(another_cache_id, cache_id);
}

TEST_F(UtestCompileCachePolicy, FindCache_GetSameId_FindTheCacheAdded) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s1{1, 3, 256, 256};
  std::vector<gert::Shape> shapes1{s1};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes1);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);

  // find cache 1
  CacheItemId another_cache_id = cp->FindCache(cache_desc);
  EXPECT_EQ(another_cache_id, cache_id);
}

TEST_F(UtestCompileCachePolicy, FindCache_GetIdMatched_1HashWith2CacheDescs) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s1{1, 3, 256, 256};
  std::vector<gert::Shape> shapes1{s1};
  gert::Shape s2{3, 256, 256};
  std::vector<gert::Shape> shapes2{s2};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes1);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);
  auto cache_desc_hash = cache_desc->GetCacheDescHash();

  // modify cache_desc which cp saved
  cache_desc->SetShapes(shapes2);

  // add cache 2 with same hash key
  auto another_cache_desc = std::make_shared<RuntimeCacheDesc>();
  another_cache_desc->SetShapes(shapes1);
  EXPECT_EQ(another_cache_desc->GetCacheDescHash(), cache_desc_hash);
  CacheItemId another_cache_id = cp->AddCache(another_cache_desc);
  EXPECT_EQ(another_cache_id, KInvalidCacheItemId);
  another_cache_id = cp->AddCache(another_cache_desc);
  EXPECT_NE(another_cache_id, KInvalidCacheItemId);
  EXPECT_NE(another_cache_id, cache_id);

  // find cache 2
  auto find_cache_desc = std::make_shared<RuntimeCacheDesc>();
  find_cache_desc->SetShapes(shapes1);
  EXPECT_EQ(find_cache_desc->GetCacheDescHash(), cache_desc_hash);
  auto find_cache_id = cp->FindCache(find_cache_desc);
  EXPECT_EQ(another_cache_id, find_cache_id);
}

TEST_F(UtestCompileCachePolicy, FindCache_ReturnKInvalidCacheItemId_HashKeyNotMatched) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s1{0, 3, 256, 256};
  std::vector<gert::Shape> shapes1{s1};
  gert::Shape s2{3, 256, 256};
  std::vector<gert::Shape> shapes2{s2};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes1);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);

  // find cache 2
  auto another_cache_desc = std::make_shared<RuntimeCacheDesc>();
  another_cache_desc->SetShapes(shapes2);
  CacheItemId another_cache_id = cp->FindCache(another_cache_desc);
  EXPECT_EQ(another_cache_id, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, FindCache_ReturnKInvalidCacheItemId_HashMatchedButCacheDescNotMatch) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>();
  auto cp = CachePolicy::Create(mp, ap);

  gert::Shape s1{1, 3, 256, 256};
  std::vector<gert::Shape> shapes1{s1};
  gert::Shape s2{3, 256, 256};
  std::vector<gert::Shape> shapes2{s2};

  // add cache 1
  auto cache_desc = std::make_shared<RuntimeCacheDesc>();
  cache_desc->SetShapes(shapes1);
  CacheItemId cache_id = cp->AddCache(cache_desc);
  EXPECT_EQ(cache_id, KInvalidCacheItemId);
  cache_id = cp->AddCache(cache_desc);
  EXPECT_NE(cache_id, KInvalidCacheItemId);
  auto cache_desc_hash = cache_desc->GetCacheDescHash();

  // modify cache 1 which cp saved
  cache_desc->SetShapes(shapes2);

  // find diff cache with the same hash of cache 1
  auto find_cache_desc = std::make_shared<RuntimeCacheDesc>();
  find_cache_desc->SetShapes(shapes1);
  EXPECT_EQ(find_cache_desc->GetCacheDescHash(), cache_desc_hash);
  auto find_cache_id = cp->FindCache(find_cache_desc);
  EXPECT_EQ(find_cache_id, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, DoAging_NoAgingId_CacheQueueNotReachDepth) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>(1);
  auto cp = CachePolicy::Create(mp, ap);

  uint16_t depth = 1;
  auto add_cache_ids = AddCachesByDepth(cp, depth);
  ASSERT_EQ(add_cache_ids.size(), depth);

  auto delete_ids = cp->DoAging();
  EXPECT_EQ(delete_ids.size(), 0);
}

TEST_F(UtestCompileCachePolicy, DoAging_GetAgingIds_CacheQueueOverDepth) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLruK>(1);
  auto cp = CachePolicy::Create(mp, ap);

  uint16_t depth = 2;
  auto add_cache_ids = AddCachesByDepth(cp, depth);
  ASSERT_EQ(add_cache_ids.size(), depth);

  auto delete_ids = cp->DoAging();
  EXPECT_EQ(delete_ids.size(), 1);
  EXPECT_EQ(delete_ids[0], add_cache_ids[0]);
}

TEST_F(UtestCompileCachePolicy, DoAging_Aging2Times_CacheQueueOverDepth) {
  size_t cached_aging_depth = 1U;
  auto cp = CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_FOR_EXACTLY_THE_SAME,
                                ge::AgingPolicyType::AGING_POLICY_LRU_K, cached_aging_depth);

  uint16_t depth = 3;
  auto add_cache_ids = AddCachesByDepth(cp, depth);
  ASSERT_EQ(add_cache_ids.size(), depth);

  for (size_t i = 0U; i < 2U; ++i) {
    auto delete_ids = cp->DoAging();
    ASSERT_EQ(delete_ids.size(), 1);
    EXPECT_EQ(delete_ids[0], add_cache_ids[i]);
  }
}

TEST_F(UtestCompileCachePolicy, DoAging_TestSetIntervalForLRU) {
  auto mp = std::make_shared<MatchPolicyForExactlyTheSame>();
  auto ap = std::make_shared<AgingPolicyLru>();
  auto cp = CachePolicy::Create(mp, ap);

  uint16_t depth = 3;
  auto add_cache_ids = AddCachesByDepthForLRU(cp, depth);
  ASSERT_EQ(add_cache_ids.size(), depth);

  ap->SetDeleteInterval(depth);
  auto delete_ids = cp->DoAging();
  EXPECT_EQ(delete_ids.size(), 0U);

  ap->SetDeleteInterval(1U);
  delete_ids = cp->DoAging();
  ASSERT_EQ(delete_ids.size(), 2);
  EXPECT_TRUE(delete_ids[0] == 0 || delete_ids[0] == 1);
  EXPECT_TRUE(delete_ids[1] == 0 || delete_ids[1] == 1);
}
}
