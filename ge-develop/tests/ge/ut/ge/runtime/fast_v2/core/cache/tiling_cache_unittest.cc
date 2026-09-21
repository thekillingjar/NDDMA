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
#include "kernel/cache_strategy.h"
#include "kernel/tiling_cache.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "framework/ge_runtime_stub/include/faker/ge_model_builder.h"
#include "faker/global_data_faker.h"
#include "faker/node_faker.h"
#include "exe_graph/lowering/value_holder.h"

namespace gert {
class TilingCacheUt : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

namespace {
TilingCacheKey GetTempTilingCacheKey(const Shape &shape) {
  HashBuffer hash_buf;
  hash_buf.AddParamToBuf(shape);
  return hash_buf.GetTilingCacheKey();
}
}  // namespace

TEST_F(TilingCacheUt, TilingCacheAddAndGet_Ok_TriggleEviction) {
  constexpr size_t kCacheCapacity = 6UL;
  constexpr size_t kCacheEnvicNum = 4UL;
  std::unique_ptr<TilingCacheStrategy> strategy(new TilingCacheLruStrategy(kCacheCapacity, kCacheEnvicNum));
  ASSERT_NE(strategy.get(), nullptr);
  TilingCacheManager tiling_cache_mgr(std::move(strategy));
  std::vector<TilingCacheValue> values(8UL);
  std::vector<Shape> shapes;
  for (size_t i = 0; i < values.size(); i++) {
    values[i].tiling_key = i;
    shapes.push_back({static_cast<int64_t>(i), static_cast<int64_t>(i)});
  }
  // 成功添加, {5, 4, 3, 2, 1, 0}
  for (size_t i = 0; i < kCacheCapacity; i++) {
    tiling_cache_mgr.AddNewCache(GetTempTilingCacheKey(shapes[i]), std::move(values[i]));
    EXPECT_EQ(tiling_cache_mgr.Exist(GetTempTilingCacheKey(shapes[i])), true);
  }
  // Lru元素刷新 {0, 5, 4, 3, 2, 1}
  auto cache0 = tiling_cache_mgr.TryFetchCache(GetTempTilingCacheKey(shapes[0]));
  ASSERT_NE(cache0, nullptr);
  EXPECT_EQ(cache0->GetTilingCacheValue().tiling_key, 0);
  // 触发Lru回收, 添加6, {0, 5, 4, 3, 2, 1}->{6, 0, 5}
  tiling_cache_mgr.AddNewCache(GetTempTilingCacheKey(shapes[6]), std::move(values[6]));
  for (size_t i = 1; i <= kCacheEnvicNum; i++) {
    EXPECT_EQ(tiling_cache_mgr.Exist(GetTempTilingCacheKey(shapes[i])), false);
  }
  EXPECT_EQ(tiling_cache_mgr.Exist(GetTempTilingCacheKey(shapes[0])), true);
  EXPECT_EQ(tiling_cache_mgr.Exist(GetTempTilingCacheKey(shapes[6])), true);
  // 成功添加, {6, 0, 5}->{7, 6, 0, 5}
  tiling_cache_mgr.AddNewCache(GetTempTilingCacheKey(shapes[7]), std::move(values[7]));
  for (size_t i = kCacheEnvicNum + 1; i < values.size(); i++) {
    const auto cache = tiling_cache_mgr.TryFetchCache(GetTempTilingCacheKey(shapes[i]));
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->GetTilingCacheValue().tiling_key, i);
  }
}

TEST_F(TilingCacheUt, TilingCacheAddAndGet_Ok_AddRepeatedCache) {
  constexpr size_t kCacheCapacity = 3UL;
  constexpr size_t kCacheEnvicNum = 2UL;
  std::unique_ptr<TilingCacheStrategy> strategy(new TilingCacheLruStrategy(kCacheCapacity, kCacheEnvicNum));
  ASSERT_NE(strategy.get(), nullptr);
  TilingCacheManager tiling_cache(std::move(strategy));
  std::vector<TilingCacheValue> values(4UL);
  std::vector<Shape> shapes;
  std::vector<TilingCache> caches;
  for (size_t i = 0; i < values.size(); i++) {
    values[i].tiling_key = i;
    shapes.push_back({static_cast<int64_t>(4), static_cast<int64_t>(256)});
    TilingCache new_cache(GetTempTilingCacheKey(shapes[i]), {});
    caches.emplace_back(std::move(new_cache));
  }

  for (size_t i = 0; i < values.size(); i++) {
    // call move assignment operator
    TilingCache new_cache(GetTempTilingCacheKey(shapes[i]), {});
    new_cache = std::move(caches[i]);
    EXPECT_EQ(TilingCacheKeyHash()(new_cache.GetTilingCacheKey()),
              TilingCacheKeyHash()(GetTempTilingCacheKey(shapes[i])));
    EXPECT_EQ(new_cache.GetTilingCacheKey(), GetTempTilingCacheKey(shapes[i]));
    tiling_cache.AddNewCache(new_cache.GetTilingCacheKey(), std::move(values[i]));
    const auto cache = tiling_cache.TryFetchCache(GetTempTilingCacheKey(shapes[i]));
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->GetTilingCacheValue().tiling_key, 0);
  }
}

TEST_F(TilingCacheUt, TilingCache_Ok_InvalidCacheKey) {
  constexpr size_t kCacheCapacity = 3UL;
  constexpr size_t kCacheEnvicNum = 2UL;
  std::unique_ptr<TilingCacheStrategy> strategy(new TilingCacheLruStrategy(kCacheCapacity, kCacheEnvicNum));
  ASSERT_NE(strategy.get(), nullptr);
  TilingCacheManager tiling_cache(std::move(strategy));
  TilingCacheKey invalid_key;
  EXPECT_EQ(tiling_cache.TryFetchCache(invalid_key), nullptr);
}

TEST_F(TilingCacheUt, TilingCache_Fail_CacheLenExceeds) {
  HashBuffer hash_buf;
  for (size_t i = 0; i < kMaxHashBufSize; ++i) {
    hash_buf.AddParamToBuf({1,2,3,4});
  }
  EXPECT_EQ(hash_buf.GetTilingCacheKey().len, kInvalidHashBufOffset);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().IsValid(), false);
}

TEST_F(TilingCacheUt, TilingCache_Ok_HashBufferWithTailBlock) {
  auto allocator = memory::CachingMemAllocator::GetAllocator();
  std::vector<int8_t> td = {1};
  Tensor tensor = {{{1}, {1}}, {}, kOnHost, ge::DT_INT8, td.data()};
  HashBuffer hash_buf;
  hash_buf.AddParamToBuf(tensor);
  auto key = hash_buf.GetTilingCacheKey();
  EXPECT_EQ(key.IsValid(), true);
  EXPECT_EQ(key.len, 25);
  TilingCacheKeyHash hasher;
  (void)hasher(key);
}

TEST_F(TilingCacheUt, TilingCache_Ok_MultipleHashBufferInstance) {
  HashBuffer hash_buf_ok;
  HashBuffer hash_buf_other;
  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(1 * 2 * 3 * 4 * 2);
  Tensor tensor = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, kOnHost, {}, const_cast<void *>(mem_block->GetAddr())};

  hash_buf_ok.AddParamToBuf({1,2,3,4});
  hash_buf_ok.AddParamToBuf(tensor);
  const auto key_ok_expected = hash_buf_ok.GetTilingCacheKey();
  EXPECT_EQ(key_ok_expected.IsValid(), true);
  // 第二个实例无法增加参数, 不影响第一个实例使用
  hash_buf_other.AddParamToBuf({1,2,3,4});
  hash_buf_other.AddParamToBuf(tensor);
  const auto key_other = hash_buf_other.GetTilingCacheKey();
  EXPECT_EQ(key_other.IsValid(), false);
  EXPECT_EQ(key_other.buf, nullptr);
  EXPECT_EQ(key_other.len, 0);
  const auto key_ok = hash_buf_ok.GetTilingCacheKey();
  EXPECT_TRUE(key_ok == key_ok_expected);

  mem_block->Free();
}

TEST_F(TilingCacheUt, TilingCache_Ok_AddTensorDataToHashBuffer) {
  HashBuffer hash_buf;
  auto allocator = memory::CachingMemAllocator::GetAllocator();
  auto mem_block = allocator->Malloc(1 * 2 * 3 * 4 * 2);
  Tensor tensor1 = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, kOnHost, {}, const_cast<void *>(mem_block->GetAddr())};
  hash_buf.AddParamToBuf(tensor1);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().IsValid(), true);

  // Case for when tensor placement is not on host
  tensor1.SetPlacement(kOnDeviceHbm);
  hash_buf.AddParamToBuf(tensor1);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().len, kInvalidHashBufOffset);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().IsValid(), false);

  // case for when tensor size exceeds the upper bound
  tensor1.SetPlacement(kOnHost);
  tensor1.SetSize(kMaxHashBufSize);
  hash_buf.AddParamToBuf(tensor1);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().len, kInvalidHashBufOffset);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().IsValid(), false);

  // free memory
  mem_block->Free();

  // case for when tensor data's address is nullptr
  Tensor tensor2 = {{{1, 2, 3, 4}, {1, 2, 3, 4}}, {}, kOnHost, {}, 0};
  hash_buf.AddParamToBuf(tensor2);
  EXPECT_EQ(hash_buf.GetTilingCacheKey().IsValid(), false);
}

TEST_F(TilingCacheUt, TilingCache_Ok_CheckDataDependentOperator) {
  IMPL_OP(DDIT02).InputsDataDependency({0, 2});
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  bg::ValueHolder::PushGraphFrame();

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(ub_graph)
                   .NameAndType("Data0", "Data")
                   .IoNum(0, 1)
                   .Build();

  std::vector<std::string> inputNames;
  inputNames.reserve(3);
  for (int i = 0; i < 3; i++) {
    inputNames.emplace_back("x" + std::to_string(i));
  }
  auto ddit02 =
      ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT02").IoNum(3, 1).InputNames(inputNames).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"x0", "x3"});

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(i)), ge::GRAPH_SUCCESS);
  }

  auto root_model = GeModelBuilder(ub_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();

  size_t data_dependency = 0U;
  EXPECT_EQ(TilingCacheUtils::IsOpSupportTilingCache(ddit02, global_data, data_dependency), true);
  EXPECT_EQ(data_dependency, 5U);

  while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
}

TEST_F(TilingCacheUt, TilingCache_Ok_CheckDataDependentOperatorOutOfRange) {
  IMPL_OP(DDIT02).InputsDataDependency({0, 64});
  bg::ValueHolder::PushGraphFrame();

  auto ub_graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(ub_graph).NameAndType("Data0", "Data").IoNum(0, 1).Build();

  std::vector<std::string> inputNames;
  inputNames.reserve(65);
  for (int i = 0; i < 65; i++) {
    inputNames.emplace_back("x" + std::to_string(i));
  }
  auto ddit02 = ComputeNodeFaker(ub_graph).NameAndType("Test", "DDIT02").IoNum(65, 1).InputNames(inputNames).Build();
  ddit02->GetOpDesc()->SetOpInferDepends({"x0", "x64"});

  for (int i = 0; i < 65; i++) {
    EXPECT_EQ(ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), ddit02->GetInDataAnchor(i)), ge::GRAPH_SUCCESS);
  }

  auto root_model = GeModelBuilder(ub_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();

  size_t data_dependency = 0U;
  EXPECT_EQ(TilingCacheUtils::IsOpSupportTilingCache(ddit02, global_data, data_dependency), false);
  EXPECT_TRUE(data_dependency == 0U);

  while (bg::ValueHolder::PopGraphFrame() != nullptr) {
  }
}
}
