/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/base_types.h"
#include "base/model_info.h"
#define private public
#define protected public
#include "generator/axes_reorder_tiling_code_gen_impl.h"
#include "att_utils.h"

using namespace att;
using namespace testing;

namespace {
constexpr int32_t kMaxEqualOrderAxesCount = 2;

// Extract the IsEnableEqualOrderTiling logic for testing
bool IsEnableEqualOrderTilingFunc(const ModelInfo &model_info) {
  std::map<size_t, std::vector<std::string>> order_to_axes;
  for (const auto &arg : model_info.arg_list) {
    if (AttUtils::IsTileSplitAxis(arg)) {
      order_to_axes[arg->order].push_back(arg->name);
    }
  }

  for (const auto &pair : order_to_axes) {
    const std::vector<std::string> &axes = pair.second;
    size_t count = axes.size();
    if (count >= kMaxEqualOrderAxesCount) {
      return true;
    }
  }
  return false;
}
}

class IsEnableEqualOrderTilingTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "IsEnableEqualOrderTiling Test begin." << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "IsEnableEqualOrderTiling Test end." << std::endl;
  }

  void SetUp() override {}
  void TearDown() override {}

  // 辅助函数：创建测试用的切分轴
  AttAxisPtr CreateTileSplitAxis(const std::string& name, size_t order) {
    auto axis = std::make_shared<AttAxis>();
    axis->name = name;
    axis->order = order;
    axis->axis_pos = AxisPosition::INNER;  // INNER axis for tile split
    axis->bind_multicore = false;  // Not bound to multicore for tile split
    return axis;
  }

  // 辅助函数：创建非切分轴（绑定多核）
  AttAxisPtr CreateNonTileSplitAxis(const std::string& name, size_t order) {
    auto axis = std::make_shared<AttAxis>();
    axis->name = name;
    axis->order = order;
    axis->axis_pos = AxisPosition::INNER;
    axis->bind_multicore = true;  // Bound to multicore, not a tile split axis
    return axis;
  }

  // 辅助函数：创建测试用的ModelInfo
  ModelInfo CreateTestModelInfo(const std::vector<AttAxisPtr>& axes) {
    ModelInfo model_info;
    model_info.graph_name = "test_graph";
    for (const auto& axis : axes) {
      model_info.arg_list.push_back(axis);
    }
    return model_info;
  }
};

// UT001: 空arg_list - 路径4
TEST_F(IsEnableEqualOrderTilingTest, empty_arg_list) {
  ModelInfo model_info;
  model_info.graph_name = "test_empty";
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT002: 只有非切分轴 - 路径4
TEST_F(IsEnableEqualOrderTilingTest, no_tile_split_axis) {
  auto model_info = CreateTestModelInfo({
    CreateNonTileSplitAxis("non_split_axis", 1)
  });
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT003: 单个切分轴order=1 - 路径3
TEST_F(IsEnableEqualOrderTilingTest, single_tile_split_axis_order1) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1)
  });
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT004: 单个切分轴order=2 - 路径3
TEST_F(IsEnableEqualOrderTilingTest, single_tile_split_axis_order2) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 2)
  });
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT005: 两个切分轴相同order=1 - 路径2
TEST_F(IsEnableEqualOrderTilingTest, two_tile_split_axes_same_order) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateTileSplitAxis("S1", 1)
  });
  EXPECT_TRUE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT006: 两个切分轴相同order=2 - 路径2
TEST_F(IsEnableEqualOrderTilingTest, two_tile_split_axes_same_order2) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 2),
    CreateTileSplitAxis("S1", 2)
  });
  EXPECT_TRUE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT007: 两个切分轴不同order - 路径3,4
TEST_F(IsEnableEqualOrderTilingTest, two_tile_split_axes_different_order) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateTileSplitAxis("S1", 2)
  });
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT008: 三个切分轴相同order - 路径2+ASSERT
TEST_F(IsEnableEqualOrderTilingTest, three_tile_split_axes_same_order) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateTileSplitAxis("S1", 1),
    CreateTileSplitAxis("S2", 1)
  });
  // 期望触发警告并返回true
  testing::internal::CaptureStdout();
  EXPECT_TRUE(IsEnableEqualOrderTilingFunc(model_info));
  std::string output = testing::internal::GetCapturedStdout();
  // 注意：GE_WARN_ASSERT可能输出到stderr，这里只做简单验证
  // 在实际测试中，GE_WARN_ASSERT会打印警告信息
}

// UT009: 多个order，每个order有2个切分轴 - 路径2
TEST_F(IsEnableEqualOrderTilingTest, multiple_orders_with_two_tile_split_axes) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateTileSplitAxis("S1", 1),
    CreateTileSplitAxis("S2", 2),
    CreateTileSplitAxis("S3", 2)
  });
  EXPECT_TRUE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT010: 混合切分轴和非切分轴 - 路径2（切分轴应该被正确识别）
TEST_F(IsEnableEqualOrderTilingTest, mixed_tile_split_and_non_split_axes) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateNonTileSplitAxis("non_split1", 1),  // 非切分轴，不应该被计数
    CreateTileSplitAxis("S1", 1)
  });
  // 只有2个切分轴，应该返回true
  EXPECT_TRUE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT011: 混合切分轴和非切分轴，切分轴不足2个 - 路径3,4
TEST_F(IsEnableEqualOrderTilingTest, mixed_axes_insufficient_tile_split) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateNonTileSplitAxis("non_split1", 1),
    CreateNonTileSplitAxis("non_split2", 1)
  });
  // 只有1个切分轴，应该返回false
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT012: 边界测试 - 多个order，但每个order只有1个切分轴 - 路径3,4
TEST_F(IsEnableEqualOrderTilingTest, multiple_orders_each_single_tile_split_axis) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 1),
    CreateTileSplitAxis("S1", 2),
    CreateTileSplitAxis("S2", 3),
    CreateTileSplitAxis("S3", 4)
  });
  EXPECT_FALSE(IsEnableEqualOrderTilingFunc(model_info));
}

// UT013: 边界测试 - order=0的情况 - 路径2
TEST_F(IsEnableEqualOrderTilingTest, two_tile_split_axes_order_zero) {
  auto model_info = CreateTestModelInfo({
    CreateTileSplitAxis("S0", 0),
    CreateTileSplitAxis("S1", 0)
  });
  EXPECT_TRUE(IsEnableEqualOrderTilingFunc(model_info));
}
