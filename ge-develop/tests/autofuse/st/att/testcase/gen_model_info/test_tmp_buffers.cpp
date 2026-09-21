/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "gen_model_info.h"
#define private public
#include "parser/ascend_graph_parser.h"
#undef private

namespace att {
class TestTmpBuffers : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    graph = std::make_shared<ge::AscGraph>("graph");
    EXPECT_NE(graph, nullptr);
  }
  void TearDown() override {
  }
  std::shared_ptr<ge::AscGraph> graph;
};

TEST_F(TestTmpBuffers, case0)
{
  att::TuningSpacePtr tuning_space = std::make_shared<att::TuningSpace>();
  EXPECT_NE(tuning_space, nullptr);
  att::AscendGraphParser parser(tuning_space);
  std::map<int64_t, Expr> max_tmp_buffers_map;
  std::vector<ge::TmpBuffer> tmp_buffers;
  tmp_buffers = {
      {{CreateExpr(100), 1}},
      {{CreateExpr(200), 2}},
      {{CreateExpr(50), 1}},
      {{CreateExpr(300), 2}}
  };
  parser.SaveTmpBufferInfos("", max_tmp_buffers_map, tmp_buffers);
  tmp_buffers = {
      {{CreateExpr(220), 1}},
      {{CreateExpr(200), 2}},
      {{CreateExpr(50), 1}},
      {{CreateExpr(300), 2}}
  };
  parser.SaveTmpBufferInfos("", max_tmp_buffers_map, tmp_buffers);
  Expr sum = CreateExpr(0.0f);
  for (const auto &pair : max_tmp_buffers_map) {
    sum = sum + pair.second;
  }
  EXPECT_EQ(Str(sum), "770");
}
}