/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#define protected public
#define private public
#include "ub_pass_slice_info/aipp_conv_slice_info.h"
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "graph_constructor.h"
#undef protected
#undef private
using namespace std;
using namespace fe;

class AIPP_CONV_SLICE_INFO_STEST : public testing::Test {
 public:
 protected:
  virtual void SetUp() {
    aipp_conv_slice_info_ptr = std::make_shared<AippConvSliceInfo>();
  }

  virtual void TearDown() {

  }

  void BuildGraph_1(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
        original_shape);
    test.AddOpDesc(EN_IMPL_HW_TBE, "strided_read", "stridedRead", "StridedRead", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "strided_write", "stridedWrite", "StridedWrite", 1, 1)
        .SetInput("stridedRead", "Data_0")
        .SetInput("conv", "stridedRead")
        .SetInputs("dequant", {"conv", "Data"})
        .SetInput("add", "dequant")
        .SetInput("quant", "add")
        .SetInput("stridedWrite", "quant");
  }

 public:
  std::shared_ptr<AippConvSliceInfo> aipp_conv_slice_info_ptr;
};

TEST_F(AIPP_CONV_SLICE_INFO_STEST, modify_slice_info_by_pattern_suc) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_1(graph);
  ge::NodePtr fusion_node = graph->FindNode("dequant");
  const vector<ge::NodePtr> fusion_nodes;
  OpCalcInfo op_calc_info;
  op_calc_info.Initialize();
  size_t input_size = 1;
  bool is_head_fusion;
  Status ret = aipp_conv_slice_info_ptr->ModifySliceInfoByPattern(fusion_node, fusion_nodes, op_calc_info,
                                                                         input_size, is_head_fusion);
  EXPECT_EQ(fe::SUCCESS, ret);
}
