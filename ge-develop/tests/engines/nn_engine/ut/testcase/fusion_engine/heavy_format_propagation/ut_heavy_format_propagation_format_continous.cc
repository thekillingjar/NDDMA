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
#include <nlohmann/json.hpp>
#include <memory>
#include "common/util/op_info_util.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "../../../../graph_constructor/graph_constructor.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
using namespace std;
using namespace ge;
using namespace fe;


using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;
using HeavyFormatPropagationPtr = std::shared_ptr<HeavyFormatPropagation>;
class UTEST_fusion_engine_heavy_format_continous_distribution : public testing::Test
{
 protected:
  void SetUp()
  {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo heavy_op_info {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/heavy_opinfo",
        "",
        false,
        false,
        false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(heavy_op_info);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);

    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
  }

  void TearDown()
  {

  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
 protected:
};

// Ignore the failed at 11/27.
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, switch_input_format_diff_failed)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *        am1(NCHW)           am3(NHWC)
   *              \                /
   *               \              /
   *                \            /
   *                 Switch(NCHW)
   *                /     |      \ (this two edge are in the exception of
   *               /      |       \ format agonostic)
   *              /       |        \
   *      Conv2D(5HD)   am2(NCHW)  am4(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("switch", "Switch")
    .AddOpDesc("am1", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)
    .SetInput("switch:1", ge::FORMAT_NCHW, "am3", ge::FORMAT_NHWC);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "switch", ge::FORMAT_NCHW);
  test.SetInput("am2:0", "switch:1");
  test.SetInput("am4:0", "switch:2");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), OUTPUT_FORMAT_AGNOSTIC_EXCEPTION,
                            {1, 2});
  SetTensorDescIntAttr(node->GetOpDesc(), 0, true, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, true, FORMAT_CONTINUOUS, 1);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, switch_input_format_all_success)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *        am1(NCHW)           am3(NCHW)
   *              \                /
   *               \              /
   *                \            /
   *                 Switch(NCHW)
   *                /     |      \
   *               /      |       \
   *              /       |        \
   *      Conv2D(5HD)   am2(NCHW)  am4(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("switch", "Switch")
    .AddOpDesc("am1", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum", 5, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)
    .SetInput("switch:1", ge::FORMAT_NCHW, "am3", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "switch", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:2", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN);

  test.SetInput("am2:0", "switch:1");
  test.SetInput("am4:0", "switch:2");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  SetTensorDescIntAttr(node->GetOpDesc(), 0, true, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, true, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(2)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Switch") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(2).GetFormat()));
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(2).GetShape().GetDims());
    }
  }
}

// One input tensor is the exceptiono of format agonostic, dont need to check diff of format.
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, switch_input_format_all_success2)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *        am1(NCHW)           am3(NHWC)
   *              \                /(this edge is in the exception of format agonostic)
   *               \              /
   *                \            /
   *                 Switch(NCHW)
   *                /     |      \
   *               /      |       \
   *              /       |        \
   *      Conv2D(5HD)   am2(NCHW)  am4(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("switch", "Switch")
    .AddOpDesc("am1", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum", 5, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)
    .SetInput("switch:1", ge::FORMAT_NCHW, "am3", ge::FORMAT_NHWC);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "switch", ge::FORMAT_NCHW)
    .SetInput("conv2d:1", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:2", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN);

  test.SetInput("am2:0", "switch:1");
  test.SetInput("am4:0", "switch:2");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION,{1});
  SetTensorDescIntAttr(node->GetOpDesc(), 0, true, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, true, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(2)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Switch") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      vector<int64_t> src_shape({1, 2, 3, 4});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(2).GetFormat()));
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(src_shape, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(2).GetShape().GetDims());
    }
  }
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, switch_input_format_pair_success)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *        am1(NCHW)           am3(NHWC)
   *              \                /
   *               \              /
   *                \            /
   *                 Switch(HWCN)
   *                /            \
   *               /              \
   *              /                \
   *      Conv2D(5HD)            am4(HWCN)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("switch", "Switch")
    .AddOpDesc("am1", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum", 5, 1);

  test.SetInput("switch:0", ge::FORMAT_HWCN, "am1", ge::FORMAT_NCHW)
    .SetInput("switch:1", ge::FORMAT_HWCN, "am3", ge::FORMAT_NHWC);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "switch", ge::FORMAT_HWCN)
    .SetInput("conv2d:1", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:2", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN);

  test.SetInput("am4:0", ge::FORMAT_HWCN, "switch:1", ge::FORMAT_HWCN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT));

  SetTensorDescIntAttr(node->GetOpDesc(), 0, true, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, true, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Switch") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 5, 6, 12});
      vector<int64_t> dim_result2({3, 12, 5, 6});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NHWC, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NHWC, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(dim_result2, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result2, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, switch_input_format_pair_success2)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *        am1(NCHW)           am3(NHWC)
   *              \                /
   *               \              / (this edge is in the exception of format agonostic)
   *                \            /
   *                 Switch(HWCN)
   *                /            \
   *               /              \
   *              /                \
   *      Conv2D(5HD)            am4(HWCN)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("switch", "Switch")
    .AddOpDesc("am1", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum", 5, 1);

  test.SetInput("switch:0", ge::FORMAT_HWCN, "am1", ge::FORMAT_NCHW)
      .SetInput("switch:1", ge::FORMAT_HWCN, "am3", ge::FORMAT_NHWC);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "switch", ge::FORMAT_HWCN)
      .SetInput("conv2d:1", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:2", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN);

  test.SetInput("am4:0", ge::FORMAT_HWCN, "switch:1", ge::FORMAT_HWCN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION,{1});
  SetTensorDescIntAttr(node->GetOpDesc(), 0, true, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, true, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Switch") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      vector<int64_t> src_shape({1, 2, 3, 4});
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_HWCN, opdesc->GetInputDesc(1).GetFormat());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_HWCN, opdesc->GetOutputDesc(1).GetFormat());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(src_shape, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(src_shape, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

// Ignore the failed at 11/27.
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_diff_failed)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)   am1(NCHW)  am2(NCHW)
   *                \         |         /
   *                 \        |        / (this two edge are in the exception of
   *                  \       |       / format agonostic)
   *                  merge (NCHW,NCHW)
   *                  /              \
   *                 /                \
   *                /                  \
   *          am3(NCHW)             am4(NHWC)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am1", "ApplyMomentum", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0)
      .SetInput("merge", "am1")
      .SetInput("merge", "am2");

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_NCHW);
  test.SetInput("am4:0", ge::FORMAT_NHWC,"merge:1", ge::FORMAT_NCHW);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION,
                            {1, 2});
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, false, FORMAT_CONTINUOUS, 1);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_all_success)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)   am1(NCHW)  am2(NCHW)
   *                \         |         /
   *                 \        |        /
   *                  \       |       /
   *                  merge (NCHW,NCHW)
   *                  /              \
   *                 /                \
   *                /                  \
   *          am3(NCHW)             am4(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am1", "ApplyMomentum", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge:0", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge:1", "am1")
    .SetInput("merge:2", "am2");

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_NCHW);
  test.SetInput("am4:0", ge::FORMAT_NCHW,"merge:1", ge::FORMAT_NCHW);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;
  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, false, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableInputDesc(2)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(2).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(2).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

// One output tensor is the exceptiono of format agonostic, dont need to check diff of format.
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_all_success2)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)   am1(NCHW)  am2(NCHW)
   *                \         |         /
   *                 \        |        /
   *                  \       |       /
   *                  merge (CHWN,CHWN)
   *                  /              \
   *                 /                \ (this edge is in the exception of format agonostic)
   *                /                  \
   *          am3(NCHW)             am4(NHWC)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am1", "ApplyMomentum", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", ge::FORMAT_CHWN, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge", ge::FORMAT_CHWN, "am1", ge::FORMAT_NCHW)
    .SetInput("merge", ge::FORMAT_CHWN, "am2", ge::FORMAT_NCHW);

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_CHWN);
  test.SetInput("am4:0", ge::FORMAT_NHWC,"merge:1", ge::FORMAT_CHWN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), OUTPUT_FORMAT_AGNOSTIC_EXCEPTION, {1});
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, false, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableInputDesc(2)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      vector<int64_t> src_shape({1, 2, 3, 4});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(2).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_CHWN, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(2).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(src_shape, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_all_success3)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)   am1(NCHW)  am2(NCHW)
   *                \         |         /
   *                 \        |        /
   *                  \       |       /
   *                  merge (NCHW,NCHW)
   *                         /\
   *                        /  \
   *                       /    \
   *                 am3(NCHW)  am4(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am1", "ApplyMomentum", 5, 1)
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge:0", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge:1", "am1")
    .SetInput("merge:2", "am2");

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_NCHW);
  test.SetInput("am4:0", ge::FORMAT_NCHW,"merge:0", ge::FORMAT_NCHW);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;
  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableInputDesc(2)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(2).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(2).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_pair_success)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)           am2(HWCN)
   *                \                   /
   *                 \                 /
   *                  \               /
   *                  merge (HWCN,HWCN)
   *                  /              \
   *                 /                \
   *                /                  \
   *          am3(NCHW)             am4(NHWC)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge:0", ge::FORMAT_HWCN, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge:1", ge::FORMAT_HWCN, "am2", ge::FORMAT_HWCN);

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_HWCN);
  test.SetInput("am4:0", ge::FORMAT_NHWC,"merge:1", ge::FORMAT_HWCN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;
  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT));
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, false, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result1({3, 12, 5, 6});
      vector<int64_t> dim_result2({3, 5, 6, 12});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NHWC, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NHWC, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(dim_result1, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result2, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result1, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result2, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_pair_success2)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)           am2(HWCN)
   *                \                   /
   *                 \                 /
   *                  \               /
   *                  merge (HWCN,HWCN)
   *                  /              \
   *                 /                \ (this edge is in the exception of format agonostic)
   *                /                  \
   *          am3(NCHW)             am4(NHWC)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge:0", ge::FORMAT_HWCN, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge:1", ge::FORMAT_HWCN, "am2", ge::FORMAT_HWCN);

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_HWCN);
  test.SetInput("am4:0", ge::FORMAT_NHWC,"merge:1", ge::FORMAT_HWCN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;
  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), OUTPUT_FORMAT_AGNOSTIC_EXCEPTION, {1});
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, false, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result({3, 12, 5, 6});
      vector<int64_t> src_shape({1, 2, 3, 4});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_HWCN, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_HWCN, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(dim_result, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(src_shape, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(src_shape, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_pair_success3)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)           am2(HWCN)
   *                \                   /
   *                 \                 /
   *                  \               /
   *                  merge (HWCN,HWCN)
   *                 /\              /\
   *                /  \            /  \
   *               /    \          /    \
   *       am3(NCHW) am5(NCHW) am4(NHWC) am6(NHWC)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am5", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am6", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge:0", ge::FORMAT_HWCN, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge:1", ge::FORMAT_HWCN, "am2", ge::FORMAT_HWCN);

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_HWCN);
  test.SetInput("am5", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_HWCN);
  test.SetInput("am4:0", ge::FORMAT_NHWC,"merge:1", ge::FORMAT_HWCN);
  test.SetInput("am6:0", ge::FORMAT_NHWC,"merge:1", ge::FORMAT_HWCN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;
  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT));
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);
  SetTensorDescIntAttr(node->GetOpDesc(), 1, false, FORMAT_CONTINUOUS, 1);

  auto opdesc = node->GetOpDesc();
  ge::GeShape pre_shape = GeShape({1, 2, 3, 4});
  opdesc->MutableInputDesc(0)->SetShape(pre_shape);
  opdesc->MutableInputDesc(1)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(0)->SetShape(pre_shape);
  opdesc->MutableOutputDesc(1)->SetShape(pre_shape);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_result1({3, 12, 5, 6});
      vector<int64_t> dim_result2({3, 5, 6, 12});
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NHWC, (opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, (opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NHWC, (opdesc->GetOutputDesc(1).GetFormat()));
      EXPECT_EQ(dim_result1, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result2, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(dim_result1, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(dim_result2, opdesc->GetOutputDesc(1).GetShape().GetDims());
    }
  }
}

// Ignore the failed at 11/27.
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, merge_input_format_pair_failed)
{
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *          Conv2D(5HD)           am2(HWCN)
   *                \                   /
   *                 \                 /
   *                  \               /
   *                  merge (HWCN,HWCN)
   *                        /\
   *                       /  \
   *                      /    \
   *                am3(NCHW)  am4(NHWC)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
    .AddOpDesc("merge", "Merge")
    .AddOpDesc("am2", "ApplyMomentum", 5, 1)
    .AddOpDesc("am3", "ApplyMomentum2", 5, 1)
    .AddOpDesc("am4", "ApplyMomentum2", 5, 1)
    .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
    .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
    .SetInput("conv2d:2", "", {12});

  test.SetInput("merge:0", ge::FORMAT_HWCN, "conv2d", ge::FORMAT_NC1HWC0)
    .SetInput("merge:1", ge::FORMAT_HWCN, "am2", ge::FORMAT_HWCN);

  test.SetInput("am3", ge::FORMAT_NCHW, "merge:0", ge::FORMAT_HWCN);
  test.SetInput("am4:0", ge::FORMAT_NHWC,"merge:0", ge::FORMAT_HWCN);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;
  test.GetNodeByName("merge", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT));
  SetTensorDescIntAttr(node->GetOpDesc(), 0, false, FORMAT_CONTINUOUS, 1);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Only if all successors can support the propagated heavy format, the variable will be considered
 * as penetrable(through which we can keep propagating the heavy format. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_01) {
  /*
   * Graph will be like:
   *
   *        am1(NCHW, exception)   Variable(NCHW)----Conv2D(NCHW)
   *              \                / \    \
   *               \              /  \      \
   *                \            /   \        \
   *                 \          /    \       Aicpu(not format agnotisc)
   *                 Switch(NCHW) ReluSpecial(NCHW)
   *                /           \
   *               /            \
   *              /             \
   *      am2(NCHW)           am3(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("switch", "Switch")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("var", "Variable", 1, 4)
      .AddOpDesc("aicpu", "AICPU", 1, 1)
      .AddOpDesc("relu_special", "ReluSpecial", 1, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)

      .SetInput("switch:1", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("relu_special:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("aicpu:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "var:0", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am2:0", "switch:0");
  test.SetInput("am3:0", "switch:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION, {0});

  auto opdesc = node->GetOpDesc();
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "switch") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(1).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "relu_special") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am1") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am3") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }
  }
}


/* Only if all successors can support the propagated heavy format, the variable will be considered
 * as penetrable(through which we can keep propagating the heavy format. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_02) {
  /*
   * Graph will be like:
   *
   *        am1(NCHW)   Variable(NCHW)----Conv2D(NCHW)
   *              \                / \ \
   *               \              /  \  \
   *                \            /   \   \
   *                 \          /    \  Aicpu(not format agnotisc)
   *                 Switch(NCHW) Relu6(NCHW)
   *                /           \
   *               /            \
   *              /             \
   *      am2(NCHW)           am3(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("switch", "Switch")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("var", "Variable", 1, 4)
      .AddOpDesc("aicpu", "AICPU", 1, 1)
      .AddOpDesc("relu6", "Relu6", 1, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)

      .SetInput("switch:1", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("relu6:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("aicpu:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "var:0", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am2:0", "switch:0");
  test.SetInput("am3:0", "switch:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));

  auto opdesc = node->GetOpDesc();
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "switch") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(1).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "relu6") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am1") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am3") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }
  }
}

/* Only if all successors can support the propagated heavy format, the variable will be considered
 * as penetrable(through which we can keep propagating the heavy format. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_03) {
  /*
   * Graph will be like:
   *
   *        am1(NCHW)   Variable(NCHW)-------Conv2D(NCHW)
   *              \                / \ \
   *               \              /  \  \
   *                \            /   \   \
   *                 \          /    \  Aicpu(format agnotisc)
   *                 Switch(NCHW) Relu6(NCHW)
   *                /           \
   *               /            \
   *              /             \
   *      am2(NCHW)           am3(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("switch", "Switch")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("var", "Variable", 1, 4)
      .AddOpDesc("aicpu", "AICPU", 1, 1)
      .AddOpDesc("relu6", "Relu6", 1, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)

      .SetInput("switch:1", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("relu6:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("aicpu:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "var:0", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am2:0", "switch:0");
  test.SetInput("am3:0", "switch:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION, {0});

  ge::NodePtr aicpu_node;
  test.GetNodeByName("aicpu", aicpu_node);
  ge::AttrUtils::SetInt(aicpu_node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));

  auto opdesc = node->GetOpDesc();
  HeavyFormatPropagationPtr heavt_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);

  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "switch") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(1).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(1).GetFormat()), ge::FORMAT_NC1HWC0);
    }

    if (node->GetName() == "relu6") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
    }

    if (node->GetName() == "am1") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

    }

    if (node->GetName() == "am3") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

    }
  }
}

/* Only if all successors can support the propagated heavy format, the variable will be considered
 * as penetrable(through which we can keep propagating the heavy format. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_04) {
  /*
   * Graph will be like:
   *
   *        am1(NCHW)   Variable(NCHW)-------Conv2D(NCHW)
   *              \                / \ \
   *               \              /  \  \
   *                \            /   \   \
   *                 \   exception   \  Aicpu(not format agnotisc)
   *                 Switch(NCHW) Relu6(NCHW)
   *                /           \
   *               /            \
   *              /             \
   *      am2(NCHW)           am3(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("switch", "Switch")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("var", "Variable", 1, 4)
      .AddOpDesc("aicpu", "AICPU", 1, 1)
      .AddOpDesc("relu6", "Relu6", 1, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)

      .SetInput("switch:1", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("relu6:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("aicpu:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "var:0", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am2:0", "switch:0");
  test.SetInput("am3:0", "switch:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION, {1});

  ge::NodePtr aicpu_node;
  test.GetNodeByName("aicpu", aicpu_node);
  ge::AttrUtils::SetInt(aicpu_node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));

  auto opdesc = node->GetOpDesc();
  HeavyFormatPropagationPtr heavt_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);

  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "switch") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(1).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "relu6") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am1") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);

    }

    if (node->GetName() == "am3") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);

    }
  }
}


/* Only if all successors can support the propagated heavy format,(for switch
 * we will check the successors of switch. The variable will be considered
 * as penetrable(through which we can keep propagating the heavy format. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_05) {
  /*
   * Graph will be like:
   *
   *        am1(NCHW)   Variable(NCHW)-------Conv2D(NCHW)
   *              \                / \ \
   *               \              /  \  \
   *                \            /   \   \
   *                 \          /    \  Aicpu(format agnotisc)
   *                 Switch(NCHW) Relu6(NCHW)
   *                /    |       \
   *               /     |        \
   *              /      |         \
   *      am2(NCHW)    Switch      am3(NCHW)
   *
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("switch", "Switch")
      .AddOpDesc("switch2", "Switch")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("var", "Variable", 1, 4)
      .AddOpDesc("aicpu", "AICPU", 1, 1)
      .AddOpDesc("relu6", "Relu6", 1, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)

      .SetInput("switch:1", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("relu6:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("aicpu:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "var:0", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am2:0", "switch:0");
  test.SetInput("am3:0", "switch:1");
  test.SetInput("switch2:0", "switch:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION, {0});

  test.GetNodeByName("switch2", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));

  ge::NodePtr aicpu_node;
  test.GetNodeByName("aicpu", aicpu_node);
  ge::AttrUtils::SetInt(aicpu_node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));

  auto opdesc = node->GetOpDesc();
  HeavyFormatPropagationPtr heavt_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);

  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "switch") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(1).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(1).GetFormat()), ge::FORMAT_NC1HWC0);
    }

    if (node->GetName() == "switch2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_ND);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_ND);
      EXPECT_EQ(desc->GetOutputDesc(1).GetFormat(), ge::FORMAT_ND);
    }
	
    if (node->GetName() == "relu6") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);
    }

    if (node->GetName() == "am1") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

    }

    if (node->GetName() == "am3") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetInputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

      EXPECT_EQ(ge::GetPrimaryFormat(desc->GetOutputDesc(0).GetFormat()), ge::FORMAT_NC1HWC0);

    }
  }
}


/* ReluSpecial is cannot support NC1HWC0, we will stop the propagation. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_06) {
  /*
   * Graph will be like:
   *
   *        am1(NCHW)   Variable(NCHW)-------Conv2D(NCHW)
   *              \                / \ \
   *               \              /  \  \
   *                \            /   \   \
   *                 \          /    \  Aicpu(format agnotisc)
   *                 Switch(NCHW) Relu6(NCHW)
   *                /    |      \
   *               /     |       \
   *              /      |        \
   *      am2(NCHW) ReluSpecial  am3(NCHW)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("switch", "Switch")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("var", "Variable", 1, 4)
      .AddOpDesc("aicpu", "AICPU", 1, 1)
      .AddOpDesc("relu6", "Relu6", 1, 1)
      .AddOpDesc("reluspec", "ReluSpecial", 1, 1);

  test.SetInput("switch:0", ge::FORMAT_NCHW, "am1", ge::FORMAT_NCHW)

      .SetInput("switch:1", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("relu6:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW)
      .SetInput("aicpu:0", ge::FORMAT_NCHW, "var:0", ge::FORMAT_NCHW);

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "var:0", ge::FORMAT_NCHW)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am2:0", "switch:0");
  test.SetInput("am3:0", "switch:1");
  test.SetInput("reluspec:0", "switch:2");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("switch", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION, {0});

  ge::NodePtr aicpu_node;
  test.GetNodeByName("aicpu", aicpu_node);
  ge::AttrUtils::SetInt(aicpu_node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));

  auto opdesc = node->GetOpDesc();
  HeavyFormatPropagationPtr heavt_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);

  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "switch") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(1).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "relu6") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am1") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(1).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }

    if (node->GetName() == "am2") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);

    }

    if (node->GetName() == "am3") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
    }
  }
}


/* NextIteration will is always same data type with its user. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_07) {
  /*
   * Graph will be like:
   *
   *         NextIteration(fp16)
   *              |
   *              |
   *             A(fp32)
   *
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("nextIter", fe::NEXT_ITERATION, 1, 1)
      .AddOpDesc("a", "A", 1, 1);

  test.SetInput("nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT16, "Data:0", ge::FORMAT_NCHW, ge::DT_FLOAT16)
      .SetInput("a:0", ge::FORMAT_NCHW, ge::DT_FLOAT, "nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT16);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("nextIter", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), FORMAT_CONTINUOUS, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), REFRESH_CONTINUOUS_FLAG, true);

  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  Status ret = graphFusion.SetContinuousDtypeForOutput(*graph);

  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "nextIter") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
    }
  }
}


/* NextIteration will is always same data type with its user. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_08) {
  /*
   * Graph will be like:
   *
   *               NextIteration(fp32)
   *                    |
   *                    |
   *                  A(fp16)
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("nextIter", fe::NEXT_ITERATION, 1, 1)
      .AddOpDesc("a", "A", 1, 1);

  test.SetInput("nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT, "Data:0", ge::FORMAT_NCHW, ge::DT_FLOAT)
      .SetInput("a:0", ge::FORMAT_NCHW, ge::DT_FLOAT16, "nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("nextIter", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), FORMAT_CONTINUOUS, 1);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION, {0});
  ge::AttrUtils::SetListInt(node->GetOpDesc(), OUTPUT_FORMAT_AGNOSTIC_EXCEPTION, {0});
  ge::AttrUtils::SetBool(node->GetOpDesc(), REFRESH_CONTINUOUS_FLAG, true);

  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  Status ret = graphFusion.SetContinuousDtypeForOutput(*graph);

  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "nextIter") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
    }
  }
}


/* NextIteration will is always same data type with its user. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_09) {
  /*
   * Graph will be like:
   *
   *         NextIteration(fp16)
   *              |
   *             A(fp32)
   *
   *
   *         The input and Output of NextIteration is in exception list.
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("nextIter", fe::NEXT_ITERATION, 1, 1)
      .AddOpDesc("a", "A", 1, 1);

  test.SetInput("nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT16, "Data:0", ge::FORMAT_NCHW, ge::DT_FLOAT16)
      .SetInput("a:0", ge::FORMAT_NCHW, ge::DT_FLOAT, "nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT16);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("nextIter", node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC,
      static_cast<int64_t>(FormatSelectionType::FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS));
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), FORMAT_CONTINUOUS, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), REFRESH_CONTINUOUS_FLAG, true);

  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  Status ret = graphFusion.SetContinuousDtypeForOutput(*graph.get());

  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "nextIter") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(0).GetDataType(), ge::DT_FLOAT);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
    }
  }
}


/* NextIteration and A and B will is always same data type with its user.
 * They will all be set as fp16. */
TEST_F(UTEST_fusion_engine_heavy_format_continous_distribution, variable_optimize_10) {
  /*
   * Graph will be like:
   *
   *         NextIteration(fp32)
   *              |
   *             A(fp32)
   *              |
   *             B(fp32)
   *              |
   *             C(fp16)
   *       This will be implemented using the reversed order of topological sorting.
   *
   */
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("nextIter", fe::NEXT_ITERATION, 1, 1)
      .AddOpDesc("a", "A", 1, 1)
      .AddOpDesc("b", "B", 1, 1)
      .AddOpDesc("c", "C", 1, 1);

  test.SetInput("nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT, "Data:0", ge::FORMAT_NCHW, ge::DT_FLOAT)
      .SetInput("a:0", ge::FORMAT_NCHW, ge::DT_FLOAT, "nextIter:0", ge::FORMAT_NCHW, ge::DT_FLOAT)
      .SetInput("b:0", ge::FORMAT_NCHW, ge::DT_FLOAT, "a:0", ge::FORMAT_NCHW, ge::DT_FLOAT)
      .SetInput("c:0", ge::FORMAT_NCHW, ge::DT_FLOAT16, "b:0", ge::FORMAT_NCHW, ge::DT_FLOAT);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("nextIter", node);
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), FORMAT_CONTINUOUS, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), REFRESH_CONTINUOUS_FLAG, true);
  test.GetNodeByName("a", node);
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), FORMAT_CONTINUOUS, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), REFRESH_CONTINUOUS_FLAG, true);
  test.GetNodeByName("b", node);
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), FORMAT_CONTINUOUS, 1);
  ge::AttrUtils::SetBool(node->GetOpDesc(), REFRESH_CONTINUOUS_FLAG, true);

  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  Status ret = graphFusion.SetContinuousDtypeForOutput(*graph.get());

  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "nextIter") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
    }
    if (node->GetName() == "a") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
    }
    if (node->GetName() == "b") {
      auto desc = node->GetOpDesc();
      EXPECT_EQ(desc->GetInputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetInputDesc(0).GetDataType(), ge::DT_FLOAT16);

      EXPECT_EQ(desc->GetOutputDesc(0).GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(desc->GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
    }
  }
}


