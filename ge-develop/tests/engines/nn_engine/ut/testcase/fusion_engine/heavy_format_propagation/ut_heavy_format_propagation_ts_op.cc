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
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "../../../../graph_constructor/graph_constructor.h"
#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace ge;
using namespace fe;


using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;
using HeavyFormatPropagationPtr = std::shared_ptr<HeavyFormatPropagation>;
class UTestHeavyFormatDistributionTsOp : public testing::Test {
 protected:

  void SetUp() {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo heavy_op_info{
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

  void TearDown() {

  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
};

Status CreateGraphOfTsOp(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                      |
   *                    am3
   *
   * After format inference:
   *     am1(5HD)     am2(5HD)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                      |
   *                    am3(5HD)
   */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2");

  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0);

  test.SetInput("am3", "merge");
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}

Status CreateGraphOfZeroShapeOp1(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /(zero-shape)
   *                 Merge(NCHW)
   *                      |
   *                    am3
   *
   * After format inference:
   *     am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                      |
   *                    am3(NCHW)
   */
  vector<int64_t> conv_original_dims = {3, 12, 5, 6};
  ge::GeShape original_shape = GeShape(conv_original_dims);
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2");

  vector<int64_t> dst_zero_shape = {3, 12, 0, 6};
  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, dst_zero_shape,
                conv_original_dims);

  test.SetInput("am3", "merge");
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}


Status CreateGraphOfZeroShapeOp2(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |             /
   *            \         |            /
   *              \       |(0-shape)  /
   *                 Merge(NCHW)
   *                      |
   *                    am3
   *
   * After format inference:
   *     am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                      |
   *                    am3(NCHW)
   */
  vector<int64_t> conv_original_dims = {3, 12, 5, 6};
  ge::GeShape original_shape = GeShape(conv_original_dims);
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1");

  vector<int64_t> dst_zero_shape = {3, 12, 0, 6};
  test.SetInput("merge", ge::FORMAT_NCHW, "am2", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, dst_zero_shape,
                conv_original_dims);

  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0);

  test.SetInput("am3", "merge");
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}


Status CreateGraphOfZeroShapeOp3(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |(0-shape)  /
   *            \         |          /
   *              \       |         /
   *                 Merge(NCHW)
   *                      |
   *                    am3
   *
   * After format inference:
   *     am1(5HD)     am2(NCHW)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                      |
   *                    am3(5HD)
   */
  vector<int64_t> conv_original_dims = {3, 12, 5, 6};
  ge::GeShape original_shape = GeShape(conv_original_dims);
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1");

  vector<int64_t> dst_zero_shape = {3, 12, 0, 6};
  test.SetInput("merge", ge::FORMAT_NCHW, "am2", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, conv_original_dims,
                dst_zero_shape);

  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0);

  test.SetInput("am3", "merge");
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}


Status CreateGraphOfZeroShapeOp4(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |           /
   *            \         |          /
   *              \       |         /
   *                 Merge(NCHW)
   *                      |(0-shape)
   *                      |
   *                      |
   *                    am3
   *
   * After format inference:
   *     am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                      |
   *                    am3(NCHW)
   */
  vector<int64_t> conv_original_dims = {3, 12, 5, 6};
  ge::GeShape original_shape = GeShape(conv_original_dims);
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2");

  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0);

  vector<int64_t> dst_zero_shape = {3, 12, 0, 6};
  test.SetInput("am3", ge::FORMAT_NCHW, "merge", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, conv_original_dims,
                dst_zero_shape);
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}

Status CreateGraphOfZeroShapeOp5(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     Conv2D(5HD)
   *          \           |           /
   *            \         |          /
   *              \       |         /
   *                 Merge(NCHW)
   *                      |
   *                      |
   *                      |(0-shape)
   *                    am3
   *
   * After format inference:
   *     am1(5HD)     am2(5HD)     Conv2D(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                      |
   *                    am3(NCHW)
   */
  vector<int64_t> conv_original_dims = {3, 12, 5, 6};
  ge::GeShape original_shape = GeShape(conv_original_dims);
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2");
  test.SetInput("merge", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0);

  vector<int64_t> dst_zero_shape = {3, 12, 0, 6};
  test.SetInput("am3", ge::FORMAT_NCHW, "merge", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, dst_zero_shape,
                conv_original_dims);
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}

Status CreateGraphOfTsOp_1(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *    am1(NCHW)     am2(NCHW)     am3(NCHW)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                      |
   *                  Conv2D(5HD)
   *
   * After format inference:
   *     am1(5HD)     am2(5HD)     am3(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                      |
   *                  Conv2D(5HD)
   */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2")
      .SetInput("merge", "am3");

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "merge", ge::FORMAT_NCHW);

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  return fe::SUCCESS;
}

Status CreateGraphOfTsOp_1_1(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *                                only_nch_w(NCHW) _format_agnostic = 1
   *                                    |
   *    am1(NCHW)     am2(NCHW)     am3(NCHW)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                      |
   *                  Conv2D(5HD)
   *
   * After format inference:
   *                               only_nch_w(NCHW) format_agnostic = 1
   *                                    |
   *     am1(5HD)     am2(5HD)     am3(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                      |
   *                  Conv2D(5HD)
   */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("onlyNCHW", "OnlyNCHW", 1, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2")
      .SetInput("merge", "am3");

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "merge", ge::FORMAT_NCHW);

  test.SetInput("am3:0", "onlyNCHW");
  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  ge::NodePtr node2;
  test.GetNodeByName("onlyNCHW", node2);
  ge::AttrUtils::SetInt(node2->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, 1);
  return fe::SUCCESS;
}


Status CreateGraphOfTsOp_1_exception_one_output_edge(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *    am1(NCHW)     am2(NCHW)     am3(NCHW)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                   /     \ (this edge is in the exception of
   *                  /        \ format agonostic)
   *                 /           \
   *            Conv2D(5HD)     am4(NCHW)
   *
   * After format inference:
   *
   *     am1(5HD)     am2(5HD)     am3(5HD)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                   /     \ (this edge is in the exception of
   *                  /        \ format agonostic)
   *                 /           \
   *            Conv2D(5HD)     am4(NCHW)
   */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am4", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2")
      .SetInput("merge", "am3");

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "merge", ge::FORMAT_NCHW);
  test.SetInput("am4:0", "merge:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), OUTPUT_FORMAT_AGNOSTIC_EXCEPTION,
                            {1});
  return fe::SUCCESS;
}


Status CreateGraphOfTsOp_1_exception_one_input_edge(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs. OnlyNCHW will be set _format_agnostic = 1, but it's a
   * tbe op.
   * Graph will be like:
   *
   *    am1(NCHW)     am2(NCHW)     am3(NCHW) (this input edge is in the
   *          \           |            /      exception of format agnostic)
   *            \         |          /
   *              \       |       /
   *                 Merge(NCHW)
   *                   /     \ (this edge is in the exception of
   *                  /        \ format agnostic)
   *                 /           \
   *            Conv2D(5HD)     am4(NCHW)
   *
   * After format inference:
   *
   *     am1(5HD)     am2(5HD)     am3(NCHW)
   *          \           |            /
   *            \         |          /
   *              \       |       /
   *                 Merge(5HD)
   *                   /     \ (this edge is in the exception of
   *                  /        \ format agonostic)
   *                 /           \
   *            Conv2D(5HD)     am4(NCHW)
   */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("merge", "Merge")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am4", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12});

  test.SetInput("merge", "am1")
      .SetInput("merge", "am2")
      .SetInput("merge", "am3");

  test.SetInput("conv2d", ge::FORMAT_NC1HWC0, "merge", ge::FORMAT_NCHW);
  test.SetInput("am4:0", "merge:1");

  GraphConstructor::DumpGraph(graph);
  ge::NodePtr node;

  test.GetNodeByName("merge", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 1);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), OUTPUT_FORMAT_AGNOSTIC_EXCEPTION,
                            {1});
  ge::AttrUtils::SetListInt(node->GetOpDesc(), INPUT_FORMAT_AGNOSTIC_EXCEPTION,
                            {2});
  return fe::SUCCESS;
}

Status CreateGraphOfTsOp_2(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   * Conv2DBackpropFilter(Fz)     am1(NCHW)     Conv2D(5HD)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                        HcomAllReduce(NCHW)
   *                            /   |    \
   *                          /     |     \
   *                        /       |      \
   *                  am2(NCHW)  am3(NCHW)  am4(NCHW)
   *
   * After format inference:
   * Conv2DBackpropFilter(Fz)     am1(NCHW)     Conv2D(5HD)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                        HcomAllReduce(Fz,NCHW,5HD)
   *                            /   |    \
   *                          /     |     \
   *                        /       |      \
   *                  am2(Fz)  am3(NCHW)  am4(5HD)   */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D)
      .AddOpDesc("conv2dback", "Conv2DBackpropFilter")
      .AddOpDesc("hcom", "HcomAllReduce")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am4", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12})

      .SetInput("conv2dback:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2dback:1", "", ge::FORMAT_NC1HWC0);

  test.SetInput("hcom", "conv2dback", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("hcom", "am2");

  test.SetInput("hcom", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0);

  test.SetInput("am2", "hcom:0")
      .SetInput("am3", "hcom:1")
      .SetInput("am4", "hcom:2");

  ge::NodePtr node;
  test.GetNodeByName("hcom", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 2);
  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}


Status CreateGraphOfTsOp_3(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *                am1(NCHW)     am2(NCHW)     am3(NCHW)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                        HcomAllReduce(NCHW)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                  Conv2D(5HD)  am4(NCHW)  Conv2D(Fz)
   *
   * After format inference:
   *                  am1(5HD)     am2(NCHW)      am3(Fz)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                        HcomAllReduce(5HD,NCHW,Fz)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                 Conv2D(5HD)  am4(NCHW)  Conv2D(Fz)  */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D, 3, 1)
      .AddOpDesc("conv2d1", fe::CONV2D, 2, 1)
      .AddOpDesc("hcom", "HcomAllReduce")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am4", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12})

      .SetInput("conv2d1:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d1:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("hcom", "am1")
      .SetInput("hcom", "am2")
      .SetInput("hcom", "am3");

  test.SetInput("conv2d:0", ge::FORMAT_NC1HWC0, "hcom:0", ge::FORMAT_NCHW);
  test.SetInput("am4:0", "hcom:1");
  test.SetInput("conv2d1:1", ge::FORMAT_FRACTAL_Z, "hcom:2", ge::FORMAT_HWCN);

  ge::NodePtr node;
  test.GetNodeByName("hcom", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 2);
  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}


Status CreateGraphOfTsOp_4(ComputeGraphPtr graph) {
  /* In this graph we will create a ts op which is format agnostic for all
   * inputs and outputs.
   * Graph will be like:
   *                am1(ND)     am2(NCHW)     am3(ND)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                        HcomAllReduce(NCHW)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                  Conv2D(5HD)  am4(NCHW)  Conv2D(Fz)
   *
   * After format inference:
   *                  am1(ND)     am2(NCHW)      am3(ND)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                        HcomAllReduce(5HD,NCHW,Fz)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                 Conv2D(5HD)  am4(NCHW)  Conv2D(Fz)  */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("conv2d", fe::CONV2D, 3, 1)
      .AddOpDesc("conv2d1", fe::CONV2D, 2, 1)
      .AddOpDesc("hcom", "HcomAllReduce")
      .AddOpDesc("am1", "ApplyMomentum", 5, 1)
      .AddOpDesc("am2", "ApplyMomentum", 5, 1)
      .AddOpDesc("am3", "ApplyMomentum", 5, 1)
      .AddOpDesc("am4", "ApplyMomentum", 5, 1)
      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12})

      .SetInput("conv2d1:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d1:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("hcom", ge::FORMAT_NCHW, "am1", ge::FORMAT_ND)
      .SetInput("hcom", "am2")
      .SetInput("hcom", ge::FORMAT_NCHW, "am3", ge::FORMAT_ND);

  test.SetInput("conv2d:0", ge::FORMAT_NC1HWC0, "hcom:0", ge::FORMAT_NCHW);
  test.SetInput("am4:0", "hcom:1");
  test.SetInput("conv2d1:1", ge::FORMAT_FRACTAL_Z, "hcom:2", ge::FORMAT_HWCN);

  ge::NodePtr node;
  test.GetNodeByName("hcom", node);
  FE_CHECK_NOTNULL(node);
  ge::AttrUtils::SetInt(node->GetOpDesc(), FORMAT_AGNOSTIC, 2);
  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}

Status CreateGraphOfFcAndRelu(ComputeGraphPtr graph, const string& fcType) {
  /* Relu1-> FC -> Relu2 -> Relu3
   * FC's reshape type is NC */
  ge::GeShape original_shape = GeShape({3, 12, 1, 1});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16, original_shape);
  vector<int64_t> original_dims = {3, 12, 1, 1};
  vector<int64_t> original_dims2 = {3, 1, 1, 1, 16};
  test.AddOpDesc("relu1", "Relu6", 1, 1)
      .SetInputs({"DATA_1"})

      .AddOpDesc("fc", fcType, 1, 1)
      .SetInput("fc:0", ge::FORMAT_NC1HWC0, "relu1:0", ge::FORMAT_NCHW)

      .AddOpDesc("relu2", "Relu6", 1, 1)

      .AddOpDesc("relu3", "Relu6", 1, 1)

      .AddOpDesc("out", "NetOutput", 1, 1);
  test.SetInput("relu2", ge::FORMAT_NCHW, "fc", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims, original_dims2);
  test.SetInput("relu3", ge::FORMAT_NCHW, "relu2", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims, original_dims);
  test.SetInput("out", ge::FORMAT_NCHW, "relu3", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims, original_dims);
  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}

Status CreateGraphOfFcAndRelu2(ComputeGraphPtr graph, const string& fcType) {
  /* Relu1-> FC -> Relu2 -> Relu3
   * FC's reshape type is NC */
  ge::GeShape original_shape = GeShape({1, 3, 12, 1});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16, original_shape);
  vector<int64_t> original_dims1 = {1, 3, 12, 1};
  vector<int64_t> original_dims2 = {3, 12, 1, 1};
  vector<int64_t> original_dims3 = {3, 1, 1, 1, 16};

  test.AddOpDesc("relu1", "Relu6", 1, 1)
      .SetInputs({"DATA_1"})

      .AddOpDesc("fc", fcType, 1, 1)
      .SetInput("fc:0", ge::FORMAT_NC1HWC0, "relu1:0", ge::FORMAT_NCHW)

      .AddOpDesc("relu2", "Relu6", 1, 1)

      .AddOpDesc("relu3", "Relu6", 1, 1)

      .AddOpDesc("out", "NetOutput", 1, 1);
  test.SetInput("relu2", ge::FORMAT_NCHW, "fc", ge::FORMAT_NC1HWC0,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims2, original_dims3);
  test.SetInput("relu3", ge::FORMAT_NCHW, "relu2", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims2, original_dims2);
  test.SetInput("out", ge::FORMAT_NCHW, "relu3", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims2, original_dims2);

  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}

Status CreateGraphOfFcAndRelu3(ComputeGraphPtr graph, const string& fcType) {
  /* Relu1-> FC -> Relu2 -> NOT_SUPPORT_NC -> Relu3
   * FC's reshape type is NC and there is a node
   * which is not support reshape type nc between relu2 and relu3. */
  ge::GeShape original_shape = GeShape({3, 12, 1, 1});
  vector<int64_t> original_dims1 = {1, 3, 12, 1};
  vector<int64_t> original_dims2 = {3, 12, 1, 1};
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16, original_shape);

  test.AddOpDesc("relu1", "Relu6", 1, 1)
      .SetInputs({"DATA_1"})

      .AddOpDesc("fc", fcType, 1, 1)
      .SetInput("fc:0", ge::FORMAT_NC1HWC0, "relu1:0", ge::FORMAT_NCHW)

      .AddOpDesc("relu2", "Relu6", 1, 1)
      .SetInput("relu2:0", ge::FORMAT_NCHW, "fc:0", ge::FORMAT_NC1HWC0)

      .AddOpDesc("reluNotSupportNC", "ReluNotSupportNc", 1, 1)

      .AddOpDesc("relu3", "Relu6", 1, 1)

      .AddOpDesc("out", "NetOutput", 1, 1);
  test.SetInput("reluNotSupportNC", ge::FORMAT_NCHW, "relu2", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims1, original_dims2);
  test.SetInput("relu3", ge::FORMAT_NCHW, "reluNotSupportNC", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims1, original_dims1);
  test.SetInput("out", ge::FORMAT_NCHW, "relu3", ge::FORMAT_NCHW,
                ge::FORMAT_NCHW, ge::FORMAT_NCHW, original_dims1, original_dims1);

  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}

Status CreateGraphOfFcAndRelu4(ComputeGraphPtr graph, const string& fcType) {
  /* Relu1-> FC -> Relu2 -> Relu3
   * FC's reshape type is NC */
  ge::GeShape original_shape = GeShape({3, 12});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16, original_shape);

  test.AddOpDesc("relu1", "Relu6", 1, 1)
      .SetInputs({"DATA_1"})

      .AddOpDesc("fc", fcType, 1, 1)
      .SetInput("fc:0", ge::FORMAT_NC1HWC0, "relu1:0", ge::FORMAT_NCHW)

      .AddOpDesc("relu2", "Relu6", 1, 1)
      .SetInput("relu2:0", ge::FORMAT_NCHW, "fc:0", ge::FORMAT_NC1HWC0)

      .AddOpDesc("relu3", "Relu6", 1, 1)
      .SetInputs({"relu2:0"})

      .AddOpDesc("out", "NetOutput", 1, 1)
      .SetInputs({"relu3"});

  GraphConstructor::DumpGraph(graph);
  return fe::SUCCESS;
}

Status CreateGraphOfFunctionOp1(ComputeGraphPtr graph) {
  /* In this graph we will create a function op case
   * which have three inputs and three outputs.
   * Graph will be like:
   *                am1(NCHW)     am2(NCHW)     am3(NCHW)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                              Case(NCHW,NCHW,NCHW)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                  Conv2D(5HD)  am4(NCHW)  Conv2D(Fz)
   *
   * Inside the case operator, the subgraph look like:
   *                      Data    Data   Data
   *                         \     |     /
   *                          \    |    /
   *                           \   |   /
   *                            \  |  /
   *                            X  X X (three formatAgnosticOp)
   *                             / | \
   *                           NetOutput
   *
   *
   * After format inference:
   *                  am1(5HD)     am2(NCHW)      am3(Fz)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                             Case(5HD,NCHW,Fz)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                 Conv2D(5HD)  am4(NCHW)  Conv2D(Fz)  */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  string fa = "formatAgnosticOp";
  test.AddOpDesc("conv2d", fe::CONV2D, 3, 1)
      .AddOpDesc("conv2d1", fe::CONV2D, 2, 1)
      .AddOpDesc("x1", fa)
      .AddOpDesc("x2", fa)
      .AddOpDesc("x3", fa)
      .AddOpDesc("am1", fa, 1, 1)
      .AddOpDesc("am2", fa, 1, 1)
      .AddOpDesc("am3", fa, 1, 1)
      .AddOpDesc("am4", fa, 1, 1)

      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12})

      .SetInput("conv2d1:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d1:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("x1:0", "am1:0")
      .SetInput("x2", "am2")
      .SetInput("x3", "am3");

  test.SetInput("conv2d:0", ge::FORMAT_NC1HWC0, "x1:0", ge::FORMAT_NCHW);
  test.SetInput("am4:0", "x2:0");
  test.SetInput("conv2d1:1", ge::FORMAT_FRACTAL_Z, "x3:0", ge::FORMAT_HWCN);
  ge::NodePtr x1;
  ge::NodePtr x2;
  ge::NodePtr x3;
  test.GetNodeByName("x1", x1);
  test.GetNodeByName("x2", x2);
  test.GetNodeByName("x3", x3);
  vector<NodePtr> nodes = {x1, x2, x3};
  std::shared_ptr<GraphComm> graph_comm_ptr = nullptr;
  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<GraphComm>("ffts_plus"),
                 return fe::FAILED);
  FE_CHECK(graph_comm_ptr == nullptr, FE_LOGE("graphCommPtr is nullptr."), return fe::FAILED);
  test.graph_comm_ptr_ = graph_comm_ptr;
  graph->SetExtAttr("part_src_graph", graph);
  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  test.TransSingleSubGraph(*graph.get(), nodes);

  GraphConstructor::DumpGraph(graph);
  GraphConstructor::DumpGraph(graph->GetSubgraph("test_sgt_graph_0"));
  return fe::SUCCESS;
}


Status CreateGraphOfFunctionOp2(ComputeGraphPtr graph) {
  /* In this graph we will create a function op case
   * which have three inputs and three outputs.
   * Graph will be like:
   *                Conv2D(5HD)    am4(NCHW)     Conv2D(Fz)
   *                     \          |            /
   *                      \         |          /
   *                       \        |        /
   *                              Case(NCHW,NCHW,NCHW)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                  am1(NCHW)  am2(NCHW)  am3(NCHW)
   *
   *
   * Inside the case operator, the subgraph look like:
   *                      Data    Data   Data
   *                         \     |     /
   *                          \    |    /
   *                           \   |   /
   *                            \  |  /
   *                            X  X X (three formatAgnosticOp)
   *                             / | \
   *                           NetOutput
   *
   *
   * After format inference:
   *                Conv2D(5HD)   am4(NCHW)     Conv2D(Fz)
   *                     \          |            /
   *                      \         |          /
   *                       \        |        /
   *                             Case(5HD,NCHW,Fz)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                 am1(5HD)  am2(NCHW)  am3(Fz)
   */

  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  string fa = "formatAgnosticOp";
  test.AddOpDesc("conv2d", fe::CONV2D, 3, 1)
      .AddOpDesc("conv2d1", fe::CONV2D, 2, 1)
      .AddOpDesc("x1", fa)
      .AddOpDesc("x2", fa)
      .AddOpDesc("x3", fa)
      .AddOpDesc("am1", fa, 1, 1)
      .AddOpDesc("am2", fa, 1, 1)
      .AddOpDesc("am3", fa, 1, 1)
      .AddOpDesc("am4", fa, 1, 1)

      .SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z, ge::FORMAT_HWCN)
      .SetInput("conv2d:2", "", {12})

      .SetInput("conv2d1:0", "", ge::FORMAT_NC1HWC0)
      .SetInput("conv2d1:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("x1:0", ge::FORMAT_NCHW, "conv2d:0", ge::FORMAT_NC1HWC0)
      .SetInput("x2", "am4")
      .SetInput("x3", ge::FORMAT_HWCN, "conv2d1:0", ge::FORMAT_FRACTAL_Z);

  test.SetInput("am1:0", "x1:0");
  test.SetInput("am2:0", "x2:0");
  test.SetInput("am3:0", "x3:0");
  ge::NodePtr x1;
  ge::NodePtr x2;
  ge::NodePtr x3;
  test.GetNodeByName("x1", x1);
  test.GetNodeByName("x2", x2);
  test.GetNodeByName("x3", x3);
  vector<NodePtr> nodes = {x1, x2, x3};
  std::shared_ptr<GraphComm> graph_comm_ptr = nullptr;
  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<GraphComm>("ffts_plus"),
                 return fe::FAILED);
  FE_CHECK(graph_comm_ptr == nullptr, FE_LOGE("graphCommPtr is nullptr."), return fe::FAILED);
  test.graph_comm_ptr_ = graph_comm_ptr;
  graph->SetExtAttr("part_src_graph", graph);
  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  test.TransSingleSubGraph(*graph.get(), nodes);

  GraphConstructor::DumpGraph(graph);
  GraphConstructor::DumpGraph(graph->GetSubgraph("test_sgt_graph_0"));
  return fe::SUCCESS;
}

Status CreateGraphOfFunctionOp3(ComputeGraphPtr graph) {
  /* In this graph we will create a function op case
   * which have three inputs and three outputs.
   * Graph will be like:
   *                am1(NCHW)     am2(NCHW)     am3(NCHW)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                              Case(NCHW,NCHW,NCHW)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                  am4(NCHW)  am5(NCHW)  am6(NCHW)
   *
   * Inside the case operator, the subgraph look like:
   *                      Data    Data   Data
   *                         \     |     /
   *                          \    |   conv2d(5HD)
   *                           \   |   /
   *                            \  |  /
   *                            X  X X (three formatAgnosticOp)
   *                             / | \
   *                           NetOutput
   *
   *
   * After format inference:
   *                  am1(NCHW)     am2(NCHW)      am3(5HD)
   *                     \           |            /
   *                      \         |          /
   *                        \       |        /
   *                             Case(NCHW,NCHW,5HD)
   *                            /   |     \
   *                          /     |      \
   *                        /       |       \
   *                 am4(NCHW)  am5(NCHW)  am6(5HD)  */
  ge::GeShape original_shape = GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  string fa = "formatAgnosticOp";
  test.AddOpDesc("conv2d", fe::CONV2D, 2, 1)
      .AddOpDesc("x1", fa)
      .AddOpDesc("x2", fa)
      .AddOpDesc("x3", fa)
      .AddOpDesc("am1", fa, 1, 1)
      .AddOpDesc("am2", fa, 1, 1)
      .AddOpDesc("am3", fa, 1, 1)
      .AddOpDesc("am4", fa, 1, 1)
      .AddOpDesc("am5", fa, 1, 1)
      .AddOpDesc("am6", fa, 1, 1)

      .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z);

  test.SetInput("x1:0", "am1:0")
      .SetInput("x2", "am2");

  test.SetInput("conv2d:0", ge::FORMAT_NC1HWC0, "am3:0", ge::FORMAT_NCHW);
  test.SetInput("x3:0", ge::FORMAT_NCHW, "conv2d:0", ge::FORMAT_NC1HWC0);
  test.SetInput("am4:0", "x1:0");
  test.SetInput("am5:0", "x2:0");
  test.SetInput("am6:0", ge::FORMAT_NCHW, "x3:0", ge::FORMAT_NCHW);
  ge::NodePtr x1;
  ge::NodePtr x2;
  ge::NodePtr x3;
  ge::NodePtr conv2d;
  test.GetNodeByName("x1", x1);
  test.GetNodeByName("x2", x2);
  test.GetNodeByName("x3", x3);
  test.GetNodeByName("conv2d", conv2d);
  vector<NodePtr> nodes = {x1, x2, x3, conv2d};
  std::shared_ptr<GraphComm> graph_comm_ptr = nullptr;
  FE_MAKE_SHARED(graph_comm_ptr = std::make_shared<GraphComm>("ffts_plus"),
                 return fe::FAILED);
  FE_CHECK(graph_comm_ptr == nullptr, FE_LOGE("graphCommPtr is nullptr."), return fe::FAILED);
  test.graph_comm_ptr_ = graph_comm_ptr;
  graph->SetExtAttr("part_src_graph", graph);
  ge::AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "0_1");
  test.TransSingleSubGraph(*graph.get(), nodes);

  GraphConstructor::DumpGraph(graph);
  GraphConstructor::DumpGraph(graph->GetSubgraph("test_sgt_graph_0"));
  return fe::SUCCESS;
}


TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_all_inputs_and_outputs) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_all_inputs_and_outputs_1) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_1(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2" ||
        node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }

    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_ND);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}


TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_all_inputs_and_outputs_1_with_input_exception) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_1_exception_one_input_edge(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetName() == "am3" ||
        node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      auto count = 0;
      for (auto& input : opdesc->GetAllInputsDesc()) {
        if (count == 2) {
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        } else {
          EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        }
        count++;
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto output =opdesc->GetOutputDesc(1);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_ND);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}


TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_all_inputs_and_outputs_1_with_output_exception) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_1_exception_one_output_edge(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2" ||
        node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }

    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto output =opdesc->GetOutputDesc(1);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_ND);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_all_inputs_and_outputs_2) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_1_1(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2" ||
        node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetName() == "onlyNCHW") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_ND);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_paired_input_and_output) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_2(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {30, 1, 16, 16};
  vector<int64_t> dims_fz_conv2_d = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
        if (i == 0) {
          auto input =opdesc->GetInputDesc(i);
          EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        }
        if (i == 1 || i == 2 || i == 3 || i == 4) {
          auto input =opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        }
      }
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
        if (i == 0) {
          auto input =opdesc->GetInputDesc(i);
          EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
        }
        if (i == 1 || i == 2 || i == 3 || i == 4) {
          auto input =opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        }
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
      }
    }
    if (node->GetName() == "am1" ||
        node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
        auto input =opdesc->GetInputDesc(i);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "HcomAllReduce") {
      auto opdesc = node->GetOpDesc();
      auto input0 = opdesc->GetInputDesc(0);
      auto input1 = opdesc->GetInputDesc(1);
      auto input2 = opdesc->GetInputDesc(2);
      EXPECT_EQ(ge::GetPrimaryFormat(input0.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(input0.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input0.GetShape().GetDims(), dims_fz_conv2_d);

      EXPECT_EQ(input1.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(input1.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input1.GetShape().GetDims(), dims4_d);

      EXPECT_EQ(ge::GetPrimaryFormat(input2.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(input2.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input2.GetShape().GetDims(), dims5_h_d);

      auto output0 = opdesc->GetInputDesc(0);
      auto output1 = opdesc->GetInputDesc(1);
      auto output2 = opdesc->GetInputDesc(2);
      EXPECT_EQ(ge::GetPrimaryFormat(output0.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(output0.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output0.GetShape().GetDims(), dims_fz_conv2_d);

      EXPECT_EQ(output1.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output1.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output1.GetShape().GetDims(), dims4_d);

      EXPECT_EQ(ge::GetPrimaryFormat(output2.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output2.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output2.GetShape().GetDims(), dims5_h_d);
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz_conv2_d);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_paired_input_and_output_2) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_3(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {30, 1, 16, 16};
  vector<int64_t> dims_fz_conv2_d = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for (auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
        auto input = opdesc->GetInputDesc(i);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
      }
      if (node->GetName() == "am2") {
        auto opdesc = node->GetOpDesc();
        for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
          auto input = opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
        }
      }
      if (node->GetName() == "am1") {
        auto opdesc = node->GetOpDesc();
        for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
          auto input = opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
        }
      }
      if (node->GetName() == "am3") {
        auto opdesc = node->GetOpDesc();
        for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
          auto input = opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
        }
      }
      if (node->GetType() == "HcomAllReduce") {
        auto opdesc = node->GetOpDesc();
        auto input0 = opdesc->GetInputDesc(0);
        auto input1 = opdesc->GetInputDesc(1);
        auto input2 = opdesc->GetInputDesc(2);
        EXPECT_EQ(input0.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input0.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input0.GetShape().GetDims(), dims5_h_d);

        EXPECT_EQ(input1.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input1.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input1.GetShape().GetDims(), dims4_d);

        EXPECT_EQ(input2.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input2.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input2.GetShape().GetDims(), dims_fz_conv2_d);

        auto output0 = opdesc->GetInputDesc(0);
        auto output1 = opdesc->GetInputDesc(1);
        auto output2 = opdesc->GetInputDesc(2);
        EXPECT_EQ(output0.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output0.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output0.GetShape().GetDims(), dims5_h_d);

        EXPECT_EQ(output1.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output1.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output1.GetShape().GetDims(), dims4_d);

        EXPECT_EQ(output2.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(output2.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output2.GetShape().GetDims(), dims_fz_conv2_d);
      }
      if (node->GetType() == "Conv2D") {
        auto opdesc = node->GetOpDesc();
        {
          auto input = opdesc->GetInputDesc(0);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        }
        {
          auto input = opdesc->GetInputDesc(1);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims_fz_conv2_d);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        }
      }
    }
  }
}


TEST_F(UTestHeavyFormatDistributionTsOp,
       format_agnostic_for_paired_input_and_output_3) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfTsOp_4(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_fz = {30, 1, 16, 16};
  vector<int64_t> dims_fz_conv2_d = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for (auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
        auto input = opdesc->GetInputDesc(i);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
      }
      if (node->GetName() == "am2") {
        auto opdesc = node->GetOpDesc();
        for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
          auto input = opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
        }
      }
      if (node->GetName() == "am1") {
        auto opdesc = node->GetOpDesc();
        for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
          auto input = opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
        }
      }
      if (node->GetName() == "am3") {
        auto opdesc = node->GetOpDesc();
        for (size_t i = 0; i < opdesc->GetAllInputsDesc().size(); i++) {
          auto input = opdesc->GetInputDesc(i);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
        }
      }
      if (node->GetType() == "HcomAllReduce") {
        auto opdesc = node->GetOpDesc();
        auto input0 = opdesc->GetInputDesc(0);
        auto input1 = opdesc->GetInputDesc(1);
        auto input2 = opdesc->GetInputDesc(2);
        EXPECT_EQ(input0.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input0.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input0.GetShape().GetDims(), dims5_h_d);

        EXPECT_EQ(input1.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input1.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input1.GetShape().GetDims(), dims4_d);

        EXPECT_EQ(input2.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input2.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input2.GetShape().GetDims(), dims_fz_conv2_d);

        auto output0 = opdesc->GetInputDesc(0);
        auto output1 = opdesc->GetInputDesc(1);
        auto output2 = opdesc->GetInputDesc(2);
        EXPECT_EQ(output0.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output0.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output0.GetShape().GetDims(), dims5_h_d);

        EXPECT_EQ(output1.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output1.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output1.GetShape().GetDims(), dims4_d);

        EXPECT_EQ(output2.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(output2.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output2.GetShape().GetDims(), dims_fz_conv2_d);
      }
      if (node->GetType() == "Conv2D") {
        auto opdesc = node->GetOpDesc();
        {
          auto input = opdesc->GetInputDesc(0);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        }
        {
          auto input = opdesc->GetInputDesc(1);
          EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
          EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
          EXPECT_EQ(input.GetShape().GetDims(), dims_fz_conv2_d);
        }
        {
          auto output = opdesc->GetOutputDesc(0);
          EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
          EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        }
      }
    }
  }
}


TEST_F(UTestHeavyFormatDistributionTsOp,
       fc_and_relu) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFcAndRelu(graph, "FullyConnection");
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims_5hd_nc = {3, 1, 1, 1, 16};
  for (auto node : graph->GetDirectNode()) {
    if (node->GetName() == "relu1") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }

    if (node->GetName() == "relu2") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }

    if (node->GetName() == "relu3") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       fc_and_relu_2) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFcAndRelu2(graph, "FullyConnection2");
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims_5hd_nc = {3, 1, 1, 1, 16};
  vector<int64_t> dims_5hd_ch = {1, 1, 12, 1, 16};
  for (auto node : graph->GetDirectNode()) {
    if (node->GetName() == "relu1") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_ch);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_ch);
      }
    }

    if (node->GetName() == "relu2") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }

    if (node->GetName() == "relu3") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       fc_and_relu_3) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFcAndRelu3(graph, "FullyConnection");
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims_5hd_nc = {3, 1, 1, 1, 16};
  vector<int64_t> dims_5hd_ch = {1, 1, 12,1, 16};
  for (auto node : graph->GetDirectNode()) {
    if (node->GetName() == "relu1") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }

    if (node->GetName() == "relu2") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_nc);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_nc);
      }
    }

    if (node->GetName() == "reluNotSupportNC") {
      auto opdesc = node->GetOpDesc();
      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_ch);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_ch);

    }

    if (node->GetName() == "relu3") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_5hd_ch);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_5hd_ch);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       fc_and_relu_4) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFcAndRelu4(graph, "FullyConnection");
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims_nchw = {3, 1, 1, 1, 16};
  for (auto node : graph->GetDirectNode()) {
    if (node->GetName() == "relu1") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      }
    }

    if (node->GetName() == "relu2") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      }
    }

    if (node->GetName() == "relu3") {
      auto opdesc = node->GetOpDesc();
      {
        auto input = opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);
      }
      {
        auto output = opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      }
    }
  }
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       function_op_01) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFunctionOp1(graph);
  reflection_builder_ptr_->Clear();
  auto status = reflection_builder_ptr_->BuildRefRelations(*graph.get());

  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);
  vector<int64_t> dims_5hd = {3, 1, 5, 6, 16};
  vector<int64_t> dims_fz = {30, 1, 16, 16};
  vector<int64_t> dims_nchw = {3, 12, 5, 6};
  int count = 0;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "test_sgt_graph_0/x1") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_5hd);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }
    if (node->GetName() == "test_sgt_graph_0/x2" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }

    if (node->GetName() == "test_sgt_graph_0/x3") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_fz);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
      count++;
    }

    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
      count++;
    }

    if (node->GetName() == "x1x2x3") {
      count++;
    }
  }
  EXPECT_EQ(count, 7);
}


TEST_F(UTestHeavyFormatDistributionTsOp,
       function_op_03) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFunctionOp3(graph);
  reflection_builder_ptr_->Clear();
  auto status = reflection_builder_ptr_->BuildRefRelations(*graph.get());

  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);
  vector<int64_t> dims_5hd = {3, 1, 5, 6, 16};
  vector<int64_t> dims_fz = {30, 1, 16, 16};
  vector<int64_t> dims_nchw = {3, 12, 5, 6};
  int count = 0;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetOpDesc() == nullptr) {
      continue;
    }
    if (node->GetName() == "test_sgt_graph_0/x1") {
      auto opdesc = node->GetOpDesc();
      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }

    if (node->GetName() == "test_sgt_graph_0/x2" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }

    if (node->GetName() == "test_sgt_graph_0/x3") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_5hd);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }

    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }
    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }
    if (node->GetName() == "am5") {
      auto opdesc = node->GetOpDesc();
      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }
    if (node->GetName() == "am6") {
      auto opdesc = node->GetOpDesc();
      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }
    if (node->GetName() == "x1x2x3conv2d") {
      count++;
    }
  }
  EXPECT_EQ(count, 10);
}

TEST_F(UTestHeavyFormatDistributionTsOp,
       function_op_02) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfFunctionOp2(graph);
  reflection_builder_ptr_->Clear();
  auto status = reflection_builder_ptr_->BuildRefRelations(*graph.get());

  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(ret, fe::SUCCESS);
  vector<int64_t> dims_5hd = {3, 1, 5, 6, 16};
  vector<int64_t> dims_fz = {30, 1, 16, 16};
  vector<int64_t> dims_nchw = {3, 12, 5, 6};
  int count = 0;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "test_sgt_graph_0/x1") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_5hd);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_5hd);
      count++;
    }
    if (node->GetName() == "test_sgt_graph_0/x2" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_nchw);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_nchw);
      count++;
    }

    if (node->GetName() == "test_sgt_graph_0/x3") {
      auto opdesc = node->GetOpDesc();

      auto input = opdesc->GetInputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(input.GetShape().GetDims(), dims_fz);

      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
      count++;
    }

    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      auto output = opdesc->GetOutputDesc(0);
      EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_FRACTAL_Z);
      EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      EXPECT_EQ(output.GetShape().GetDims(), dims_fz);
      count++;
    }

    if (node->GetName() == "x1x2x3") {
      count++;
    }
  }
  EXPECT_EQ(count, 7);
}

/* zero shape op will remain origin shape and format. */
TEST_F(UTestHeavyFormatDistributionTsOp,
       zero_shape) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfZeroShapeOp1(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_zero = {3, 12, 0, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetInputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      int index = 0;
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        if (index == 2) {
          EXPECT_EQ(input.GetShape().GetDims(), dims_zero);
        } else {
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        }
        index++;
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

/* zero shape op will remain origin shape and format. */
TEST_F(UTestHeavyFormatDistributionTsOp,
       zero_shape_02) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfZeroShapeOp2(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_zero = {3, 12, 0, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetInputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      int index = 0;
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        if (index == 1) {
          EXPECT_EQ(input.GetShape().GetDims(), dims_zero);
        } else {
          EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        }
        index++;
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}


/* zero shape op will remain origin shape and format. */
TEST_F(UTestHeavyFormatDistributionTsOp,
       zero_shape_03) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfZeroShapeOp3(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_zero = {3, 12, 0, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims_zero);
      }
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      int index = 0;
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        index++;
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

/* zero shape op will remain origin shape and format. */
TEST_F(UTestHeavyFormatDistributionTsOp,
       zero_shape_04) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfZeroShapeOp4(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_zero = {3, 12, 0, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1" ||
        node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetInputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims4_d);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      int index = 0;
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims4_d);
        index++;
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims_zero);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}

/* zero shape op will remain origin shape and format. */
TEST_F(UTestHeavyFormatDistributionTsOp,
       zero_shape_05) {
  HeavyFormatPropagationPtr HeavyFormatPropagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOfZeroShapeOp5(graph);
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> dims5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> dims4_d = {3, 12, 5, 6};
  vector<int64_t> dims_zero = {3, 12, 0, 6};
  vector<int64_t> dims_fz = {36, 1, 16, 16};
  vector<int64_t> dims_bias = {12};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      {
        auto output =opdesc->GetInputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims_zero);
      }
    }
    if (node->GetType() == "Merge") {
      auto opdesc = node->GetOpDesc();
      int index = 0;
      for (auto& input : opdesc->GetAllInputsDesc()) {
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
        index++;
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(output.GetShape().GetDims(), dims5_h_d);
      }
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims5_h_d);
      }
      {
        auto input =opdesc->GetInputDesc(1);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_FRACTAL_Z);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_fz);
      }
      {
        auto input =opdesc->GetInputDesc(2);
        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(input.GetShape().GetDims(), dims_bias);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(output.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT);
      }
    }
  }
}
