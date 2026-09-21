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
#include <map>
#include <memory>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "proto/om.pb.h"

#define protected public
#define private public
#include "common/graph_comm.h"
#include "common/scope_allocator.h"
#include "graph/types.h"
#include "pass_manager.h"
#include "common/configuration.h"
#include "graph/compute_graph.h"
#include "common/util/op_info_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/ub_fusion/automatic_buffer_fusion.h"
#include "../../../graph_constructor/graph_constructor.h"

#undef protected
#undef private
using namespace std;
using namespace domi;
using namespace fe;

class UB_FUSION_ST_AUTO_FUSION : public testing::Test {
 public:
 protected:

  static void SetUpTestCase() { std::cout << "UB fusion SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "UB fusion TearDown" << std::endl;
  }

  virtual void SetUp() {
    graph_comm_ptr_ = std::make_shared<GraphComm>("engineName");
    graph_comm_ptr_->Initialize();
    sub_graph_optimizer_ptr_ =
        std::make_shared<BufferFusion>(graph_comm_ptr_, nullptr, nullptr);
    auto_buffer_fusion_ptr_ = std::make_shared<AutomaticBufferFusion>(nullptr);
  }

  virtual void TearDown() {

 }

  std::shared_ptr<GraphComm> graph_comm_ptr_;
  std::shared_ptr<AutomaticBufferFusion> auto_buffer_fusion_ptr_;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr_;

  void BuildGraph_01(ge::ComputeGraphPtr &graph) {
    /* add1 -> add2 */
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
        original_shape);
    string ADD = "Add";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, 0)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, 0)
        .SetInput("add1:0", "Data_1")
        .SetInput("add1:1", "Data_2")
        .SetInput("add2:0", "Data_3")
        .SetInput("add2:1", "add1:0")
        .SetInput("NetOutput", "add2:0");
  }

  /* add1 -> add2
   *     \-> sqrt1
   *     \-> sqrt2
   *     \-> sqrt3 -> sqrt4-> conv2d -> sqrt5
   *     */
  void BuildGraph_02(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .SetInput("add1:0", "Data_1")
        .SetInput("add1:1", "Data_2")
        .SetInput("add2:0", "Data_3")
        .SetInput("add2:1", "add1:0")
        .SetInput("sqrt1", "add1:0")
        .SetInput("sqrt2", "add1:0")
        .SetInput("sqrt3", "add1:0")
        .SetInput("sqrt4", "sqrt3")
        .SetInput("conv2d", "sqrt4")
        .SetInput("conv2d:1", "Data_4")
        .SetInput("sqrt5", "conv2d")
        .SetInput("NetOutput", "sqrt5:0");

    test.DumpGraph(graph);
  }

    /* add1 -> add2
     *     \-> sqrt1
     *     \-> sqrt2
     *     \-> sqrt3 -> sqrt4-> conv2d -> sqrt5
     *     */
    void BuildGraph_Long_Name(ge::ComputeGraphPtr &graph) {
        ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
        GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                              original_shape);
        string ADD = "Add";
        string SQRT = "Sqrt";
        string one_hundred_one;
        string one_hundred_two;
        string one_hundred_three;
        string one_hundred_four;
        for (int i = 0; i < 100; i++) {
            one_hundred_one += "1";
            one_hundred_two += "2";
            one_hundred_three += "3";
            one_hundred_four += "4";
        }
        test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1"+one_hundred_one, ADD, 2, 1)
            .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2"+one_hundred_two, ADD, 2, 1)
            .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1"+one_hundred_one, SQRT, 1, 1)
            .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2"+one_hundred_two, SQRT, 1, 1)
            .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3"+one_hundred_three, SQRT, 1, 1)
            .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4"+one_hundred_four, SQRT, 1, 1)
            .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
            .SetInput("add1:0", "Data_1")
            .SetInput("add1:1", "Data_2")
            .SetInput("add2:0", "Data_3")
            .SetInput("add2:1", "add1:0")
            .SetInput("sqrt1", "add1:0")
            .SetInput("sqrt2", "add1:0")
            .SetInput("sqrt3", "add1:0")
            .SetInput("sqrt4", "sqrt3")
            .SetInput("conv2d", "sqrt4")
            .SetInput("conv2d:1", "Data_4")
            .SetInput("sqrt5", "conv2d")
            .SetInput("NetOutput", "sqrt5:0");

        test.DumpGraph(graph);
    }

  /* add1 -> add2
   *     \-> sqrt1
   *     \-> sqrt2
   *     \-> sqrt3 -> sqrt4-> conv2d -> sqrt5
   *     \-> conv2d1
   *     \-> conv2d2 */
  void BuildGraph_03(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d1", fe::CONV2D, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d2", fe::CONV2D, 2, 1)
        .SetInput("add1:0", "Data_1")
        .SetInput("add1:1", "Data_2")
        .SetInput("add2:0", "Data_3")
        .SetInput("add2:1", "add1:0")
        .SetInput("sqrt1", "add1:0")
        .SetInput("sqrt2", "add1:0")
        .SetInput("sqrt3", "add1:0")
        .SetInput("sqrt4", "sqrt3")
        .SetInput("conv2d", "sqrt4")
        .SetInput("conv2d:1", "Data_4")
        .SetInput("sqrt5", "conv2d")

        .SetInput("conv2d1:0", "add1")
        .SetInput("conv2d1:1", "add1")
        .SetInput("conv2d2:0", "add1")
        .SetInput("conv2d2:1", "Data_5")

        .SetInput("NetOutput", "sqrt5:0")
        .SetInput("NetOutput", "conv2d1")
        .SetInput("NetOutput", "conv2d2");
  }

  /* add1 -> add2
   *     \-> sqrt1
   *     \-> sqrt2
   *     \-> sqrt3 -> add3-> conv2d -> sqrt5
   *     \-> conv2d1 -> /
   *     \-> conv2d2 */
  /* Cannot fuse add3 with add 1.
   * Can fuse add1, add2, sqrt1, sqrt2, sqrt3. */
  void BuildGraph_04(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add3", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d1", fe::CONV2D, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d2", fe::CONV2D, 2, 1)
        .SetInput("add1:0", "Data_1")
        .SetInput("add1:1", "Data_2")
        .SetInput("add2:0", "Data_3")
        .SetInput("add2:1", "add1:0")
        .SetInput("sqrt1", "add1:0")
        .SetInput("sqrt2", "add1:0")
        .SetInput("sqrt3", "add1:0")
        .SetInput("add3", "sqrt3")
        .SetInput("conv2d", "add3")
        .SetInput("conv2d:1", "Data_4")
        .SetInput("sqrt5", "conv2d")

        .SetInput("conv2d1:0", "add1")
        .SetInput("conv2d1:1", "add1")
        .SetInput("conv2d2:0", "add1")
        .SetInput("conv2d2:1", "Data_5")

        .SetInput("add3:1", "conv2d1")

        .SetInput("NetOutput", "sqrt5:0")
        .SetInput("NetOutput", "conv2d2");
  }


  /* add1 -> add2
   *     \-> sqrt1
   *     \-> sqrt2
   *     \-> sqrt3 -> sqrt4-> conv2d -> sqrt5 -> sqrt6
   *     \-> conv2d1   ------> /
   *     \-> conv2d2 */
  /* Cannot fuse sqrt5 with add 1, sqrt3 and sqrt4.
   * Can fuse add1, add2, sqrt1, sqrt2, sqrt3, sqrt4*/
  void BuildGraph_05(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt5", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt6", SQRT, 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d1", fe::CONV2D, 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d2", fe::CONV2D, 2, 1)
        .SetInputs("add1", {"Data_1", "Data_2"})
        .SetInputs("add2", {"Data_3", "add1:0"})
        .SetInput("sqrt1", "add1:0")
        .SetInput("sqrt2", "add1:0")
        .SetInput("sqrt3", "add1:0")
        .SetInput("sqrt4", "sqrt3")
        .SetInputs("conv2d", {"sqrt4"})
        .SetInput("sqrt5", "conv2d")
        .SetInput("sqrt6", "sqrt5")
        .SetInput("conv2d:1", "conv2d1")

        .SetInputs("conv2d1", {"add1", "add1"})

        .SetInputs("conv2d2", {"add1", "Data_5"})

        .SetInputs("NetOutput", {"sqrt6:0", "conv2d2"});
    test.DumpGraph(graph);
  }

  /* add1 -> add2(2 inputs are the same)->add3
   *
   *     */
  void BuildGraph_06(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetInputs({"Data_1", "Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .SetInputs({"add1", "add1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add3", ADD, 1, 1)
        .SetInputs({"add2", "add2"});
    test.DumpGraph(graph);
  }

  /* sqrt1 -> sqrt2 -> add3
   *    \---------------/
   *     */
  void BuildGraph_07(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1", "Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", ADD, 2, 1)
        .SetInputs({"sqrt1", "sqrt2"});
  }


  /* add1 -> sqrt1 -> sqrt3 -> sqrt5 -> Conv
   *      \- sqrt2 -> sqrt4 -> sqrt6 ---/
   *     */
  void BuildGraph_08(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetInputs({"Data_1", "Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"add1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"add1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1)
        .SetInputs({"sqrt2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt5", SQRT, 1, 1)
        .SetInputs({"sqrt3"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt6", SQRT, 1, 1)
        .SetInputs({"sqrt4"})

        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .SetInputs({"sqrt5", "sqrt6"})
        .SetInput("NetOutput", "conv2d");
    test.DumpGraph(graph);
  }

  /* Data_1 -> sqrt1 ------> conv2d
   *             \- sqrt2 ---/
   *  cannot fuse because duplicate is not allowed. */
  void BuildGraph_09(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .SetInputs({"sqrt1", "sqrt2"})
        .SetInput("NetOutput", "conv2d");
    test.DumpGraph(graph);
  }


  /* Data_1 -> sqrt1 ------> conv2d
   *             \- sqrt2 ---/
   *             \- sqrt3
   *  cannot fuse because although duplication is allowed, but the income
   *  from fusion is 2(sqrt2 and sqrt3), which is less than 3. */
  void BuildGraph_10(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .SetInputs({"sqrt1", "sqrt2"})
        .SetInput("NetOutput", "conv2d");
    test.DumpGraph(graph);
  }


  /*  Data_1 -> sqrt1 ------> conv2d
   *             \- sqrt2 ----/
   *             \- sqrt3  ->Netoutput
   *             \- sqrt4  ->Netoutput
   *  can fuse because duplication is allowed and the income
   *  from fusion is 3(sqrt2 and sqrt3 and sqrt4).
   *  After fusion, the graph is like:
   *  Data_1 -> sqrt1' -----------------> conv2d
   *         \- sqrt1sqrt2sqrt3sqrt4------/
   *                            \-------> Netoutput
   *                            \-------> Netoutput */
  void BuildGraph_11(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1)
        .SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "convolution", "conv2d", fe::CONV2D, 2, 1)
        .SetInputs({"sqrt1", "sqrt2"})
        .SetInputs("NetOutput", {"conv2d", "sqrt3", "sqrt4"});
    test.DumpGraph(graph);
  }

  /*  sqrt1 -> sqrt2 ->..... -> sqrt31 */
  void BuildGraph_12(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1).SetInputs({"Data_1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1).SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1).SetInputs({"sqrt2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1).SetInputs({"sqrt3"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt5", SQRT, 1, 1).SetInputs({"sqrt4"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt6", SQRT, 1, 1).SetInputs({"sqrt5"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt7", SQRT, 1, 1).SetInputs({"sqrt6"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt8", SQRT, 1, 1).SetInputs({"sqrt7"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt9", SQRT, 1, 1).SetInputs({"sqrt8"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt10", SQRT, 1, 1).SetInputs({"sqrt9"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt11", SQRT, 1, 1).SetInputs({"sqrt10"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt12", SQRT, 1, 1).SetInputs({"sqrt11"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt13", SQRT, 1, 1).SetInputs({"sqrt12"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt14", SQRT, 1, 1).SetInputs({"sqrt13"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt15", SQRT, 1, 1).SetInputs({"sqrt14"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt16", SQRT, 1, 1).SetInputs({"sqrt15"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt17", SQRT, 1, 1).SetInputs({"sqrt16"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt18", SQRT, 1, 1).SetInputs({"sqrt17"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt19", SQRT, 1, 1).SetInputs({"sqrt18"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt20", SQRT, 1, 1).SetInputs({"sqrt19"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt21", SQRT, 1, 1).SetInputs({"sqrt20"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt22", SQRT, 1, 1).SetInputs({"sqrt21"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt23", SQRT, 1, 1).SetInputs({"sqrt22"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt24", SQRT, 1, 1).SetInputs({"sqrt23"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt25", SQRT, 1, 1).SetInputs({"sqrt24"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt26", SQRT, 1, 1).SetInputs({"sqrt25"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt27", SQRT, 1, 1).SetInputs({"sqrt26"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt28", SQRT, 1, 1).SetInputs({"sqrt27"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt29", SQRT, 1, 1).SetInputs({"sqrt28"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt30", SQRT, 1, 1).SetInputs({"sqrt29"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt31", SQRT, 1, 1).SetInputs({"sqrt30"})
        .SetInputs("NetOutput", {"sqrt31"});
    test.DumpGraph(graph);
  }


  /*  add -> sqrt1 ->..... -> sqrt15
   *     \- sqrt16 -> .... -> sqrt31*/
  void BuildGraph_13(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1).SetInputs({"Data_1", "Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1).SetInputs({"sqrt1"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1).SetInputs({"sqrt2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1).SetInputs({"sqrt3"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt5", SQRT, 1, 1).SetInputs({"sqrt4"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt6", SQRT, 1, 1).SetInputs({"sqrt5"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt7", SQRT, 1, 1).SetInputs({"sqrt6"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt8", SQRT, 1, 1).SetInputs({"sqrt7"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt9", SQRT, 1, 1).SetInputs({"sqrt8"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt10", SQRT, 1, 1).SetInputs({"sqrt9"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt11", SQRT, 1, 1).SetInputs({"sqrt10"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt12", SQRT, 1, 1).SetInputs({"sqrt11"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt13", SQRT, 1, 1).SetInputs({"sqrt12"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt14", SQRT, 1, 1).SetInputs({"sqrt13"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt15", SQRT, 1, 1).SetInputs({"sqrt14"})


        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt16", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt17", SQRT, 1, 1).SetInputs({"sqrt16"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt18", SQRT, 1, 1).SetInputs({"sqrt17"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt19", SQRT, 1, 1).SetInputs({"sqrt18"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt20", SQRT, 1, 1).SetInputs({"sqrt19"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt21", SQRT, 1, 1).SetInputs({"sqrt20"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt22", SQRT, 1, 1).SetInputs({"sqrt21"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt23", SQRT, 1, 1).SetInputs({"sqrt22"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt24", SQRT, 1, 1).SetInputs({"sqrt23"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt25", SQRT, 1, 1).SetInputs({"sqrt24"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt26", SQRT, 1, 1).SetInputs({"sqrt25"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt27", SQRT, 1, 1).SetInputs({"sqrt26"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt28", SQRT, 1, 1).SetInputs({"sqrt27"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt29", SQRT, 1, 1).SetInputs({"sqrt28"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt30", SQRT, 1, 1).SetInputs({"sqrt29"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt31", SQRT, 1, 1).SetInputs({"sqrt30"})
        .SetInputs("NetOutput", {"sqrt15", "sqrt31"});
    test.DumpGraph(graph);
  }

  /*  add -> sqrt1 ->..... -> sqrt15
   *     \- sqrt16 -> .... -> sqrt31
   *  sqrt1 to sqrt9 are static impl
   *  sqrt10 to sqrt15 are dyn impl.
   */
  void BuildGraph_13_1(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1).SetInputs({"Data_1", "Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1).SetInputs({"sqrt1"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1).SetInputs({"sqrt2"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1).SetInputs({"sqrt3"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt5", SQRT, 1, 1).SetInputs({"sqrt4"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt6", SQRT, 1, 1).SetInputs({"sqrt5"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt7", SQRT, 1, 1).SetInputs({"sqrt6"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt8", SQRT, 1, 1).SetInputs({"sqrt7"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt9", SQRT, 1, 1).SetInputs({"sqrt8"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, false)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt10", SQRT, 1, 1).SetInputs({"sqrt9"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt11", SQRT, 1, 1).SetInputs({"sqrt10"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt12", SQRT, 1, 1).SetInputs({"sqrt11"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt13", SQRT, 1, 1).SetInputs({"sqrt12"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt14", SQRT, 1, 1).SetInputs({"sqrt13"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt15", SQRT, 1, 1).SetInputs({"sqrt14"})
        .Attr(ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt16", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt17", SQRT, 1, 1).SetInputs({"sqrt16"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt18", SQRT, 1, 1).SetInputs({"sqrt17"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt19", SQRT, 1, 1).SetInputs({"sqrt18"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt20", SQRT, 1, 1).SetInputs({"sqrt19"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt21", SQRT, 1, 1).SetInputs({"sqrt20"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt22", SQRT, 1, 1).SetInputs({"sqrt21"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt23", SQRT, 1, 1).SetInputs({"sqrt22"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt24", SQRT, 1, 1).SetInputs({"sqrt23"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt25", SQRT, 1, 1).SetInputs({"sqrt24"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt26", SQRT, 1, 1).SetInputs({"sqrt25"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt27", SQRT, 1, 1).SetInputs({"sqrt26"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt28", SQRT, 1, 1).SetInputs({"sqrt27"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt29", SQRT, 1, 1).SetInputs({"sqrt28"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt30", SQRT, 1, 1).SetInputs({"sqrt29"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt31", SQRT, 1, 1).SetInputs({"sqrt30"})
        .SetInputs("NetOutput", {"sqrt15", "sqrt31"});
    test.DumpGraph(graph);
  }

  /*  add -> sqrt1
   *     \- sqrt2
   *     \- sqrt3
   *     ....
   *     \- sqrt8 */
  void BuildGraph_14(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1).SetInputs({"Data_1", "Data_2"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt4", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt5", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt6", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt7", SQRT, 1, 1).SetInputs({"add1:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt8", SQRT, 1, 1).SetInputs({"add1:0"})
        .SetInputs("NetOutput", {"sqrt1", "sqrt2", "sqrt3", "sqrt4", "sqrt5",
                                 "sqrt6", "sqrt7", "sqrt8"});
    test.DumpGraph(graph);
  }

  /*   sqrt1--->Add1---->Add2---->Add3
   *     \-----A(unfusible)-------/
   */
  void BuildGraph_15(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetInputs({"sqrt1", "Data_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .SetInputs({"add1", "Data_3"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Non-ElemWise", "A", "A", 1, 1)
        .SetInputs({"sqrt1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add3", ADD, 2, 1)
        .SetInputs({"add2", "A"})

        .SetInputs("NetOutput", {"add3"});
    test.DumpGraph(graph);
  }


  /*   sqrt1--->Add1---->Add2------->Add3
   *     \-----sqrt2(2)--sqrt3(2)-----/
   *
   * sqrt2 and sqrt3 will be set as scope id 2 */
  void BuildGraph_16(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetInputs({"sqrt1", "Data_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .SetInputs({"add1", "Data_3"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"sqrt1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .SetInputs({"sqrt2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add3", ADD, 2, 1)
        .SetInputs({"add2", "sqrt3"})

        .SetInputs("NetOutput", {"add3"});
    test.DumpGraph(graph);
    ge::NodePtr sqrt2;
    ge::NodePtr sqrt3;
    test.GetNodeByName("sqrt2", sqrt2);
    int64_t scope_id = 2;
    ge::AttrUtils::SetInt(sqrt2->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    test.GetNodeByName("sqrt3", sqrt3);
    ge::AttrUtils::SetInt(sqrt3->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    ScopeAllocator::Instance().scope_id_ = 2;
  }

  /*   sqrt1--->Add1---->Add2------->Add3
   *     \-----sqrt2(2)--sqrt3(2)-----/
   *
   * sqrt2 and sqrt3 will be set as scope id 2 */
  void BuildGraph_17(ge::ComputeGraphPtr &graph, ge::NodePtr &sqrt1) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetInputs({"sqrt1", "Data_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .SetInputs({"add1", "Data_3"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"sqrt1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt3", SQRT, 1, 1)
        .SetInputs({"sqrt2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add3", ADD, 2, 1)
        .SetInputs({"add2", "sqrt3"})

        .SetInputs("NetOutput", {"add3"});
    test.DumpGraph(graph);
    ge::NodePtr sqrt2;
    ge::NodePtr sqrt3;
    test.GetNodeByName("sqrt2", sqrt2);
    int64_t scope_id = 2;
    ge::AttrUtils::SetInt(sqrt2->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    test.GetNodeByName("sqrt3", sqrt3);
    ge::AttrUtils::SetInt(sqrt3->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    ScopeAllocator::Instance().scope_id_ = 2;
    test.GetNodeByName("sqrt1", sqrt1);
  }

  /*   sqrt1--->Add1
   *   sqrt2----/
   *
   * test the connectivity matrix */
  void BuildGraph_18(ge::ComputeGraphPtr &graph,
      ge::NodePtr& sqrt1, ge::NodePtr& sqrt2, ge::NodePtr& add1) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt1", SQRT, 1, 1)
        .SetInputs({"Data_1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt2", SQRT, 1, 1)
        .SetInputs({"Data_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", ADD, 2, 1)
        .SetInputs({"sqrt1", "sqrt2"})

        .SetInputs("NetOutput", {"add1"});
    test.GetNodeByName("sqrt1", sqrt1);
    test.GetNodeByName("sqrt2", sqrt2);
    test.GetNodeByName("add1", add1);
  }


  /*   realdiv------->sign---------------->mul
   *        \---abs-->add(2)--->sqrt(2)--->/
   *
   * add and sqrt will be set as scope id 2 */
  void BuildGraph_19(ge::ComputeGraphPtr &graph, ge::NodePtr &realdiv,
      ge::NodePtr& sign, ge::NodePtr &abs, ge::NodePtr &mul) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "realdiv", "RealDiv", 2, 2)
        .SetInputs({"Data_1", "Data_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sign", "Sign", 1, 1)
        .SetInputs({"realdiv"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "abs", "Abs", 1, 1)
        .SetInputs({"realdiv"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", ADD, 2, 2)
        .SetInputs({"abs", "Data_3"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sqrt", SQRT, 1, 1)
        .SetInputs({"add"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul", "Mul", 2, 2)
        .SetInputs({"sign", "sqrt"})

        .SetInputs("NetOutput", {"add1"});

    ge::NodePtr add;
    ge::NodePtr sqrt;
    test.GetNodeByName("add", add);
    int64_t scope_id = 2;
    ge::AttrUtils::SetInt(add->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    test.GetNodeByName("sqrt", sqrt);
    ge::AttrUtils::SetInt(sqrt->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    ScopeAllocator::Instance().scope_id_ = 2;

    test.GetNodeByName("realdiv", realdiv);
    test.GetNodeByName("sign", sign);
    test.GetNodeByName("abs", abs);
    test.GetNodeByName("mul", mul);
  }

  /*                    /------Square2-----\
   *       /-------Maximum------\          \
   *  Square------>Mul1-------> Mul2---->Maximum2------> Mul3----->Maximum3
   *                \------->Maximum1(2)------>Square3(2)------->/
   *
   * add and sqrt will be set as scope id 2 */
  void BuildGraph_20(ge::ComputeGraphPtr &graph, ge::NodePtr &mul1,
                     ge::NodePtr& square, ge::NodePtr &mul2, ge::NodePtr &max3) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    string SQRT = "Sqrt";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Square", "Square", 1, 1)
        .SetInputs({"Data_1"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Maximum", "Maximum", 2, 1)
        .SetInputs({"Square:0", "Data_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul1", "Mul", 2, 1)
        .SetInputs({"Data_3", "Square:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Maximum1", "Maximum", 2, 1)
        .SetInputs({"Mul1:0", "Data_4"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul2", "Mul", 2, 1)
        .SetInputs({"Maximum:0", "Mul1:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Square2", "Square", 1, 1)
        .SetInputs({"Maximum:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Maximum2", "Maximum", 2, 1)
        .SetInputs({"Mul2:0", "Square2:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Square3", "Square", 1, 1)
        .SetInputs({"Maximum1:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul3", "Mul", 2, 1)
        .SetInputs({"Maximum2:0", "Data_5"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Maximum3", "Maximum", 2, 1)
        .SetInputs({"Mul3:0", "Square3:0"})

        .SetInputs("NetOutput", {"Maximum3:0"});

    ge::NodePtr max1;
    ge::NodePtr square3;
    test.GetNodeByName("Maximum1", max1);
    int64_t scope_id = 2;
    ge::AttrUtils::SetInt(max1->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    test.GetNodeByName("Square3", square3);
    ge::AttrUtils::SetInt(square3->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    ScopeAllocator::Instance().scope_id_ = 2;

    test.GetNodeByName("Mul1", mul1);
    test.GetNodeByName("Square", square);
    test.GetNodeByName("Mul2", mul2);
    test.GetNodeByName("Maximum3", max3);
  }
  /* BNTrainingUpdateV2��has other 5 output�� -> add2(2 inputs are the same)->add3
   *
   */
  void BuildGraph_21(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);
    string ADD = "Add";
    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "bn", "BNTrainingUpdateV2", 3, 3)
        .SetInputs({"Data_1", "Data_2", "Data_3"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add2", ADD, 2, 1)
        .SetInputs({"bn:0", "bn:0"})
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add3", ADD, 2, 1)
        .SetInputs({"add2:0", "add2:0"})
        .SetInput("other_1", "bn:1")
        .SetInput("other_2", "bn:1")
        .SetInput("other_3", "bn:2")
        .SetInput("other_4", "bn:2")
        .SetInput("other_5", "add3:0");
    test.DumpGraph(graph);
  }


  /* contains control edges. */
  void BuildGraph_22(ge::ComputeGraphPtr &graph, ge::NodePtr &addn,
                     ge::NodePtr& realdiv1, ge::NodePtr &realdiv,
                     ge::NodePtr& transdata) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);

    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "AddN", "AddN", 2, 1)
        .SetInputs({"PlaceHolder_1", "PlaceHolder_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul1", "Mul", 2, 1)
        .SetInputs({"AddN:0", "PlaceHolder_4"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "RealDiv", "RealDiv", 2, 1)
        .SetInputs({"AddN:0", "PlaceHolder_3"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Add", "Add", 2, 1)
        .SetInputs({"Mul1:0", "PlaceHolder_5"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "Assign", "Assign", 2, 1)
        .SetInputs({"Add:0", "PlaceHolder_6"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "TransData", "TransData", 1, 1)
        .SetInputs({"Assign:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "RealDiv1", "RealDiv", 2, 1)
        .SetInputs({"PlaceHolder_7", "PlaceHolder_8"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul4", "Mul", 2, 1)
        .SetInputs({"RealDiv:0", "PlaceHolder_10"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul5", "Mul", 2, 1)
        .SetInputs({"RealDiv1:0", "PlaceHolder_9"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Add2", "Add", 2, 1)
        .SetInputs({"Mul4:0", "Mul5:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "End", "End", 1, 1)
        .SetInputs({"Add2:0"});

    test.GetNodeByName("AddN", addn);
    test.GetNodeByName("RealDiv1", realdiv1);
    test.GetNodeByName("RealDiv", realdiv);
    test.GetNodeByName("TransData", transdata);
  }


  /* contains control edges. */
  void BuildGraph_23(ge::ComputeGraphPtr &graph, ge::NodePtr &addn,
                     ge::NodePtr& realdiv1, ge::NodePtr &realdiv,
                     ge::NodePtr& transdata) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);

    test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "AddN", "AddN", 2, 1)
        .SetInputs({"PlaceHolder_1", "PlaceHolder_2"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "RealDiv", "RealDiv", 2, 1)
        .SetInputs({"AddN:0", "PlaceHolder_3"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul1", "Mul", 2, 1)
        .SetInputs({"AddN:0", "PlaceHolder_4"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Add", "Add", 2, 1)
        .SetInputs({"Mul1:0", "PlaceHolder_5"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "Assign", "Assign", 2, 1)
        .SetInputs({"Add:0", "PlaceHolder_6"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "TransData", "TransData", 1, 1)
        .SetInputs({"Assign:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "RealDiv1", "RealDiv", 2, 1)
        .SetInputs({"PlaceHolder_7", "PlaceHolder_8"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul5", "Mul", 2, 1)
        .SetInputs({"RealDiv1:0", "PlaceHolder_9"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul4", "Mul", 2, 1)
        .SetInputs({"RealDiv:0", "PlaceHolder_10"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Add2", "Add", 2, 1)
        .SetInputs({"Mul4:0", "Mul5:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "End", "End", 1, 1)
        .SetInputs({"Add2:0"});

    test.GetNodeByName("AddN", addn);
    test.GetNodeByName("RealDiv1", realdiv1);
    test.GetNodeByName("RealDiv", realdiv);
    test.GetNodeByName("TransData", transdata);
  }

  /* Do not contain any cycle. */
  void BuildGraph_24(ge::ComputeGraphPtr &graph, ge::NodePtr &abs,
                     ge::NodePtr& square, ge::NodePtr &neg) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);

    test.AddOpDesc("Data", "Data", 1, 1)

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Abs", "Abs", 1, 1)
        .SetInputs({"Data:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Square", "Square", 1, 1)
        .SetInputs({"Abs:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Mul", "Mul", 2, 1)
        .SetInputs({"Square:0", "Const_0:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Square_1", "Square", 1, 1)
        .SetInputs({"Mul:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Sub", "Sub", 2, 1)
        .SetInputs({"Abs:0", "Square_1:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Neg", "Neg", 1, 1)
        .SetInputs({"Sub:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "Add", "Add", 2, 1)
        .SetInputs({"Sub:0", "Const_1:0"})

        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "FloorMod", "FloorMod", 2, 1)
        .SetInputs({"Neg:0", "Add:0"});

    test.GetNodeByName("Abs", abs);
    test.GetNodeByName("Square", square);
    test.GetNodeByName("Neg", neg);
  }

  ge::NodePtr AddNode(ge::ComputeGraphPtr graph, const string &name, const string &type,
                      int32_t out_anchors_num = 1, int32_t in_anchors_num = 1,
                      bool flag = false, string pattern = "") {
    ge::GeTensorDesc tensor_desc(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>(name, type);
    if (flag) {
      auto key_pattern = "_pattern";
      ge::AttrUtils::SetStr(opdesc, key_pattern, pattern);
      (void)ge::AttrUtils::SetInt(opdesc, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
      const char tbe_bin[] = "tbe_bin";
      std::vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
      ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(name, std::move(buffer));
      opdesc->SetExtAttr("tbeKernel", tbe_kernel_ptr);
    }
    for (int32_t i = 0; i < out_anchors_num; i++) {
      opdesc->AddOutputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < in_anchors_num; i++) {
      opdesc->AddInputDesc(tensor_desc);
    }
    ge::NodePtr node = graph->AddNode(opdesc);
    if (node->GetType() != "PlaceHolder") {
      ge::AttrUtils::SetInt(opdesc, FE_IMPLY_TYPE, 6);
    }
    return node;
  }

  void BuildGraph_25(ge::ComputeGraphPtr &graph) {
    ge::NodePtr sub = AddNode(graph, "sub", "Sub", 1, 2, true, "ElemWise");
    ge::NodePtr sub1 = AddNode(graph, "sub1", "Sub", 1, 2, true, "ElemWise");
    ge::NodePtr square = AddNode(graph, "square", "Square", 1, 1, false, "");
    ge::NodePtr square2 = AddNode(graph, "square2", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr square3 = AddNode(graph, "square3", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr square4 = AddNode(graph, "square4", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr square5 = AddNode(graph, "square5", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr square6 = AddNode(graph, "square6", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr square7 = AddNode(graph, "square7", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr square8 = AddNode(graph, "square8", "Square", 1, 1, true, "ElemWise");
    ge::NodePtr add = AddNode(graph, "add", "Add", 1, 2, true, "ElemWise");
    ge::NodePtr add1 = AddNode(graph, "add1", "Add", 1, 2, false, "");
    ge::NodePtr add3 = AddNode(graph, "add3", "Add", 1, 2, true, "ElemWise");
    ge::NodePtr add4 = AddNode(graph, "add4", "Add", 1, 2, true, "ElemWise");
    ge::NodePtr add5 = AddNode(graph, "add5", "Add", 1, 2, true, "ElemWise");
    ge::NodePtr add6 = AddNode(graph, "add6", "Add", 1, 2, true, "ElemWise");
    ge::NodePtr neg = AddNode(graph, "neg", "Neg", 1, 1, false, "");
    ge::NodePtr placeholder = AddNode(graph, "placeholder", "PlaceHolder");
    ge::NodePtr placeholder1 = AddNode(graph, "placeholder1", "PlaceHolder");

    ge::GraphUtils::AddEdge(sub->GetOutDataAnchor(0), square->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sub->GetOutDataAnchor(0), square2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sub->GetOutDataAnchor(0), square5->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sub->GetOutDataAnchor(0), square7->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(square->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(square->GetOutDataAnchor(0), add4->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(add1->GetOutDataAnchor(0), neg->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(placeholder->GetOutDataAnchor(0), sub1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(placeholder1->GetOutDataAnchor(0), sub1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(sub1->GetOutDataAnchor(0), square3->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sub1->GetOutDataAnchor(0), square4->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sub1->GetOutDataAnchor(0), square6->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(sub1->GetOutDataAnchor(0), square8->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(square2->GetOutDataAnchor(0), add->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(square3->GetOutDataAnchor(0), add->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(neg->GetOutDataAnchor(0), add3->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(add4->GetOutDataAnchor(0), add3->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(square4->GetOutDataAnchor(0), add4->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(square5->GetOutDataAnchor(0), add5->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(square6->GetOutDataAnchor(0), add5->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(square7->GetOutDataAnchor(0), add6->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(square8->GetOutDataAnchor(0), add6->GetInDataAnchor(1));
  }
};


TEST_F(UB_FUSION_ST_AUTO_FUSION, two_add_fuse_succ) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_01(graph);

  ffts::ThreadSliceMapPtr threadSliceMap = std::make_shared<ffts::ThreadSliceMap>();
  threadSliceMap->ori_output_tensor_shape = {{{10,20,30,40}}};
  threadSliceMap->output_tensor_slice = {{{{0,10}}}};
  for (auto &node : graph->GetDirectNode()) {
    node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, threadSliceMap);
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), "_thread_id", 0);
  }

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  EXPECT_EQ(sub_graph_optimizer_ptr_->BuildFusionGraph(*graph), fe::SUCCESS);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" <<
        node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
  }
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, three_add_fuse_succ) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_06(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ++node_count;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    if (node->GetName() == "add1add2add3") {
      ++fusion_node_count;
    }
  }
  EXPECT_EQ(node_count, 3);
  EXPECT_EQ(fusion_node_count, 1);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, three_add_fuse_fail) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_06(graph);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "add1") {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr1");
    }
    if (node->GetName() == "add2") {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr2");
    }
    if (node->GetName() == "add3") {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr3");
    }
  }

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, multiple_input_from_same_node) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_07(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ++node_count;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    if (node->GetName() == "sqrt1sqrt2add") {
      ++fusion_node_count;
    }
  }
  EXPECT_EQ(node_count, 3);
  EXPECT_EQ(fusion_node_count, 1);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_01) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_02(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ++node_count;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    if (node->GetName() == "add1sqrt3sqrt4sqrt2sqrt1add2") {
      ++fusion_node_count;
    }
  }
  EXPECT_EQ(node_count, 8);
  EXPECT_EQ(fusion_node_count, 1);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_long_name) {
    string one_hundred_one;
    for (int i = 0; i < 100; i++) {
        one_hundred_one += "1";
    }
    ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
    BuildGraph_Long_Name(graph);

    sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
    // find sub-graphs that match UB fusion pattern
    auto_buffer_fusion_ptr_->Run(*graph);

    // create fused Graph, and merge matched sub-graphs into fusion ops
    sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

    cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
    uint32_t node_count = 0;
    uint32_t fusion_node_count = 0;
    for (auto &node : graph->GetDirectNode()) {
        ++node_count;
        uint32_t scope_id = 0;
        cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

        if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
            cerr << "scope id : " << scope_id << endl;
        }
        if (node->GetName() == "add1"+one_hundred_one) {
            ++fusion_node_count;
        }
    }
    EXPECT_EQ(node_count, 19);
    EXPECT_EQ(fusion_node_count, 1);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_02) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_03(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ++node_count;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    if (node->GetName() == "sqrt3sqrt4") {
      ++fusion_node_count;
    }
  }
  EXPECT_EQ(node_count, 15);
  EXPECT_EQ(fusion_node_count, 1);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_03) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_04(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ++node_count;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    if (node->GetName() == "sqrt3add3") {
      ++fusion_node_count;
    }
  }
  EXPECT_EQ(node_count, 15);
  EXPECT_EQ(fusion_node_count, 1);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_04) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_05(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "sqrt3sqrt4" ||
        node->GetName() == "sqrt5sqrt6") {
      fusion_node_count++;
    }
  }
  EXPECT_EQ(node_count, 14);
  EXPECT_EQ(fusion_node_count, 2);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_05) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_08(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "add1sqrt2sqrt4sqrt6sqrt1sqrt3sqrt5") {
      fusion_node_count++;
    }
  }
  EXPECT_EQ(node_count, 5);
  EXPECT_EQ(fusion_node_count, 1);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_06) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_09(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->may_duplicate_ = true;
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "sqrt1sqrt2") {
      fusion_node_count++;
    }
  }
  auto_buffer_fusion_ptr_->may_duplicate_ = false;
  EXPECT_EQ(node_count, 4);
  EXPECT_EQ(fusion_node_count, 1);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_07) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_10(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->may_duplicate_ = true;
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "sqrt1sqrt3sqrt2") {
      fusion_node_count++;
    }
  }
  auto_buffer_fusion_ptr_->may_duplicate_ = false;
  EXPECT_EQ(node_count, 4);
  EXPECT_EQ(fusion_node_count, 1);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_08) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_11(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "sqrt1sqrt2sqrt3sqrt4") {
      fusion_node_count++;
    }
  }
  EXPECT_EQ(node_count, 7);
  EXPECT_EQ(fusion_node_count, 0);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_09) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_12(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:"
         << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "sqrt1sqrt2" ||
        node->GetName() ==
        "sqrt3sqrt4sqrt5sqrt6sqrt7sqrt8sqrt9sqrt10sqrt11sqrt12sqrt13sqrt14sqrt15sqrt16sqrt17sqrt18sqrt19sqrt20sqrt21sqrt22sqrt23sqrt24sqrt25sqrt26sqrt27sqrt28sqrt29sqrt30sqrt31") {
      fusion_node_count++;
    }
  }
  EXPECT_EQ(node_count, 4);
  EXPECT_EQ(fusion_node_count, 2);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_10) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_13(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:"
         << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "sqrt16sqrt17sqrt18sqrt19sqrt20sqrt21sqrt22sqrt23sqrt24sqrt25sqrt26sqrt27sqrt28sqrt29sqrt30sqrt31" ||
        node->GetName() ==
        "add1sqrt1sqrt2sqrt3sqrt4sqrt5sqrt6sqrt7sqrt8sqrt9sqrt10sqrt11sqrt12sqrt13sqrt14sqrt15") {
      fusion_node_count++;
    }
  }
  EXPECT_EQ(node_count, 5);
  EXPECT_EQ(fusion_node_count, 2);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_10_1) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_13_1(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:"
         << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }

    if (node->GetName() == "add1sqrt16sqrt17sqrt18sqrt19sqrt20sqrt21sqrt22sqrt23sqrt24sqrt25sqrt26sqrt27sqrt28sqrt29sqrt30sqrt31sqrt1sqrt2sqrt3sqrt4sqrt5sqrt6sqrt7sqrt8sqrt9" ||
        node->GetName() == "sqrt10sqrt11sqrt12sqrt13sqrt14sqrt15") {
      fusion_node_count++;
    }
  }
  EXPECT_EQ(node_count, 5);
  EXPECT_EQ(fusion_node_count, 2);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, complex_11) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_14(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:"
         << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
  }
  EXPECT_EQ(node_count, 12);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_01) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_15(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:"
         << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
  }
  EXPECT_EQ(node_count, 7);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_02) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_16(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  uint32_t node_count = 0;
  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    node_count++;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:"
         << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
  }
  EXPECT_EQ(node_count, 7);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_03) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr sqrt1;
  BuildGraph_17(graph, sqrt1);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  bool path_exists = auto_buffer_fusion_ptr_->CheckPathExists(sqrt1, 3, nullptr);
  EXPECT_EQ(path_exists, true);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_04) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr realdiv;
  ge::NodePtr sign;
  ge::NodePtr abs;
  ge::NodePtr mul;

  BuildGraph_19(graph, realdiv, sign, abs, mul);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id_realdiv = -1;
  int64_t scope_id_sign = -1;
  int64_t scope_id_mul = -1;
  int64_t scope_id_abs = -1;
  ge::AttrUtils::GetInt(realdiv->GetOpDesc(), SCOPE_ID_ATTR, scope_id_realdiv);
  ge::AttrUtils::GetInt(sign->GetOpDesc(), SCOPE_ID_ATTR, scope_id_sign);
  ge::AttrUtils::GetInt(mul->GetOpDesc(), SCOPE_ID_ATTR, scope_id_mul);
  ge::AttrUtils::GetInt(abs->GetOpDesc(), SCOPE_ID_ATTR, scope_id_abs);

  EXPECT_EQ(scope_id_realdiv, 4);
  EXPECT_EQ(scope_id_abs, 4);

  EXPECT_EQ(scope_id_sign, 3);
  EXPECT_EQ(scope_id_mul, 3);

  bool path_exists = auto_buffer_fusion_ptr_->CheckPathExists(realdiv, 3, nullptr);
  EXPECT_EQ(path_exists, true);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_05) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr mul1;
  ge::NodePtr square;
  ge::NodePtr mul2;
  ge::NodePtr max3;

  BuildGraph_20(graph, mul1, square, mul2, max3);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id_mul1 = -1;
  int64_t scope_id_square = -1;
  int64_t scope_id_mul2 = -1;
  int64_t scope_id_max3 = -1;
  ge::AttrUtils::GetInt(mul1->GetOpDesc(), SCOPE_ID_ATTR, scope_id_mul1);
  ge::AttrUtils::GetInt(square->GetOpDesc(), SCOPE_ID_ATTR, scope_id_square);
  ge::AttrUtils::GetInt(mul2->GetOpDesc(), SCOPE_ID_ATTR, scope_id_mul2);
  ge::AttrUtils::GetInt(max3->GetOpDesc(), SCOPE_ID_ATTR, scope_id_max3);

  EXPECT_EQ(scope_id_mul1, 4);
  EXPECT_EQ(scope_id_square, 4);

  EXPECT_EQ(scope_id_mul2, 3);
  EXPECT_EQ(scope_id_max3, 3);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_06) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr addn;
  ge::NodePtr realdiv;
  ge::NodePtr realdiv1;
  ge::NodePtr transdata;

  BuildGraph_22(graph, addn, realdiv1, realdiv, transdata);
  auto out_control_anchor = transdata->GetOutControlAnchor();
  if (out_control_anchor == nullptr) {
    out_control_anchor = std::make_shared<ge::OutControlAnchor>(transdata);
  }
  auto in_control_anchor = realdiv1->GetInControlAnchor();
  if (in_control_anchor == nullptr) {
    in_control_anchor = std::make_shared<ge::InControlAnchor>(realdiv1);
  }
  ge::GraphUtils::AddEdge(out_control_anchor, in_control_anchor);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id_addn = -1;
  int64_t scope_id_r1 = -1;
  int64_t scope_id_r2 = -1;
  ge::AttrUtils::GetInt(addn->GetOpDesc(), SCOPE_ID_ATTR, scope_id_addn);
  ge::AttrUtils::GetInt(realdiv->GetOpDesc(), SCOPE_ID_ATTR, scope_id_r1);
  ge::AttrUtils::GetInt(realdiv1->GetOpDesc(), SCOPE_ID_ATTR, scope_id_r2);

  EXPECT_EQ(scope_id_addn > 0, true);
  EXPECT_EQ(scope_id_r1 > 0, true);
  EXPECT_EQ(scope_id_r2 > 0, true);
  EXPECT_NE(scope_id_addn, scope_id_r1);
  EXPECT_NE(scope_id_addn, scope_id_r2);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_06_1) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr addn;
  ge::NodePtr realdiv;
  ge::NodePtr realdiv1;
  ge::NodePtr transdata;

  BuildGraph_23(graph, addn, realdiv1, realdiv, transdata);
  auto out_control_anchor = transdata->GetOutControlAnchor();
  if (out_control_anchor == nullptr) {
    out_control_anchor = std::make_shared<ge::OutControlAnchor>(transdata);
  }
  auto in_control_anchor = realdiv1->GetInControlAnchor();
  if (in_control_anchor == nullptr) {
    in_control_anchor = std::make_shared<ge::InControlAnchor>(realdiv1);
  }
  ge::GraphUtils::AddEdge(out_control_anchor, in_control_anchor);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id_addn = -1;
  int64_t scope_id_r1 = -1;
  int64_t scope_id_r2 = -1;
  ge::AttrUtils::GetInt(addn->GetOpDesc(), SCOPE_ID_ATTR, scope_id_addn);
  ge::AttrUtils::GetInt(realdiv->GetOpDesc(), SCOPE_ID_ATTR, scope_id_r1);
  ge::AttrUtils::GetInt(realdiv1->GetOpDesc(), SCOPE_ID_ATTR, scope_id_r2);

  EXPECT_EQ(scope_id_addn > 0, true);
  EXPECT_EQ(scope_id_r1 > 0, true);
  EXPECT_EQ(scope_id_r2 > 0, true);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_06_2) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr addn;
  ge::NodePtr realdiv;
  ge::NodePtr realdiv1;
  ge::NodePtr transdata;

  BuildGraph_22(graph, addn, realdiv1, realdiv, transdata);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id_addn = -1;
  int64_t scope_id_r1 = -1;
  int64_t scope_id_r2 = -1;
  ge::AttrUtils::GetInt(addn->GetOpDesc(), SCOPE_ID_ATTR, scope_id_addn);
  ge::AttrUtils::GetInt(realdiv->GetOpDesc(), SCOPE_ID_ATTR, scope_id_r1);
  ge::AttrUtils::GetInt(realdiv1->GetOpDesc(), SCOPE_ID_ATTR, scope_id_r2);

  EXPECT_EQ(scope_id_addn > 0, true);
  EXPECT_EQ(scope_id_r1 > 0, true);
  EXPECT_EQ(scope_id_r2 > 0, true);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_detection_07) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr abs;
  ge::NodePtr square;
  ge::NodePtr neg;
  ge::NodePtr transdata;

  BuildGraph_24(graph, abs, square, neg);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id_abs = -1;
  int64_t scope_id_square = -1;
  int64_t scope_id_neg = -1;
  ge::AttrUtils::GetInt(abs->GetOpDesc(), SCOPE_ID_ATTR, scope_id_abs);
  ge::AttrUtils::GetInt(square->GetOpDesc(), SCOPE_ID_ATTR, scope_id_square);
  ge::AttrUtils::GetInt(neg->GetOpDesc(), SCOPE_ID_ATTR, scope_id_neg);

  EXPECT_EQ(scope_id_abs > 0, true);
  EXPECT_EQ(scope_id_square > 0, true);
  EXPECT_EQ(scope_id_neg > 0, true);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_loop_dectection_08) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_25(graph);
  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);
  int64_t scope_id = -1;
  for (auto &node : graph->GetDirectNode()) {
    ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id);
    if (node->GetName() == "square7" || node->GetName() == "square5" || node->GetName() == "square2" ||
        node->GetName() == "square8" || node->GetName() == "square6" || node->GetName() == "square4" ||
        node->GetName() == "square3" || node->GetName() == "sub1" || node->GetName() == "add6" ||
        node->GetName() == "add5" || node->GetName() == "add4" ||
        node->GetName() == "add3" || node->GetName() == "add") {
      EXPECT_EQ(scope_id, 12);
    }
    if (node->GetName() == "sub") {
      EXPECT_EQ(scope_id, -1);
    }
  }
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
  EXPECT_EQ(graph->GetDirectNode().size(), 7);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_reachability_map) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr sqrt1;
  ge::NodePtr sqrt2;
  ge::NodePtr add1;
  BuildGraph_18(graph, sqrt1, sqrt2, add1);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  EXPECT_EQ(auto_buffer_fusion_ptr_->connection_matrix_->IsConnected(sqrt1, sqrt2), true);
  EXPECT_EQ(auto_buffer_fusion_ptr_->connection_matrix_->IsConnected(sqrt1, add1), true);
}


TEST_F(UB_FUSION_ST_AUTO_FUSION, Bnv2) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_21(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  auto_buffer_fusion_ptr_->Run(*graph);

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  cerr << "UB_FUSION_UT_ELT_ELT UB fusion result" << endl;
  uint32_t node_count = 0;
  uint32_t fusion_node_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ++node_count;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;

    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    std::vector<string> peer_in_names = {
        "other_1", "other_2", "other_3", "other_4", "other_5"
    };
    if (node->GetName() == "bnadd2add3") {
      ++fusion_node_count;
      int i = 0;
      for (auto peer_in_node : node->GetOutAllNodes()) {
        EXPECT_EQ(peer_in_names[i], peer_in_node->GetName());
        i++;
      }
    }
  }
  EXPECT_EQ(node_count, 9);
  EXPECT_EQ(fusion_node_count, 1);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, change_scope_id_test){
  int64_t old_scope_id = 0;
  int64_t new_scope_id = 2;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("test", "Test");
  vector<int64_t> dims = {1, 2, 3};
  ge::GeShape shape = ge::GeShape(dims);
  ge::GeTensorDesc tensor(shape);
  op->AddInputDesc(tensor);
  op->AddOutputDesc(tensor);
  ge::NodePtr node1 = graph->AddNode(op);

  ScopeInfo inner;
  inner.nodes.emplace(std::make_pair(0, node1));
  inner.nodes.emplace(std::make_pair(1, node1));
  inner.nodes.emplace(std::make_pair(2, node1));
  inner.nodes.emplace(std::make_pair(3, node1));
  inner.nodes.emplace(std::make_pair(4, node1));
  inner.nodes.emplace(std::make_pair(5, node1));
  inner.nodes.emplace(std::make_pair(6, node1));
  inner.nodes.emplace(std::make_pair(7, node1));
  inner.nodes.emplace(std::make_pair(8, node1));
  inner.nodes.emplace(std::make_pair(9, node1));
  inner.nodes.emplace(std::make_pair(10, node1));
  inner.nodes.emplace(std::make_pair(11, node1));
  inner.nodes.emplace(std::make_pair(12, node1));
  inner.nodes.emplace(std::make_pair(13, node1));
  inner.nodes.emplace(std::make_pair(14, node1));
  inner.nodes.emplace(std::make_pair(15, node1));
  auto_buffer_fusion_ptr_->scope_id_nodes_map_.emplace(std::make_pair(0, inner));
  auto_buffer_fusion_ptr_->scope_id_nodes_map_.emplace(std::make_pair(2, inner));
  Status ret = auto_buffer_fusion_ptr_->ChangeScopeId(old_scope_id, new_scope_id);
  EXPECT_EQ(ret, GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, test_get_op_pattern) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::NodePtr mul_node = graph->AddNode(mul);
  string op_pattern;
  bool ret = auto_buffer_fusion_ptr_->GetOpAttrPattern(mul_node, op_pattern);
  EXPECT_EQ(ret, false);
}

TEST_F(UB_FUSION_ST_AUTO_FUSION, fuse_two_nodes){
  using namespace ge;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");

  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("DATA0", fe::DATA);
  ge::OpDescPtr data1 = std::make_shared<ge::OpDesc>("DATA1", fe::DATA);
  ge::OpDescPtr add = std::make_shared<ge::OpDesc>("add", "Add");
  ge::OpDescPtr sqrt1 = std::make_shared<ge::OpDesc>("sqrt1", "ElemWise");
  ge::OpDescPtr sqrt2 = std::make_shared<ge::OpDesc>("sqrt2", "ElemWise");

  AttrUtils::SetInt(add, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  AttrUtils::SetInt(sqrt1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  AttrUtils::SetInt(sqrt2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);

  // add descriptor
  vector<int64_t> dim = {4, 4, -1, 4};
  ge::GeShape shape(dim);
  ge::GeTensorDesc tenosr_desc(shape);

  data->AddOutputDesc(tenosr_desc);
  data1->AddOutputDesc(tenosr_desc);
  add->AddInputDesc(tenosr_desc);
  add->AddInputDesc(tenosr_desc);
  add->AddOutputDesc(tenosr_desc);
  sqrt1->AddInputDesc(tenosr_desc);
  sqrt1->AddOutputDesc(tenosr_desc);
  sqrt2->AddInputDesc(tenosr_desc);
  sqrt2->AddOutputDesc(tenosr_desc);
  AttrUtils::SetBool(sqrt1, ATTR_NAME_IS_OP_DYNAMIC_IMPL, true);

  ge::NodePtr data_node = graph->AddNode(data);
  ge::NodePtr data1_node = graph->AddNode(data1);
  ge::NodePtr sqrt1_node = graph->AddNode(sqrt1);
  ge::NodePtr add_node = graph->AddNode(add);
  ge::NodePtr sqrt2_node = graph->AddNode(sqrt2);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), sqrt1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(sqrt1_node->GetOutDataAnchor(0), sqrt2_node->GetInDataAnchor(0));

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
  int64_t consumer_scope_id = -1;
  int64_t producer_scope_id = -1;
  // find sub-graphs that match UB fusion pattern
  auto ret = auto_buffer_fusion_ptr_->FuseTwoNodes(sqrt1_node, sqrt2_node, producer_scope_id, consumer_scope_id);
  EXPECT_EQ(ret, fe::SUCCESS);
}

//TEST_F(UB_FUSION_ST_AUTO_FUSION, update_all_related_nodes_when_two_scope_merged) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("cycle_detection_network.txt");
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("ut");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//
//  const string ele_pattern = "ElemWise";
//  for (auto &node : graph->GetDirectNode()) {
//    auto op = node->GetOpDesc();
//    if (node->GetType() == "Add" || node->GetType() == "LessEqual" || node->GetType() == "Minimum") {
//      ge::AttrUtils::SetStr(op, kOpPattern, ele_pattern);
//      const char tbe_bin[] = "tbe_bin";
//      vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
//      ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
//          node->GetName(), std::move(buffer));
//      op->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
//    }
//  }
//  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
//
//  auto_buffer_fusion_ptr_->Run(*graph);
//
//  // create fused Graph, and merge matched sub-graphs into fusion ops
//  ASSERT_EQ(fe::SUCCESS, sub_graph_optimizer_ptr_->BuildFusionGraph(*graph));
//}

//TEST_F(UB_FUSION_ST_AUTO_FUSION, realdiv_self_cycle_by_control_edge) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("test_cycle_detection_control_edges.txt");
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("ut");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//
//  const string ele_pattern = "ElemWise";
//  for (auto &node : graph->GetDirectNode()) {
//    auto op = node->GetOpDesc();
//    auto type = node->GetType();
//    if (type != OP_TYPE_PLACE_HOLDER &&
//        type != OP_TYPE_END && type != "StridedSliceGrad") {
//      ge::AttrUtils::SetStr(op, kOpPattern, ele_pattern);
//      const char tbe_bin[] = "tbe_bin";
//      vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
//      ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
//          node->GetName(), std::move(buffer));
//      op->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
//    }
//  }
//  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
//  std::shared_ptr<FusionCycleDetector> cycle_detector;
//  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
//  cycle_detector->Initialize(*graph);
//
//  auto create_fn1 = []() -> BufferFusionPassBase * { return new (std::nothrow) TbeMultiOutputFusionPass(); };
//  BufferFusionPassRunner *test_pass1 = new (std::nothrow)
//      BufferFusionPassRunner("test_pass1", create_fn1, cycle_detector);
//  auto create_fn2= []() -> BufferFusionPassBase * { return new (std::nothrow) TbeEltwiseFusionPass(); };
//  BufferFusionPassRunner *test_pass2 = new (std::nothrow)
//      BufferFusionPassRunner("test_pass2", create_fn2, cycle_detector);
//  test_pass1->Run(*graph);
//  test_pass2->Run(*graph);
//  auto_buffer_fusion_ptr_->Run(*graph);
//  ge::GraphUtils::DumpGEGraphToOnnx(*graph, "before");
//  // create fused Graph, and merge matched sub-graphs into fusion ops
//  EXPECT_EQ(fe::SUCCESS, sub_graph_optimizer_ptr_->BuildFusionGraph(*graph));
//  ge::GraphUtils::DumpGEGraphToOnnx(*graph, "after");
//}

//TEST_F(UB_FUSION_ST_AUTO_FUSION, realdiv_self_cycle_by_control_edge_only_automatic_fusion) {
//  auto &fe_env = fe_env::FeRunningEnv::Instance();
//  string network_path = fe_env::FeRunningEnv::GetNetworkPath("test_cycle_detection_control_edges.txt");
//  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("ut");
//  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//  ASSERT_EQ(state, true);
//
//  const string ele_pattern = "ElemWise";
//  for (auto &node : graph->GetDirectNode()) {
//    auto op = node->GetOpDesc();
//    auto type = node->GetType();
//    if (type != OP_TYPE_PLACE_HOLDER &&
//        type != OP_TYPE_END && type != "StridedSliceGrad") {
//      ge::AttrUtils::SetStr(op, kOpPattern, ele_pattern);
//      const char tbe_bin[] = "tbe_bin";
//      vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
//      ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
//          node->GetName(), std::move(buffer));
//      op->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
//    }
//  }
//  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;
//  std::shared_ptr<FusionCycleDetector> cycle_detector;
//  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
//  cycle_detector->Initialize(*graph);
//
//  auto create_fn1 = []() -> BufferFusionPassBase * { return new (std::nothrow) TbeMultiOutputFusionPass(); };
//  BufferFusionPassRunner *test_pass1 = new (std::nothrow)
//      BufferFusionPassRunner("test_pass1", create_fn1, cycle_detector);
//  auto create_fn2= []() -> BufferFusionPassBase * { return new (std::nothrow) TbeEltwiseFusionPass(); };
//  BufferFusionPassRunner *test_pass2 = new (std::nothrow)
//      BufferFusionPassRunner("test_pass2", create_fn2, cycle_detector);
//  test_pass1->Run(*graph);
//  test_pass2->Run(*graph);
//  auto_buffer_fusion_ptr_->Run(*graph);
//  ge::GraphUtils::DumpGEGraphToOnnx(*graph, "before");
//  // create fused Graph, and merge matched sub-graphs into fusion ops
//  EXPECT_EQ(fe::SUCCESS, sub_graph_optimizer_ptr_->BuildFusionGraph(*graph));
//  ge::GraphUtils::DumpGEGraphToOnnx(*graph, "after");
//}
