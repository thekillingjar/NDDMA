/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "elewise/pad_api_call.h"

using namespace ge::ops;
using namespace codegen;
using namespace ge::ascir_op;
using namespace testing;
using namespace codegen;
namespace ge{
namespace ascir{
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPadTmpSize(const ge::AscNode &node);

class PadApiCallTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

/* 补充Pad CodeGen UT测试用例 */
TEST(PadApiCallTest, PadApiCall_0) {
    ge::AscGraph graph("test_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    /* x是否需要指定位宽，调度API函数需要指定 */
    Data x_op("x", graph);
    Load load_op("load");
    ge::ascir_op::Pad pad_op("Pad");
    graph.AddNode(load_op);
    graph.AddNode(pad_op);

    //load_op.x.dtype = ge::DT_FLOAT;
    load_op.x = x_op.y;
    load_op.y.dtype = ge::DT_FLOAT;

    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id};
    *load_op.y.repeats = {s0,      s1, s2};
    *load_op.y.strides = {s1 * s2, s2, One};

    /* TODO：perm成员赋值方式 */
    pad_op.x = load_op.y;
    //pad_op.perm = std::vector<int32>{0, 2, 1};

    *pad_op.y.axis = {z1.id, z0.id, z2.id};
    *pad_op.y.repeats = {s1,      s0, s2};
    *pad_op.y.strides = {s0 * s2, s2, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
    load->attr.api.type = ge::ApiType::kAPITypeCompute;
    load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;

    /*  TODO:无循环轴怎么配置*/
    load->attr.sched.loop_axis = z0.id;

    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
    load->outputs[0].attr.vectorized_strides = {s1 * s2, s2, One};
    load->outputs[0].attr.dtype = ge::DT_FLOAT;
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto pad = graph.FindNode("Pad");
    pad->attr.api.type = ge::ApiType::kAPITypeCompute;
    pad->attr.api.unit = ge::ComputeUnit::kUnitVector;

    /* TODO:不配置循环轴 */
    pad->attr.sched.loop_axis = z0.id;

    /* TODO：新增perm节点 ？？ */
    pad->outputs[0].attr.vectorized_axis = {z1.id, z0.id, z2.id};
    pad->outputs[0].attr.vectorized_strides = {s0 * s2, s2, One};
    pad->outputs[0].attr.dtype = ge::DT_FLOAT;
    pad->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
    pad->outputs[0].attr.mem.tensor_id = 1;
    pad->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    pad->outputs[0].attr.que.id = 2;
    pad->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);

    /*  TODO：AddTensor只需要add输出吗？ Perm是否需要体现 */
    tpipe.AddTensor(load->outputs[0]);
    tpipe.AddTensor(pad->outputs[0]);

    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddAxis(z2);
    tiler.AddSizeVar(ge::SizeVar(s0));
    tiler.AddSizeVar(ge::SizeVar(s1));
    tiler.AddSizeVar(ge::SizeVar(s2));
    tiler.SetTilingCaseId(0);

    std::vector<std::unique_ptr<ge::TmpBufDesc>> resultCalc = ge::ascir::CalcPadTmpSize(*pad);
    ASSERT_EQ(resultCalc.size(), 1);
    ASSERT_EQ(resultCalc[0]->size, 32*32);

    codegen::ApiTensor x;
    x.id = load->outputs[0].attr.mem.tensor_id;
    codegen::PadApiCall call("Pad");
    EXPECT_EQ(call.Init(pad), 0);

    call.inputs.push_back(&x);

    std::string result;
    call.Generate(tpipe, vector<ge::AxisId>{}, result);

    EXPECT_EQ(result, std::string{
            "AscendC::DataCopy(local_1[0], local_0[0], KernelUtils::BlkAlign<float>(z1_actual_size * z0_actual_size * z2_actual_size));\n"
    });
}

/* 补充Pad CodeGen UT测试用例 */
TEST(PadApiCallTest, PadApiCall_1) {
    ge::AscGraph graph("test_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2);

    /* x是否需要指定位宽，调度API函数需要指定 */
    Data x_op("x", graph);
    Load load_op("load");
    ge::ascir_op::Pad pad_op("Pad");
    graph.AddNode(load_op);
    graph.AddNode(pad_op);

    //load_op.x.dtype = ge::DT_FLOAT;
    load_op.x = x_op.y;
    load_op.y.dtype = ge::DT_FLOAT;

    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id};
    *load_op.y.repeats = {s0,      s1, s2};
    *load_op.y.strides = {s1 * s2, s2, One};

    /* TODO：perm成员赋值方式 */
    pad_op.x = load_op.y;
    //pad_op.perm = std::vector<int32>{0, 2, 1};

    *pad_op.y.axis = {z1.id, z0.id, z2.id};
    *pad_op.y.repeats = {s1,      s0, s2};
    *pad_op.y.strides = {s0 * s2, s2, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
    load->attr.api.type = ge::ApiType::kAPITypeCompute;
    load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;

    /*  TODO:无循环轴怎么配置*/
    load->attr.sched.loop_axis = z0.id;

    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
    load->outputs[0].attr.vectorized_strides = {s1 * s2, s2, One};
    load->outputs[0].attr.dtype = ge::DT_FLOAT;
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto pad = graph.FindNode("Pad");
    pad->attr.api.type = ge::ApiType::kAPITypeCompute;
    pad->attr.api.unit = ge::ComputeUnit::kUnitVector;
    pad->attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};

    /* TODO:不配置循环轴 */
    pad->attr.sched.loop_axis = z0.id;

    /* TODO：新增perm节点 ？？ */
    pad->outputs[0].attr.vectorized_axis = {z1.id, z0.id, z2.id};
    pad->outputs[0].attr.vectorized_strides = {ge::Symbol(2) * s0 * s2, ge::Symbol(2) * s2, One};
    pad->outputs[0].attr.dtype = ge::DT_FLOAT;
    pad->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
    pad->outputs[0].attr.mem.tensor_id = 1;
    pad->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    pad->outputs[0].attr.que.id = 2;
    pad->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);

    /*  TODO：AddTensor只需要add输出吗？ Perm是否需要体现 */
    tpipe.AddTensor(load->outputs[0]);
    tpipe.AddTensor(pad->outputs[0]);

    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddAxis(z2);
    tiler.AddSizeVar(ge::SizeVar(s0));
    tiler.AddSizeVar(ge::SizeVar(s1));
    tiler.AddSizeVar(ge::SizeVar(s2));
    tiler.SetTilingCaseId(0);

    std::vector<std::unique_ptr<ge::TmpBufDesc>> resultCalc = ge::ascir::CalcPadTmpSize(*pad);
    ASSERT_EQ(resultCalc.size(), 1);
    ASSERT_EQ(resultCalc[0]->size, 32*32);

    codegen::ApiTensor x;
    x.id = load->outputs[0].attr.mem.tensor_id;
    codegen::PadApiCall call("Pad");
    EXPECT_EQ(call.Init(pad), 0);

    call.inputs.push_back(&x);

    std::string result;
    call.Generate(tpipe, vector<ge::AxisId>{}, result);

    EXPECT_EQ(result, std::string{
            "if ((2 * t->s2) != t->s2) {\nAscendC::PadParams padParams = {0, static_cast<uint16_t>((2 * t->s2) - t->s2), 0};\nPadTiling apiTilingData = t->Pad_tilingData_0;\nAscendC::Pad(local_1[0], local_0[0], padParams, tmp_buf_0, apiTilingData);\n} else {\nAscendC::DataCopy(local_1[0], local_0[0], KernelUtils::BlkAlign<float>(z1_actual_size * z0_actual_size * z2_actual_size));\n}\n"
    });
}
} // namespace ascir
} // namespace ge