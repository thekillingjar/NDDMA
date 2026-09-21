
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <utility>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"

#include "graph/utils/graph_utils_ex.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "platform_context.h"
#include "ascgen_log.h"

#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include "depends/runtime/src/runtime_stub.h"
#include <gtest/gtest.h>

using namespace std;
using namespace testing;

namespace ge {
class LoopNodeLoweringUTV2 : public testing::Test {
public:
protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("Hi Lowering graph"));
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
    RegisterAllOpCreator();
  }
  void TearDown() override {
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

template <typename T>
es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  return result;
}

REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                          DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16,       DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                                   DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                                   DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                                   DT_BF16,       DT_BOOL}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

TEST_F(LoopNodeLoweringUTV2, LoweringSplitInt32) {
  [this]() {
    es_graph_->CreateInput(0, "split_dim", nullptr);
    auto data = es_graph_->CreateInput(1, "x", nullptr);
    data.SetSymbolShape({"s0", "6"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto outputs = es::Split(split_dim, data, 2);
    outputs[0].SetSymbolShape({"s0", "3"});
    outputs[1].SetSymbolShape({"s0", "3"});
    es_graph_->SetOutput(outputs[0], 0);
    es_graph_->SetOutput(outputs[1], 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto split = cg->FindNode("Split_1");
  ASSERT_NE(split, nullptr);
  char soc_version[128] = {};
  auto res = rtGetSocVersion(soc_version, 128U);
  GELOGI("soc_version: %s", soc_version);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(split->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.StoreSplit([\"Split_1:0\"], [tmp0], split_dim=1)\n");
  auto kernel1 = ge::loop::GetKernelBox(split->GetOutDataAnchor(1));
  ASSERT_FALSE(kernel1.IsExternKernel());
  EXPECT_EQ(kernel1.Readable(),
            "tmp0 = ops.Load(\"x:0\")\n"
            "tmp1 = ops.StoreSplit([\"Split_1:1\"], [tmp0], split_dim=1)\n");
}
}  // namespace ge
