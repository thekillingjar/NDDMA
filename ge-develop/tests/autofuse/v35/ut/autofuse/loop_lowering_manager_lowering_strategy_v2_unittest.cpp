
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <chrono>
#include <gtest/gtest.h>

#include "graph/debug/ge_op_types.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"

#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_ir_lowerer.h"
#include "lowering/lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"
#include "platform_context.h"

#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace std;
using namespace testing;
namespace ge {
using namespace autofuse;
const static bool _ = []() {
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_reduce = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_slice = true;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_gather = true;
  return true;
}();

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

class LoopGraphLoweringStrategyUTV2 : public testing::Test {
public:
protected:
  void SetUp() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
    RegisterOpCreatorV2("ZeroLikeStub", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
    RegisterOpCreatorV2("AbsRepeat5", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  }
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopGraphLoweringStrategyUTV2, SkipSplitAsDisabled) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data", nullptr);
    data.SetSymbolShape({"6", "s1", "s2"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto outputs = es::Split(split_dim,data, 2);
    outputs[0].SetSymbolShape({"3", "s1", "s2"});
    outputs[1].SetSymbolShape({"3", "s1", "s2"});
    es_graph_->SetOutput(outputs[0], 0);
    es_graph_->SetOutput(outputs[1], 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto split = cg->FindNode("Split_1");
  ASSERT_NE(split, nullptr);

  auto origin = AutoFuseConfig::LoweringConfig().experimental_lowering_split;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = origin; });
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_split = false;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto kernel = ge::loop::GetKernelBox(split->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}
}  // namespace ge
