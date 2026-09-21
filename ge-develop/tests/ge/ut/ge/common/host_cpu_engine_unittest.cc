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

#include "macro_utils/dt_public_scope.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {

using ConstTenMap = const std::map<std::string, const Tensor>;
using TenMap = std::map<std::string, Tensor>;

class SubHostCpuOp : public HostCpuOp{
public:
    SubHostCpuOp();
    virtual graphStatus Compute(Operator &op,
                              const std::map<std::string, const Tensor> &inputs,
                              std::map<std::string, Tensor> &outputs);
    bool compute_flag_;
    bool outputs_flag_;
};

SubHostCpuOp::SubHostCpuOp(){
    compute_flag_ = true;
    outputs_flag_ = false;
}

graphStatus SubHostCpuOp::Compute(Operator &op, ConstTenMap &inputs, TenMap &outputs){
    if (outputs_flag_){
        outputs["__output0"] = Tensor();
    }
    if (compute_flag_){
        return SUCCESS;
    }
    return FAILED;
}


class UTEST_host_cpu_engine : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
 public:
  NodePtr UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt){
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    std::vector<int64_t> shape = {1, 1, 224, 224};
    tensor_desc->SetShape(GeShape(shape));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);
    tensor_desc->SetOriginFormat(FORMAT_NCHW);
    tensor_desc->SetOriginShape(GeShape(shape));
    tensor_desc->SetOriginDataType(DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < in_cnt; ++i) {
        op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int i = 0; i < out_cnt; ++i) {
        op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    return graph->AddNode(op_desc);
  }
};

TEST_F(UTEST_host_cpu_engine, PrepareOutputs_success) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  op_desc->AddOutputDesc("1", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_BOOL));
  op_desc->AddOutputDesc("2", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT8));
  op_desc->AddOutputDesc("3", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT16));
  op_desc->AddOutputDesc("4", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc("5", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT64));
  op_desc->AddOutputDesc("6", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT8));
  op_desc->AddOutputDesc("7", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT16));
  op_desc->AddOutputDesc("8", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT32));
  op_desc->AddOutputDesc("9", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT64));
  op_desc->AddOutputDesc("10", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_FLOAT16));
  op_desc->AddOutputDesc("11", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_FLOAT));
  op_desc->AddOutputDesc("12", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_DOUBLE));
  op_desc->AddOutputDesc("13", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT4));

  vector<GeTensorPtr> outputs;
  GeTensorPtr value = std::make_shared<GeTensor>();
  for (int32_t i = 0; i < 13; i++) {
    outputs.push_back(value);
  }

  map<std::string, Tensor> named_outputs;
  auto ret = HostCpuEngine::GetInstance().PrepareOutputs(op_desc, outputs, named_outputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(named_outputs.size(), 13);
}

TEST_F(UTEST_host_cpu_engine, PrepareOutputs_need_create_success) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  op_desc->AddOutputDesc("output_1", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32));

  vector<GeTensorPtr> outputs;
  map<std::string, Tensor> named_outputs;
  auto ret = HostCpuEngine::GetInstance().PrepareOutputs(op_desc, outputs, named_outputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(named_outputs.size(), 1);
  EXPECT_EQ(named_outputs["output_1"].GetSize(), 16);
  EXPECT_EQ(named_outputs["output_1"].GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(named_outputs["output_1"].GetTensorDesc().GetShape().GetShapeSize(), 4);
}

TEST_F(UTEST_host_cpu_engine, GetEngineRealPath) {
  auto &instance = HostCpuEngine::GetInstance();
  std::string path = "/tmp";
  EXPECT_EQ(instance.GetEngineRealPath(path), SUCCESS);
}

TEST_F(UTEST_host_cpu_engine, LoadLib) {
  auto &instance = HostCpuEngine::GetInstance();
  std::string path = "/tmp";
  EXPECT_NE(instance.LoadLib(path), SUCCESS);
}

TEST_F(UTEST_host_cpu_engine, PrepareInputs) {
  auto &instance = HostCpuEngine::GetInstance();
  auto op_desc = std::make_shared<OpDesc>("name", "type");
  std::vector<ConstGeTensorPtr> inputs;
  std::map<std::string, const Tensor> named_inputs;
  EXPECT_EQ(instance.PrepareInputs(op_desc, inputs, named_inputs), SUCCESS);
  inputs.push_back(std::make_shared<const GeTensor>());
  EXPECT_EQ(instance.PrepareInputs(op_desc, inputs, named_inputs), PARAM_INVALID);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  EXPECT_EQ(instance.PrepareInputs(op_desc, inputs, named_inputs), SUCCESS);
}

TEST_F(UTEST_host_cpu_engine, PrepareOutputs) {
  auto &instance = HostCpuEngine::GetInstance();
  auto op_desc = std::make_shared<OpDesc>("name", "type");
  std::vector<GeTensorPtr> outputs;
  std::map<std::string, Tensor> named_outputs;
  EXPECT_EQ(instance.PrepareOutputs(op_desc, outputs, named_outputs), SUCCESS);
  outputs.push_back(std::make_shared<GeTensor>());
  EXPECT_EQ(instance.PrepareOutputs(op_desc, outputs, named_outputs), SUCCESS);
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  EXPECT_EQ(instance.PrepareOutputs(op_desc, outputs, named_outputs), SUCCESS);
  auto op_desc1 = std::make_shared<OpDesc>("name1", "type1");
  GeTensorDesc output_desc1;
  output_desc1.SetDataType(DT_STRING);
  op_desc1->AddOutputDesc("output", output_desc1);
  EXPECT_EQ(instance.PrepareOutputs(op_desc1, outputs, named_outputs), NOT_CHANGED);
  auto op_desc2 = std::make_shared<OpDesc>("name2", "type2");
  GeTensorDesc output_desc2;
  output_desc2.SetDataType(DT_STRING);
  auto vec = std::vector<int64_t>();
  vec.push_back(1);
  vec.push_back(1);
  vec.push_back(256);
  vec.push_back(256);
  GeShape shape(vec);
  shape.SetIsUnknownDimNum();
  output_desc2.SetShape(shape);
  op_desc2->AddOutputDesc("output", output_desc2);
  outputs.push_back(std::make_shared<GeTensor>());
  EXPECT_NE(instance.PrepareOutputs(op_desc2, outputs, named_outputs), SUCCESS);
}

TEST_F(UTEST_host_cpu_engine, RunInternal) {
  auto &instance = HostCpuEngine::GetInstance();
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  SubHostCpuOp op_kernel;
  std::map<std::string, const Tensor> named_inputs;
  std::map<std::string, Tensor> named_outputs;
  EXPECT_EQ(instance.RunInternal(op_desc, op_kernel, named_inputs, named_outputs), SUCCESS);
  op_kernel.compute_flag_ = false;
  EXPECT_EQ(instance.RunInternal(op_desc, op_kernel, named_inputs, named_outputs), FAILED);
}

TEST_F(UTEST_host_cpu_engine, Run) {
  auto &instance = HostCpuEngine::GetInstance();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data1", "DATA", 0, 1);
  SubHostCpuOp kernel;
  std::vector<ConstGeTensorPtr> inputs;
  std::vector<GeTensorPtr> outputs;
  outputs.push_back(std::make_shared<GeTensor>());
  EXPECT_EQ(instance.Run(node, kernel, inputs, outputs), SUCCESS);
  kernel.outputs_flag_ = true;
  EXPECT_EQ(instance.Run(node, kernel, inputs, outputs), SUCCESS);
}

TEST_F(UTEST_host_cpu_engine, GetDataNumberTest) {
  auto &instance = HostCpuEngine::GetInstance();
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  op_desc->AddOutputDesc("1", GeTensorDesc(GeShape({-2, -2}), FORMAT_NCHW, DT_INT8));

  std::vector<GeTensorPtr> outputs;
  std::map<std::string, Tensor> named_outputs;
  auto ret = instance.PrepareOutputs(op_desc, outputs, named_outputs);
  EXPECT_EQ(ret, SUCCESS);
}

// TEST_F(UTEST_host_cpu_engine, LoadLibSuccess) {
//   auto &instance = HostCpuEngine::GetInstance();
//   std::string caseDir = __FILE__;
//   std::size_t idx = caseDir.find_last_of("/");
//   caseDir = caseDir.substr(0, idx);
//   std::string path = caseDir + "/libc_sec.so";
//   auto ret = instance.LoadLib(path);
//   EXPECT_EQ(ret, SUCCESS);
// }
}  // namespace ge
