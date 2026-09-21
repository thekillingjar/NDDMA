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
#include <dlfcn.h>
#include <dirent.h>
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "aicpu_engine/engine/aicpu_engine.h"
#define private public
#include "aicpu_engine/optimizer/aicpu_optimizer.h"
#include "common/util/cpu_engine_util.h"
#undef private
#include "util/util.h"
#include "utils/graph_utils.h"
#include "util/util_stub.h"
#include "config/config_file.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "common/sgt_slice_type.h"
#include "ge/ge_api_types.h"

using namespace aicpu;
using namespace ge;
using namespace std;

namespace {
const string kOpNoTiling = "_op_no_tiling";
}

static void RegisterOpCreator(const std::string &op_type, const std::vector<std::string> &input_names,
                              const std::vector<std::string> &output_names) {
  auto op_creator = [op_type, input_names, output_names](const std::string &name) -> Operator {
    auto op_desc = make_shared<OpDesc>(name, op_type);
    for (const auto &tensor_name : input_names) {
      op_desc->AddInputDesc(tensor_name, {});
    }
    for (const auto &tensor_name : output_names) {
      op_desc->AddOutputDesc(tensor_name, {});
    }
    return OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  };
  OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator);
}

TEST(AicpuGraphOptimizer, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "CUSTAICPUKernel,AICPUKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPU_ASCENDGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "AICPUOptimizer");
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ge::GetThreadLocalContext().SetGraphOption(options);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->Initialize(options, nullptr), SUCCESS);
    map<string, string> options2;
    ge::GetThreadLocalContext().SetGraphOption(options2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->Initialize(options, nullptr), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_SUCCESS_001)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->SetAttr("test_str", GeAttrValue::CreateFrom<std::string>("test_str"));
    Op2DescPtr->SetAttr("test_float", GeAttrValue::CreateFrom<float>(3.14));
    Op2DescPtr->SetAttr("test_bool", GeAttrValue::CreateFrom<bool>(1));
    Op2DescPtr->SetAttr("test_int", GeAttrValue::CreateFrom<int64_t>(10));
    GeTensorPtr attrTensorPtr = make_shared<GeTensor>(tensor1);
    Op2DescPtr->SetAttr("test_tensor",
                        GeAttrValue::CreateFrom<GeTensorPtr>(attrTensorPtr));
    Op2DescPtr->SetAttr("test_data_type",
                        GeAttrValue::CreateFrom<DataType>(DT_FLOAT));
    Op2DescPtr->SetAttr("test_list_float",
                        GeAttrValue::CreateFrom<std::vector<float>>({1.2, 2.9}));
    Op2DescPtr->SetAttr("test_list_int",
                        GeAttrValue::CreateFrom<std::vector<int64_t>>({0, 1, 2}));
    vector<vector<int64_t>> attrListListInt = {{1},{2}};
    Op2DescPtr->SetAttr("test_list_list_int",
                        GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(attrListListInt));
    Op2DescPtr->SetAttr("test_list_datatype",
                        GeAttrValue::CreateFrom<std::vector<DataType>>({DT_FLOAT, DT_INT8}));
    Op2DescPtr->SetAttr("test_list_str",
                        GeAttrValue::CreateFrom<std::vector<std::string>>({"str1", "str2"}));
    Op2DescPtr->SetAttr("test_warn", GeAttrValue::CreateFrom<GeTensorDesc>(tensor1));

    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_001) {
  map<string, GraphOptimizerPtr> graphOptimizers;
  GetGraphOptimizerObjs(graphOptimizers);
  ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

  shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
  OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

  vector<int64_t> tensorShape = {1, 2, 3, 4};
  GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
  tensor1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor1.SetOriginShape(GeShape(tensorShape));

  opDescPtr->AddOutputDesc("z", tensor1);
  auto node1 = graphPtr->AddNode(opDescPtr);
  OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");

  Op1DescPtr->AddOutputDesc("z", tensor1);
  auto node2 = graphPtr->AddNode(Op1DescPtr);

  node2->AddLinkFrom(node1);

  OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
  Op2DescPtr->SetAttr("test_str", GeAttrValue::CreateFrom<std::string>("test_str"));
  Op2DescPtr->SetAttr("test_float", GeAttrValue::CreateFrom<float>(3.14));
  Op2DescPtr->SetAttr("test_bool", GeAttrValue::CreateFrom<bool>(1));
  Op2DescPtr->SetAttr("test_int", GeAttrValue::CreateFrom<int64_t>(10));
  GeTensorPtr attrTensorPtr = make_shared<GeTensor>(tensor1);
  Op2DescPtr->SetAttr("test_tensor", GeAttrValue::CreateFrom<GeTensorPtr>(attrTensorPtr));
  Op2DescPtr->SetAttr("test_data_type", GeAttrValue::CreateFrom<DataType>(DT_FLOAT));
  Op2DescPtr->SetAttr("test_list_float", GeAttrValue::CreateFrom<std::vector<float>>({1.2, 2.9}));
  Op2DescPtr->SetAttr("test_list_int", GeAttrValue::CreateFrom<std::vector<int64_t>>({0, 1, 2}));
  vector<vector<int64_t>> attrListListInt = {{1}, {2}};
  Op2DescPtr->SetAttr("test_list_list_int",
                      GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(attrListListInt));
  Op2DescPtr->SetAttr("test_list_datatype", GeAttrValue::CreateFrom<std::vector<DataType>>({DT_FLOAT, DT_INT8}));
  Op2DescPtr->SetAttr("test_list_str", GeAttrValue::CreateFrom<std::vector<std::string>>({"str1", "str2"}));
  Op2DescPtr->SetAttr("test_warn", GeAttrValue::CreateFrom<GeTensorDesc>(tensor1));

  Op2DescPtr->AddOutputDesc("z", tensor1);
  auto node3 = graphPtr->AddNode(Op2DescPtr);

  node3->AddLinkFrom(node2);
  ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_002) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph_02");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 1, 3, -1};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("x", "Test");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("y", "Test");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_003) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = std::make_shared<ComputeGraph>("test");
    ge::OpDescPtr opdata = std::make_shared<ge::OpDesc>("test_op", "Data");
    GeTensorDesc dataInput = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    GeTensorDesc dataOutput = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    opdata->AddInputDesc("x",dataInput);
    opdata->AddOutputDesc("y",dataOutput);
    auto nodeData = graphPtr->AddNode(opdata);
    
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("AsString", "AsString");
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    opDescPtr->AddInputDesc("x",xGeTensor);
    opDescPtr->AddOutputDesc("y",yGeTensor);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    nodePtr->AddLinkFrom(nodeData);

    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_004) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = std::make_shared<ComputeGraph>("test");
    ge::OpDescPtr opdata = std::make_shared<ge::OpDesc>("test_op", "test_op");
    GeTensorDesc dataInput = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    GeTensorDesc dataOutput = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    opdata->AddInputDesc("x",dataInput);
    opdata->AddOutputDesc("y",dataOutput);
    auto nodeData = graphPtr->AddNode(opdata);
    
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("NetOutput", "NetOutput");
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(),FORMAT_NCHW,DT_STRING);
    opDescPtr->AddInputDesc("y",xGeTensor);
    opDescPtr->AddOutputDesc("z",yGeTensor);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    nodePtr->AddLinkFrom(nodeData);

    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_005) {
  map<string, GraphOptimizerPtr> graphOptimizers;
  GetGraphOptimizerObjs(graphOptimizers);
  ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

  shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
  OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

  vector<int64_t> tensorShape = {1, 2, 3, 4};
  GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
  tensor1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor1.SetOriginShape(GeShape(tensorShape));

  opDescPtr->AddOutputDesc("z", tensor1);
  auto node1 = graphPtr->AddNode(opDescPtr);
  OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
  AttrUtils::SetBool(Op1DescPtr, "_align_64_bytes_flag", true);

  Op1DescPtr->AddOutputDesc("z", tensor1);
  auto node2 = graphPtr->AddNode(Op1DescPtr);

  node2->AddLinkFrom(node1);

  OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
  AttrUtils::SetBool(Op2DescPtr, "_align_64_bytes_flag", true);

  Op2DescPtr->AddOutputDesc("z", tensor1);
  auto node3 = graphPtr->AddNode(Op2DescPtr);

  node3->AddLinkFrom(node2);

  OpDescPtr Op3DescPtr = make_shared<OpDesc>("conv", "Conv2D");
  Op2DescPtr->AddOutputDesc("z", tensor1);
  auto node4 = graphPtr->AddNode(Op3DescPtr);
  
  node4->AddLinkFrom(node3);

  ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);

  bool is_align_64_bytes_add1 = false;
  (void)AttrUtils::GetBool(Op1DescPtr, "_align_64_bytes_flag", is_align_64_bytes_add1);
  ASSERT_EQ(is_align_64_bytes_add1, false);

  bool is_align_64_bytes_add2 = false; 
  (void)AttrUtils::GetBool(Op2DescPtr, "_align_64_bytes_flag", is_align_64_bytes_add2);
  ASSERT_EQ(is_align_64_bytes_add2, false);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_006) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph_02");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 1, 3, -1};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("x", "TestAsync");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
    bool is_blocking_op = false;
    (void)ge::AttrUtils::GetBool(Op1DescPtr, "_is_blocking_op", is_blocking_op);
    ASSERT_EQ(is_blocking_op, true);
}

TEST(AicpuGraphOptimizer, OptimizeOriginalGraph_SUCCESS_007) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph_02");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 1, 3, -1};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("OutfeedEnqueueOpV2", "OutfeedEnqueueOpV2");

    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
    int32_t async_op_timeout = 1;
    (void)ge::AttrUtils::GetInt(Op1DescPtr, "_blocking_op_timeout", async_op_timeout);
    ASSERT_EQ(async_op_timeout, 0);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_SUCCESS_002)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("RandomChoiceWithMask", "RandomChoiceWithMask");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("FrameworkOpCase", kFrameworkOp);
    ge::AttrUtils::SetStr(Op2DescPtr, kOriginalType, kFrameworkOp);

    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ge::AttrUtils::SetInt(graphPtr, kAttrNameRootGraphId, 99);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_SUCCESS_004)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph_04");
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 1;
    thread_slice_map.thread_mode = 1;
    thread_slice_map.input_tensor_slice = {{{{0, 1}, {1, 2}}}};
    thread_slice_map.output_tensor_slice = {{{{0, 1}, {1, 2}}}};
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    vector<int64_t> tensorShape = {1,2,3,4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z",tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("Add", "Add");
    Op1DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("Cast", "Cast");

    Op2DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraphFftsCast_SUCCESS)
{
    RegisterOpCreator("Cast", {"x"}, {"y"});
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = 0;
    ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));
    opDescPtr->AddOutputDesc("z", tensor1);
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, tsmp_ptr);
    ge::AttrUtils::SetInt(opDescPtr, "_thread_scope_id", 1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("RandomChoiceWithMask2", "RandomChoiceWithMask2");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    Op1DescPtr->SetExtAttr(kAttrNameSgtStruct, tsmp_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, "_thread_scope_id", 1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);
    OpDescPtr Op2DescPtr = make_shared<OpDesc>("RandomChoiceWithMask2", "RandomChoiceWithMask2");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    Op2DescPtr->SetExtAttr(kAttrNameSgtStruct, tsmp_ptr);
    ge::AttrUtils::SetInt(Op2DescPtr, "_thread_scope_id", 1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), PACKAGE_BIN_FILE);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_SUCCESS_006)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph_04");
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 1;
    thread_slice_map.thread_mode = 0;
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    vector<int64_t> tensorShape = {1,2,3,4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z",tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("Add", "Add");
    Op1DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("Cast", "Cast");

    Op2DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_SUCCESS_007)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph_04");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    vector<int64_t> tensorShape = {-1,-1,3,4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z",tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("Add", "Add");

    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("Cast", "Cast");

    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_FAIL_001)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = 1;
    thread_slice_map.input_tensor_slice = {{{{0, 1}, {1, 2}}}};
    thread_slice_map.output_tensor_slice = {{{{0, 1}, {1, 2}}}};
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    vector<int64_t> tensorShape = {1,2,3,4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z",tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("Add", "Add");
    Op1DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("Cast", "Cast");

    Op2DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, kAttrNameThreadScopeId, 1);
    Op1DescPtr->AddOutputDesc("z",tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}


TEST(AicpuGraphOptimizer, OptimizeFusedGraph_FAIL_002)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = 1;
    thread_slice_map.input_tensor_slice = {{{{0, 1}, {1, 2}}, {{0, 1}, {1, 2}}}};
    thread_slice_map.output_tensor_slice = {{{{0, 1}, {1, 2}}}};
    ffts::ThreadSliceMapPtr slice_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    opDescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(opDescPtr, kAttrNameThreadScopeId, 1);
    vector<int64_t> tensorShape = {2,2,2,4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));
    opDescPtr->AddInputDesc("x", tensor1);
    opDescPtr->AddInputDesc("y", tensor1);
    opDescPtr->AddOutputDesc("z",tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("Cast", "Cast");
    Op2DescPtr->SetExtAttr(kAttrNameSgtStruct, slice_ptr);
    ge::AttrUtils::SetInt(Op2DescPtr, kAttrNameThreadScopeId, 1);
    Op2DescPtr->AddOutputDesc("z",tensor1);
    auto node2 = graphPtr->AddNode(Op2DescPtr);

    node2->AddLinkFrom(node1);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(AicpuGraphOptimizer, OptimizeFusedGraph_OPTIMIZE_CPU_KERNEL_SUCCESS)
{
    RegisterOpCreator("Cast", {"x"}, {"y"});
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));
    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("RandomChoiceWithMask2", "RandomChoiceWithMask2");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);
    OpDescPtr Op2DescPtr = make_shared<OpDesc>("RandomChoiceWithMask2", "RandomChoiceWithMask2");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_ascend_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), PACKAGE_BIN_FILE);
}

TEST(AicpuGraphOptimizer, GetBinFileName_FIALED)
{
    Dl_info dl_info;
    dladdr((void*) GetGraphOptimizerObjs, &dl_info);
    string so_path = dl_info.dli_fname;
    string so_file_path = RealPath(so_path);
    string real_dir_file_path = so_file_path.substr(0, so_file_path.rfind('/') + 1) + "op_impl/";
    string command = "rm -rf " + real_dir_file_path;
    system((char*) command.c_str());
    real_dir_file_path = so_file_path.substr(0, so_file_path.rfind('/') + 1) +
                         "op_impl/custom/cpu/aicpu_kernel/custom_impl";
    command = "mkdir -p " + real_dir_file_path;
    system((char*) command.c_str());
    AicpuOptimizer optimizer;

    OpFullInfo opInfo;
    opInfo.kernelSo = "libcpu_kernels.so";
    opInfo.userDefined = false;
    string bin_file_name;
    string fileSo0 = real_dir_file_path + "/1.so";
    command = "touch " + fileSo0;
    system((char*) command.c_str());
    ASSERT_EQ(optimizer.GetBinFileName(opInfo, real_dir_file_path,
                                       bin_file_name), NO_CPU_KERNELS_SO_EXIST);
    string fileSo1 = real_dir_file_path + "/libcpu_kernels_v1.so";
    command = "touch " + fileSo1;
    system((char*) command.c_str());
    string fileSo2 = real_dir_file_path + "/libcpu_kernels_v2.so";
    command = "touch " + fileSo2;
    system((char*) command.c_str());
    ASSERT_EQ(optimizer.GetBinFileName(opInfo, real_dir_file_path,
                                       bin_file_name), MULTI_CPU_KERNELS_SO_EXIST);

    real_dir_file_path = so_file_path.substr(0, so_file_path.rfind('/') + 1) + "op_impl/";
    command = "rm -rf " + real_dir_file_path;
    system((char*) command.c_str());
}

TEST(AicpuGraphOptimizer, GetCpuKernelSoPath_SUCCESS)
{
    char *path = "/home";
    setenv("ASCEND_AICPU_PATH", path, 1);
    AicpuOptimizer optimizer;
    std::string dir = optimizer.GetCpuKernelSoPath();
    cout<<"!!!"<<dir<<endl;
    ASSERT_EQ(dir, "/home/opp/op_impl/built-in/aicpu/aicpu_kernel/lib/Ascend910");
}

TEST(AicpuGraphOptimizer, GetCustKernelSoPath_SUCCESS)
{
    char *path = "/home";
    setenv("ASCEND_OPP_PATH", path, 1);
    AicpuOptimizer optimizer;
    std::string op_type = "Add";
    std::string dir = optimizer.GetCustKernelSoPath(op_type);
    ASSERT_EQ(dir, "/home/op_impl/custom/cpu/aicpu_kernel/custom_impl/");
}

TEST(AicpuGraphOptimizer, ValidateStr_FAILED)
{
    std::string str = "libcpu_kernels.so";
    std::string mode = "libcpu_kernels_v[0-9]+(\\.[0-9]+){0,2}.so";
    ASSERT_EQ(ValidateStr(str, mode), false);

    str = "libcpu_kernels_v2.1.0.so";
    ASSERT_EQ(ValidateStr(str, mode), true);

    mode = "libcpu_kernels_v[0-9]]]]]]]]]]]]+";
    ASSERT_EQ(ValidateStr(str, mode), false);

    AicpuOptimizer optimizer;
    optimizer.CheckAndSetSocVersion("Ascend310");
    optimizer.CheckAndSetSocVersion("Ascend910A");
    optimizer.CheckAndSetSocVersion("Ascend710A");
    optimizer.CheckAndSetSocVersion("Ascend310P");
    optimizer.CheckAndSetSocVersion("test");

    map<string, string> options;
    Initialize(options);
    options[SOC_VERSION] = "Ascend910";
    Initialize(options);
}

TEST(AicpuGraphOptimizer, Util_GetOriginType)
{
    std::string op_type;
    OpDescPtr op_desc_ptr = make_shared<OpDesc>("fmk_op", kFrameworkOp);
    ASSERT_NE(GetOriginalType(op_desc_ptr, op_type), SUCCESS);

    AttrUtils::SetStr(op_desc_ptr, kOriginalType, "IteratorV2");
    ASSERT_EQ(GetOriginalType(op_desc_ptr, op_type), SUCCESS);
    ASSERT_EQ(op_type, "IteratorV2");
}


TEST(AicpuGraphOptimizer, Util_IsNoTiling)
{
    OpDescPtr op_desc_ptr = make_shared<OpDesc>("fmk_op", kFrameworkOp);
    ASSERT_EQ(IsNotiling(op_desc_ptr), false);
}

TEST(AicpuGraphOptimizer, GetCustKernelSoPath_From_ASCEND_CUSTOM_OPP_PATH)
{
    char *path = "/home";
    setenv("ASCEND_CUSTOM_OPP_PATH", path, 1);
    AicpuOptimizer optimizer;
    std::string op_type = "Add";
    std::string dir = optimizer.GetCustKernelSoPath(op_type);
    ASSERT_EQ(dir, "/home/op_impl/custom/cpu/aicpu_kernel/custom_impl/");
}

TEST(AicpuGraphOptimizer, InsertAicpuNodeDefAttrToOp_Deterministic)
{
    OpDescPtr op_desc_ptr = make_shared<OpDesc>("Add", "Add");
    aicpuops::NodeDef node_def;
    BuildAicpuNodeDef(op_desc_ptr, node_def);
    auto ret = InsertAicpuNodeDefAttrToOp(op_desc_ptr, node_def, kCustomizedOpDef);
    ASSERT_EQ(ret, SUCCESS);
    ge::Buffer buffer_1;
    ge::AttrUtils::GetBytes(op_desc_ptr, kCustomizedOpDef, buffer_1);
    std::string str1(reinterpret_cast<const char *>(buffer_1.GetData()), buffer_1.GetSize());
    ret = InsertAicpuNodeDefAttrToOp(op_desc_ptr, node_def, kCustomizedOpDef);
    ASSERT_EQ(ret, SUCCESS);
    ge::Buffer buffer_2;
    ge::AttrUtils::GetBytes(op_desc_ptr, kCustomizedOpDef, buffer_2);
    std::string str2(reinterpret_cast<const char *>(buffer_2.GetData()), buffer_2.GetSize());
    ASSERT_EQ(str1, str2);
}

TEST(AicpuGraphOptimizer, test_GENERATETRANSPOSE_001) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_create_tranpose");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 1, 3, -1};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("x", "Test");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("y", "Test");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    GenerateTransposeBeforeNode(*(graphPtr.get()), node2);
}

TEST(AicpuGraphOptimizer, test_GENERATETRANSPOSE_002) {
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_ascend_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_create_tranpose");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 1, 3, -1};
    GeTensorDesc tensor1(GeShape(tensorShape), ge::FORMAT_NCHW, ge::DT_INT32);
    tensor1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("x", "Test");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("y", "Test");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    GenerateTransposeAfterNode(*(graphPtr.get()), node2);
}

TEST(AicpuGraphOptimizer, test_INSERTTRANSPOSE_FAIEL_001) {
    OpDescPtr opDescPtr = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape = {2, 4, 2, 2, 2};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCDHW, DT_FLOAT);
    opDescPtr->AddInputDesc("x1", tensor1);
    opDescPtr->AddInputDesc("x2", tensor1);
    opDescPtr->AddOutputDesc("y", tensor1);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_grid_sampler_3d");
    ge::NodePtr curr_node = graphPtr->AddNode(opDescPtr);
    Status status = InsertTransposeNode(opDescPtr, *(graphPtr.get()), curr_node);
    EXPECT_EQ(status, GRAPH_FAILED);
}

TEST(AicpuGraphOptimizer, test_GENERATETRANSPOSE_SUCCESS_002) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_grid_sampler_3d");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
  OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
  OpDescPtr fixpipe = std::make_shared<OpDesc>("fixpipe", "FixPipe");
  OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");

  // add descriptor
  vector<int64_t> dims = {1, 4, 4, 4, 2};
  GeShape shape(dims);

  GeTensorDesc in_desc2(shape);
  in_desc2.SetFormat(FORMAT_NCDHW);
  in_desc2.SetOriginFormat(FORMAT_NCDHW);
  in_desc2.SetDataType(DT_FLOAT);
  data->AddOutputDesc("x", in_desc2);
  data2->AddInputDesc("x", in_desc2);
  data2->AddOutputDesc("y", in_desc2);
  fixpipe->AddInputDesc("x", in_desc2);
  fixpipe->AddOutputDesc("y", in_desc2);
  out->AddInputDesc("x", in_desc2);

  NodePtr data_node = graph->AddNode(data);
  NodePtr data2_node = graph->AddNode(data2);
  NodePtr fixpipe_node = graph->AddNode(fixpipe);
  NodePtr out_node = graph->AddNode(out);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), data2_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), fixpipe_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(fixpipe_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  Status status = InsertTransposeNode(data2, *(graph.get()), data2_node);
  EXPECT_EQ(status, GRAPH_SUCCESS);
}