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
#define protected public
#define private public
#include "engine/tf_engine.h"
#include "tf_optimizer/tf_optimizer_utils.h"
#include "tf_optimizer/tf_optimizer.h"
#include "tf_optimizer/tensorflow_util.h"
#include "ir2tf/ir2tf_base_parser.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#undef private
#undef protected

#include "util/tf_util.h"
#include "config/config_file.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/option/optimization_option.h"
#include "register/optimization_option_registry.h"
#include "common/sgt_slice_type.h"

using namespace aicpu;
using namespace ge;
using namespace std;

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

TEST(TfGraphOptimizer, Initialize_SUCCESS)
{
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    string kernelConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUOpsKernel", kernelConfig), true);
    ASSERT_EQ(kernelConfig, "TFKernel");
    string optimizerConfig;
    ASSERT_EQ(ConfigFile::GetInstance().GetValue("DNN_VM_AICPUGraphOptimizer", optimizerConfig), true);
    ASSERT_EQ(optimizerConfig, "TFOptimizer");
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeOriginalGraphJudgeInsert_SUCCESS)
{
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("kFrameworkOp", kFrameworkOp);
    auto node0 = graphPtr->AddNode(opDescPtr0);
    //coverage format_agnostic is true
    OpDescPtr opDescPtr1 = make_shared<OpDesc>("Add", "Add");
    vector<int64_t> tensorShape0 = {1, 5, 5, 3, 1};
    GeTensorDesc tensor0(GeShape(tensorShape0), FORMAT_NCHW, DT_INT32);
    opDescPtr1->AddOutputDesc("z", tensor0);
    auto node1 = graphPtr->AddNode(opDescPtr1);
    node1->AddLinkFrom(node0);

    OpDescPtr opDescPtr2 = make_shared<OpDesc>("NOTEXISTS", "NOTEXISTS");
    opDescPtr2->AddOutputDesc("z", tensor0);
    auto node2 = graphPtr->AddNode(opDescPtr2);
    node2->AddLinkFrom(node1);

    OpDescPtr opDescPtr3 = make_shared<OpDesc>("Test", "Test");
    opDescPtr3->AddOutputDesc("z", tensor0);
    auto node3 = graphPtr->AddNode(opDescPtr3);
    node3->AddLinkFrom(node2);

    OpDescPtr opDescPtr4 = make_shared<OpDesc>("Test_success", "Test");
    vector<int64_t> tensorShape11 = {1, 5, 5, 3};
    GeTensorDesc tensor11(GeShape(tensorShape11), FORMAT_NHWC, DT_INT32);
    opDescPtr4->AddInputDesc("images", tensor11);
    vector<int64_t> tensorShape21 = {2};
    GeTensorDesc tensor21(GeShape(tensorShape21), FORMAT_ND, DT_INT32);
    opDescPtr4->AddInputDesc("size", tensor21);
    vector<int64_t> tensorShape31 = {1, 4, 4, 3};
    GeTensorDesc tensor31(GeShape(tensorShape31), FORMAT_NHWC, DT_INT32);
    opDescPtr4->AddOutputDesc("y", tensor31);
    auto node4 = graphPtr->AddNode(opDescPtr4);
    node4->AddLinkFrom(node3);

    OpDescPtr opDescPtr = make_shared<OpDesc>("ResizeArea02", "ResizeArea");
    vector<int64_t> tensorShape12 = {1, 5, 5, 3};
    GeTensorDesc tensor12(GeShape(tensorShape12), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("images", tensor12);
    vector<int64_t> tensorShape22 = {2};
    GeTensorDesc tensor22(GeShape(tensorShape22), FORMAT_ND, DT_INT32);
    opDescPtr->AddInputDesc("size", tensor22);
    vector<int64_t> tensorShape32 = {1, 4, 4, 3};
    GeTensorDesc tensor32(GeShape(tensorShape32), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddOutputDesc("y", tensor32);
    auto node5 = graphPtr->AddNode(opDescPtr);
    node5->AddLinkFrom(node4);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeOriginalGraphJudgeInsert(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opDescPtr->GetInputDesc("images").GetFormat(), FORMAT_NHWC);
    ASSERT_EQ(opDescPtr->GetOutputDesc("y").GetFormat(), FORMAT_NHWC);
}

TEST(TfGraphOptimizer, OptimizeOriginalGraphJudgeInsert_expect_dvpp_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    OpDescPtr opDescPtr = make_shared<OpDesc>("ResizeArea","ResizeArea");
    vector<int64_t> tensorShape1 = {1,5,5,3};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("images", tensor1);
    vector<int64_t> tensorShape2 = {2};
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_ND, DT_INT32);
    opDescPtr->AddInputDesc("size", tensor2);
    vector<int64_t> tensorShape3 = {1,4,4,3};
    GeTensorDesc tensor3(GeShape(tensorShape3), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddOutputDesc("y", tensor3);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    graphPtr->AddNode(opDescPtr);

    (void)AttrUtils::SetStr(opDescPtr, ATTR_NAME_OP_SPECIFIED_ENGINE_NAME, "DNN_VM_DVPP");
    (void)AttrUtils::SetStr(opDescPtr, ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME, "dvpp_ops_kernel");
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeOriginalGraphJudgeInsert(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(opDescPtr->GetInputDesc("images").GetFormat(), FORMAT_NCHW);
    ASSERT_EQ(opDescPtr->GetOutputDesc("y").GetFormat(), FORMAT_NCHW);
}

TEST(TfGraphOptimizer, OptimizeFusedGraph_EndOp_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    OpDescPtr opDescPtr = make_shared<OpDesc>("end", "End");
    opDescPtr->AddOutputDesc("z", tensor1);
    opDescPtr->AddInputDesc("x", tensor1);
    auto node0 =  graphPtr->AddNode(opDescPtr);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraph_SUCCESS)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr0->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeTfDebugFusedGraph_SUCCESS)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr0->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    (void)AttrUtils::SetStr(Op1DescPtr, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "Add");
    ge::OpDescUtilsEx::SetType(Op1DescPtr, "FrameworkOp");
    (void)AttrUtils::SetBool(Op1DescPtr, kAttrNameTfDebug, true);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->UpdateOpDesc(Op1DescPtr);
    node2->AddLinkFrom(node1);

    ge::OpDescPtr op_desc1 = node1->GetOpDesc();
    std::string op_type1 = op_desc1->GetType();

    ge::OpDescPtr op_desc_ptr = node2->GetOpDesc();
    std::string op_type = op_desc_ptr->GetType();

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    (void)AttrUtils::SetStr(Op2DescPtr, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "Add");
    ge::OpDescUtilsEx::SetType(Op2DescPtr, "FrameworkOp");
    (void)AttrUtils::SetBool(Op2DescPtr, kAttrNameTfDebug, true);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ge::OpDescPtr op_desc_ptr3 = node3->GetOpDesc();
    std::string op_type3 = op_desc_ptr3->GetType();

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraph_TRANSPOSE_CHW_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescPtr OpDescPtr = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor1(GeShape({1}), FORMAT_ND, DT_STRING);
    OpDescPtr->AddOutputDesc("z", tensor1);
    auto node = graphPtr->AddNode(OpDescPtr);

    ge::OpDescPtr Op1DescPtr = make_shared<ge::OpDesc>("DecodeJpeg", "DecodeJpeg");
    Op1DescPtr->AddInputDesc(0, tensor1);
    ge::GeTensorDesc tensor2(ge::GeShape({1,1,3}), ge::FORMAT_ND, ge::DT_UINT8);
    Op1DescPtr->AddOutputDesc("image", tensor2);
    ge::AttrUtils::SetStr(Op1DescPtr, "dst_img_format", "CHW");
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node);

    ge::OpDescPtr Op2DescPtr = make_shared<ge::OpDesc>("End", "End");
    Op2DescPtr->AddInputDesc(0, tensor2);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraph_TRANSPOSE_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescPtr OpDescPtr = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor1(GeShape({1}), FORMAT_ND, DT_STRING);
    OpDescPtr->AddOutputDesc("z", tensor1);
    auto node = graphPtr->AddNode(OpDescPtr);

    ge::OpDescPtr Op1DescPtr = make_shared<ge::OpDesc>("DecodeJpeg", "DecodeJpeg");
    Op1DescPtr->AddInputDesc(0, tensor1);
    ge::GeTensorDesc tensor2(ge::GeShape({1,1,3}), ge::FORMAT_ND, ge::DT_UINT8);
    Op1DescPtr->AddOutputDesc("image", tensor2);
    ge::AttrUtils::SetStr(Op1DescPtr, "dst_img_format", "HWC");
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node);

    ge::OpDescPtr Op2DescPtr = make_shared<ge::OpDesc>("End", "End");
    Op2DescPtr->AddInputDesc(0, tensor2);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraph_BatchMatMulV2_RETURN)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescPtr OpDescPtr1 = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor1(GeShape({2,3,4}), FORMAT_ND, DT_INT32);
    OpDescPtr1->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(OpDescPtr1);

    ge::OpDescPtr OpDescPtr2 = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor2(GeShape({2,4,5}), FORMAT_ND, DT_INT32);
    OpDescPtr2->AddOutputDesc("z", tensor2);
    auto node2 = graphPtr->AddNode(OpDescPtr2);

    ge::OpDescPtr OpDescPtr3 = make_shared<ge::OpDesc>("BatchMatMulV2", "BatchMatMulV2");
    OpDescPtr3->AddInputDesc("x1", tensor1);
    OpDescPtr3->AddInputDesc("x2", tensor2);
    ge::GeTensorDesc tensor3(ge::GeShape({2,3,5}), ge::FORMAT_ND, ge::DT_INT32);
    OpDescPtr3->AddOutputDesc("y", tensor3);
    auto node3 = graphPtr->AddNode(OpDescPtr3);
    node3->AddLinkFrom(node1);
    node3->AddLinkFrom(node2);

    ge::OpDescPtr OpDescPtr6 = make_shared<ge::OpDesc>("End", "End");
    OpDescPtr6->AddInputDesc(0, tensor3);
    auto node6 = graphPtr->AddNode(OpDescPtr6);
    node6->AddLinkFrom(node3);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraph_BatchMatMulV2_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    ge::OpDescPtr OpDescPtr1 = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor1(GeShape({2,3,4}), FORMAT_ND, DT_INT32);
    OpDescPtr1->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(OpDescPtr1);

    ge::OpDescPtr OpDescPtr2 = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor2(GeShape({2,4,5}), FORMAT_ND, DT_INT32);
    OpDescPtr2->AddOutputDesc("z", tensor2);
    auto node2 = graphPtr->AddNode(OpDescPtr2);

    ge::OpDescPtr OpDescPtr4 = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor4(GeShape({0}), FORMAT_RESERVED, DT_UNDEFINED);
    OpDescPtr4->AddOutputDesc("z", tensor4);
    auto node4 = graphPtr->AddNode(OpDescPtr4);

    ge::OpDescPtr OpDescPtr5 = make_shared<ge::OpDesc>("placeholder", "PlaceHolder");
    GeTensorDesc tensor5(GeShape({0}), FORMAT_RESERVED, DT_UNDEFINED);
    OpDescPtr5->AddOutputDesc("z", tensor5);
    auto node5 = graphPtr->AddNode(OpDescPtr5);

    ge::OpDescPtr OpDescPtr3 = make_shared<ge::OpDesc>("BatchMatMulV2", "BatchMatMulV2");
    OpDescPtr3->AddInputDesc("x1", tensor1);
    OpDescPtr3->AddInputDesc("x2", tensor2);
    OpDescPtr3->AddInputDesc("bias", tensor4);
    OpDescPtr3->AddInputDesc("offset_w", tensor5);
    ge::GeTensorDesc tensor3(ge::GeShape({2,3,5}), ge::FORMAT_ND, ge::DT_INT32);
    OpDescPtr3->AddOutputDesc("y", tensor3);
    auto node3 = graphPtr->AddNode(OpDescPtr3);
    node3->AddLinkFrom(node1);
    node3->AddLinkFrom(node2);
    node3->AddLinkFrom(node4);
    node3->AddLinkFrom(node5);

    ge::OpDescPtr OpDescPtr6 = make_shared<ge::OpDesc>("End", "End");
    OpDescPtr6->AddInputDesc(0, tensor3);
    auto node6 = graphPtr->AddNode(OpDescPtr6);
    node6->AddLinkFrom(node3);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraphFFTSAutoMode_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = 0;
    ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("add0", "Add");
    opDescPtr0->AddOutputDesc("z", tensor1);
    opDescPtr0->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    ge::AttrUtils::SetInt(opDescPtr0, "_thread_scope_id", 1);
    ge::AttrUtils::SetBool(opDescPtr0, "_aicpu_private", true);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    Op1DescPtr->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, "_thread_scope_id", 1);
    ge::AttrUtils::SetBool(Op1DescPtr, "_aicpu_private", true);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    Op2DescPtr->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    ge::AttrUtils::SetInt(Op2DescPtr, "_thread_scope_id", 1);
    ge::AttrUtils::SetBool(Op2DescPtr, "_aicpu_private", true);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraphFFTSManualMode_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    ffts::ThreadSliceMap thread_slice_map;
    thread_slice_map.thread_scope_id = 1;
    thread_slice_map.slice_instance_num = 2;
    thread_slice_map.thread_mode = 1;
    ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(thread_slice_map);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("add0", "Add");
    opDescPtr0->AddOutputDesc("z", tensor1);
    opDescPtr0->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    ge::AttrUtils::SetInt(opDescPtr0, "_thread_scope_id", 1);
    ge::AttrUtils::SetBool(opDescPtr0, "_aicpu_private", true);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    Op1DescPtr->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    ge::AttrUtils::SetInt(Op1DescPtr, "_thread_scope_id", 1);
    ge::AttrUtils::SetBool(Op1DescPtr, "_aicpu_private", true);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    Op2DescPtr->SetExtAttr("_sgt_struct_info", tsmp_ptr);
    ge::AttrUtils::SetInt(Op2DescPtr, "_thread_scope_id", 1);
    ge::AttrUtils::SetBool(Op2DescPtr, "_aicpu_private", false);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, CheckAndSetUnknowTypeFMK_SUCCESS)
{
    TfOptimizer tfOpt;
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("add", "Add");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FrameworkOp");
    vector<int64_t> tensorShape = {1, 2, -1, -1};
    GeTensorDesc tensor(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor.SetOriginFormat(FORMAT_NCHW);
    tensor.SetOriginShape(GeShape(tensorShape));
    opDescPtr->AddInputDesc("x1", tensor);
    opDescPtr->AddInputDesc("x2", tensor);
    opDescPtr->AddOutputDesc("z", tensor);
    NodePtr nodePtr = make_shared<Node>(opDescPtr, graphPtr);
    ASSERT_EQ(CheckAndSetUnknowType(nodePtr), ErrorCode::PARAM_INVALID);
}

TEST(TfGraphOptimizer, CheckAndSetUnknowTypeFunction_SUCCESS)
{
    TfOptimizer tfOpt;
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("add", "Add");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FunctionOp");
    vector<int64_t> tensorShape1 = {1, 2, -1, -1};
    vector<int64_t> tensorShape2 = {1, 2, 1, 1};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_NCHW, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape1));
    tensor2.SetOriginFormat(FORMAT_NCHW);
    tensor2.SetOriginShape(GeShape(tensorShape2));
    opDescPtr->AddInputDesc("x1", tensor2);
    opDescPtr->AddInputDesc("x2", tensor2);
    opDescPtr->AddOutputDesc("z", tensor1);
    NodePtr nodePtr = make_shared<Node>(opDescPtr, graphPtr);
    ASSERT_EQ(CheckAndSetUnknowType(nodePtr), SUCCESS);
}

TEST(TfGraphOptimizer, SetExceptAbortAttr_SUCCESS)
{
    TfOptimizer TfOptimizer;
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    domi::tensorflow::FunctionDefLibrary func_lib;
    domi::tensorflow::FunctionDef *func_def = func_lib.add_function();
    func_def->mutable_signature()->set_name("IteratorGetNext");

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("Test0", "IteratorGetNext");
    opDescPtr0->AddOutputDesc("z", tensor1);
    NodePtr node0 = make_shared<Node>(opDescPtr0, graphPtr);

    domi::tensorflow::NodeDef nodeDef;
    Ir2tfBaseParser::Instance()->ParseNodeDef(*node0.get(), &nodeDef);
    domi::tensorflow::NodeDef *nodeDefPtr = func_def->add_node_def();
    *nodeDefPtr = nodeDef;
    nodeDefPtr->set_name("IteratorGetNext");

    size_t size_byte = func_lib.ByteSizeLong();
    uint8_t *nodeDefData = new uint8_t[size_byte];
    func_lib.SerializeToArray(nodeDefData, size_byte);
    ge::Buffer funcBuffer = ge::Buffer::CopyFrom(nodeDefData, size_byte);

    OpDescPtr opDescPtr = make_shared<OpDesc>("kFunctionOp", kFunctionOp);
    opDescPtr->AddOutputDesc("z", tensor1);
    ge::AttrUtils::SetBytes(opDescPtr, kTfFuncDef, funcBuffer);
    NodePtr node = make_shared<Node>(opDescPtr, graphPtr);
    graphPtr->AddNode(node);
    ASSERT_EQ(TfOptimizer.SetExceptionAbortAttr(node), SUCCESS);
    delete []nodeDefData;
}

TEST(TfGraphOptimizer, SetExceptAbortAttr_EXISTED_SUCCESS)
{
    TfOptimizer TfOptimizer;
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    domi::tensorflow::FunctionDefLibrary func_lib;
    domi::tensorflow::FunctionDef *func_def = func_lib.add_function();
    func_def->mutable_signature()->set_name("IteratorGetNext");

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("Test0", "IteratorGetNext");
    opDescPtr0->AddOutputDesc("z", tensor1);
    NodePtr node0 = make_shared<Node>(opDescPtr0, graphPtr);

    domi::tensorflow::NodeDef nodeDef;
    Ir2tfBaseParser::Instance()->ParseNodeDef(*node0.get(), &nodeDef);
    domi::tensorflow::NodeDef *nodeDefPtr = func_def->add_node_def();
    *nodeDefPtr = nodeDef;
    nodeDefPtr->set_name("IteratorGetNext");

    size_t size_byte = func_lib.ByteSizeLong();
    uint8_t *nodeDefData = new uint8_t[size_byte];
    func_lib.SerializeToArray(nodeDefData, size_byte);
    ge::Buffer funcBuffer = ge::Buffer::CopyFrom(nodeDefData, size_byte);

    OpDescPtr opDescPtr = make_shared<OpDesc>("kFunctionOp", kFunctionOp);
    opDescPtr->AddOutputDesc("z", tensor1);
    ge::AttrUtils::SetBytes(opDescPtr, kTfFuncDef, funcBuffer);
    ge::AttrUtils::SetBool(opDescPtr, kExceptionAbort, false);
    NodePtr node = make_shared<Node>(opDescPtr, graphPtr);
    graphPtr->AddNode(node);
    ASSERT_EQ(TfOptimizer.SetExceptionAbortAttr(node), SUCCESS);
    delete []nodeDefData;
}

TEST(TfGraphOptimizer, SetExceptAbortAttr_FAILED)
{
    TfOptimizer TfOptimizer;
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("add", "Add");
    ge::OpDescUtilsEx::SetType(opDescPtr, "FunctionOp");
    vector<int64_t> tensorShape1 = {1, 2, -1, -1};
    vector<int64_t> tensorShape2 = {1, 2, 1, 1};
    GeTensorDesc tensor1(GeShape(tensorShape1), FORMAT_NCHW, DT_INT32);
    GeTensorDesc tensor2(GeShape(tensorShape2), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape1));
    tensor2.SetOriginFormat(FORMAT_NCHW);
    tensor2.SetOriginShape(GeShape(tensorShape2));
    opDescPtr->AddInputDesc("x1", tensor2);
    opDescPtr->AddInputDesc("x2", tensor2);
    opDescPtr->AddOutputDesc("z", tensor1);
    NodePtr nodePtr = make_shared<Node>(opDescPtr, graphPtr);
    ASSERT_EQ(TfOptimizer.SetExceptionAbortAttr(nodePtr), FUNC_DEF_NOT_EXIST);
}

TEST(TfGraphOptimizer, CreateNodeDefForGeNodeWithNodeDef_SUCCESS)
{
    TfOptimizer tfOpt;
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("add", "Add");
    Buffer nodeDefBytes;
    AttrUtils::SetBytes(*(opDescPtr.get()), "node_def", nodeDefBytes);
    NodePtr nodePtr = make_shared<Node>(opDescPtr, graphPtr);
    ASSERT_EQ(tfOpt.CreateNodeDefForGeNode(nodePtr), SUCCESS);
}


TEST(TfGraphOptimizer, OptimizeCacheGraph_SUCCESS)
{
    RegisterOpCreator("CacheUpdate", {"x"}, {"y"});
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    // create placehoder1 node
    OpDescPtr placeholderOpDescPtr1 = make_shared<OpDesc>("placeholder1", "PlaceHolder");
    AttrUtils::SetStr(placeholderOpDescPtr1, ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME, "AIcoreEngine");
    NodePtr placeholderNodePtr1 = make_shared<Node>(placeholderOpDescPtr1, graphPtr);
    placeholderNodePtr1->UpdateOpDesc(placeholderOpDescPtr1);
    vector<int64_t> tensorShape = {1};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_ND, DT_INT32);
    placeholderOpDescPtr1->AddOutputDesc("x", tensor1);
    placeholderNodePtr1->Init();
    graphPtr->AddNode(placeholderNodePtr1);

    // create cast node
    OpDescPtr castOpDescPtr = make_shared<OpDesc>("cast", "Cast");
    NodePtr castNodePtr = make_shared<Node>(castOpDescPtr, graphPtr);
    castNodePtr->UpdateOpDesc(castOpDescPtr);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    castOpDescPtr->AddOutputDesc("y", tensor1);
    castOpDescPtr->AddInputDesc("x", tensor1);
    castNodePtr->Init();
    castNodePtr->AddLinkFrom(placeholderNodePtr1);
    graphPtr->AddNode(castNodePtr);

    OpDescPtr endOpDescPtr = make_shared<OpDesc>("end", "End");
    AttrUtils::SetStr(endOpDescPtr, ATTR_NAME_END_REAR_NODE_ENGINE_NAME, "AIcoreEngine");
    AttrUtils::SetStr(endOpDescPtr, "parentOpType", "NetOutput");
    NodePtr endNodePtr = make_shared<Node>(endOpDescPtr, graphPtr);
    endNodePtr->UpdateOpDesc(endOpDescPtr);
    endOpDescPtr->AddOutputDesc("output", tensor1);
    endNodePtr->Init();
    endNodePtr->AddLinkFrom(castNodePtr);
    graphPtr->AddNode(endNodePtr);

    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    map<string, string> options;
    options[ge::SOC_VERSION] = "Hi3796es";
    ASSERT_EQ(Initialize(options), SUCCESS);
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeIsRefTensor_SUCCESS)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), 20000046);
}


TEST(TfGraphOptimizer, OptimizeOriginalGraph_SUCCESS)
{
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    map<string, string> options;
    options[ge::SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeOriginalGraph_SUCCESS_002)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr0->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);

    OpDescPtr Op3DescPtr = make_shared<OpDesc>("conv", "Conv2D");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node4 = graphPtr->AddNode(Op3DescPtr);
    node4->AddLinkFrom(node3);

    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeOriginalGraph(*(graphPtr.get())), SUCCESS);

    bool is_align_64_bytes_add1 = false;
    (void)AttrUtils::GetBool(Op1DescPtr, "_align_64_bytes_flag", is_align_64_bytes_add1);
    ASSERT_EQ(is_align_64_bytes_add1, true);

    bool is_align_64_bytes_add2 = false; 
    (void)AttrUtils::GetBool(Op2DescPtr, "_align_64_bytes_flag", is_align_64_bytes_add2);
    ASSERT_EQ(is_align_64_bytes_add2, true);

    bool is_align_64_bytes_conv = false; 
    (void)AttrUtils::GetBool(Op3DescPtr, "_align_64_bytes_flag", is_align_64_bytes_conv);
    ASSERT_EQ(is_align_64_bytes_conv, false);
}

TEST(TfGraphOptimizer, OptimizeWholeGraph_SUCCESS)
{
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    map<string, string> options;
    options[ge::SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeWholeGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, RefCreateAndInsert_Succeed)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("cast", "Cast");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("end", "End");

    Op1DescPtr->AddOutputDesc("z", tensor1);

    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);


    map<string, string> options;
    options[ge::SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);

    ComputeGraph &graph = *(graphPtr.get());

    OutDataAnchorPtr srcAnchor1 = node1->GetOutDataAnchor(0);
    InDataAnchorPtr dstAnchor1 = node2->GetInDataAnchor(0);
    NodePtr variableNode;
    NodePtr assignNode;
    NodePtr identityNode;

    Status ret = TfVariableGraph::CreateAndInsertVariable(srcAnchor1, dstAnchor1, variableNode, graph);

    ASSERT_EQ(SUCCESS, ret);

    ret = TfVariableGraph::CreateAndInsertAssign(srcAnchor1, dstAnchor1, variableNode, graph);

    ASSERT_EQ(SUCCESS, ret);

    OutDataAnchorPtr srcAnchor3 = node2->GetOutDataAnchor(0);
    InDataAnchorPtr dstAnchor3 = node3->GetInDataAnchor(0);

    ret = TfVariableGraph::CreateAndInsertIdentity(srcAnchor3, dstAnchor3, graph);

    ASSERT_EQ(SUCCESS, ret);
}

TEST(TfGraphOptimizer, RefInsert_Succeed)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    opDescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr Op1DescPtr = make_shared<OpDesc>("cast", "Cast");

    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);

    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("end", "End");

    Op1DescPtr->AddOutputDesc("z", tensor1);

    auto node3 = graphPtr->AddNode(Op2DescPtr);

    node3->AddLinkFrom(node2);


    map<string, string> options;
    options[ge::SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(options), SUCCESS);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);

    ComputeGraph &graph = *(graphPtr.get());

    OutDataAnchorPtr srcAnchor1 = node1->GetOutDataAnchor(0);
    InDataAnchorPtr dstAnchor1 = node2->GetInDataAnchor(0);
    NodePtr variableNode;
    NodePtr assignNode;
    NodePtr identityNode;
    GeTensorDesc tensor2(GeShape(tensorShape), FORMAT_ND, DT_INT32);
    GeTensorDesc tensor3(GeShape(tensorShape), FORMAT_ND, DT_INT32);
    GeTensorDesc tensor4(GeShape(tensorShape), FORMAT_ND, DT_INT32);

    Status ret = TfVariableGraph::CreateVariable(tensor2, graph, variableNode);
    ret = TfVariableGraph::InsertVariable(srcAnchor1, dstAnchor1, variableNode);

    ASSERT_EQ(SUCCESS, ret);

    ret = TfVariableGraph::CreateAssign(tensor2, tensor3, tensor4, graph, assignNode);
    ret = TfVariableGraph::InsertAssign(srcAnchor1, dstAnchor1, variableNode, assignNode);

    ASSERT_EQ(SUCCESS, ret);

    OutDataAnchorPtr srcAnchor3 = node2->GetOutDataAnchor(0);
    InDataAnchorPtr dstAnchor3 = node3->GetInDataAnchor(0);

    ret = TfVariableGraph::CreateIdentity(tensor2, tensor3, graph, identityNode);
    ret = TfVariableGraph::InsertIdentity(srcAnchor3, dstAnchor3, identityNode);

    ASSERT_EQ(SUCCESS, ret);
}

TEST(TfGraphOptimizer, RebuildTfNodeClusterMap_Succeed)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    TfOptimizer MyTfOptimizer;
    map<string, string> myOptions;
    myOptions[ge::SOC_VERSION] = "Ascend910";
    ASSERT_EQ(Initialize(myOptions), SUCCESS);
    map<string, GraphOptimizerPtr> myGraphOptimizers;
    GetGraphOptimizerObjs(myGraphOptimizers);
    ASSERT_NE(myGraphOptimizers["aicpu_tf_optimizer"], nullptr);
    ASSERT_EQ(myGraphOptimizers["aicpu_tf_optimizer"]->Initialize(myOptions, nullptr), SUCCESS);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    OpDescPtr opDescPtr = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape1 = {2,1,3,4};
    GeTensorDesc tensorDesc1(GeShape(tensorShape1), FORMAT_NCHW, DT_INT32);
    tensorDesc1.SetOriginFormat(FORMAT_NCHW);
    tensorDesc1.SetOriginShape(GeShape(tensorShape1));

    opDescPtr->AddOutputDesc("z", tensorDesc1);
    auto node1 = graphPtr->AddNode(opDescPtr);
    OpDescPtr OpDescPtr1 = make_shared<OpDesc>("cast", "Cast");

    OpDescPtr1->AddOutputDesc("z", tensorDesc1);
    auto node2 = graphPtr->AddNode(OpDescPtr1);
    node2->AddLinkFrom(node1);
    OpDescPtr OpDescPtr2 = make_shared<OpDesc>("end", "End");
    OpDescPtr1->AddOutputDesc("z", tensorDesc1);
    auto node3 = graphPtr->AddNode(OpDescPtr2);
    node3->AddLinkFrom(node2);

    OpDescPtr opDescPtr4 = make_shared<OpDesc>("placeholder", "PlaceHolder");

    vector<int64_t> tensorShape4 = {2,1,3,4};
    GeTensorDesc tensorDesc4(GeShape(tensorShape4), FORMAT_NCHW, DT_INT32);
    tensorDesc4.SetOriginFormat(FORMAT_NCHW);
    tensorDesc4.SetOriginShape(GeShape(tensorShape4));

    opDescPtr4->AddOutputDesc("z", tensorDesc4);
    auto node4 = graphPtr->AddNode(opDescPtr4);
    OpDescPtr OpDescPtr5 = make_shared<OpDesc>("cast", "Cast");

    OpDescPtr5->AddOutputDesc("z", tensorDesc4);
    auto node5 = graphPtr->AddNode(OpDescPtr5);
    node5->AddLinkFrom(node4);
    OpDescPtr OpDescPtr6 = make_shared<OpDesc>("end", "End");
    OpDescPtr5->AddOutputDesc("z", tensorDesc4);
    auto node6 = graphPtr->AddNode(OpDescPtr6);
    node6->AddLinkFrom(node5);

    ComputeGraph &graph = *(graphPtr.get());

    OutDataAnchorPtr dataSrcAnchor1 = node1->GetOutDataAnchor(0);
    InDataAnchorPtr dataDstAnchor1 = node2->GetInDataAnchor(0);
    NodePtr variableNode1;
    NodePtr assignNode1;
    NodePtr identityNode1;
    GeTensorDesc tensorDesc2(GeShape(tensorShape1), FORMAT_ND, DT_INT32);
    GeTensorDesc tensorDesc3(GeShape(tensorShape1), FORMAT_ND, DT_INT32);
    GeTensorDesc tensorDesc8(GeShape(tensorShape1), FORMAT_ND, DT_INT32);

    Status ret = TfVariableGraph::CreateVariable(tensorDesc2, graph, variableNode1);
    ret = TfVariableGraph::InsertVariable(dataSrcAnchor1, dataDstAnchor1, variableNode1);

    ASSERT_EQ(SUCCESS, ret);

    ret = TfVariableGraph::CreateAssign(tensorDesc2, tensorDesc3, tensorDesc8, graph, assignNode1);
    ret = TfVariableGraph::InsertAssign(dataSrcAnchor1, dataDstAnchor1, variableNode1, assignNode1);

    ASSERT_EQ(SUCCESS, ret);

    OutDataAnchorPtr srcAnchor3 = node2->GetOutDataAnchor(0);
    InDataAnchorPtr dstAnchor3 = node3->GetInDataAnchor(0);

    ret = TfVariableGraph::CreateIdentity(tensorDesc2, tensorDesc3, graph, identityNode1);
    ret = TfVariableGraph::InsertIdentity(srcAnchor3, dstAnchor3, identityNode1);

    ASSERT_EQ(SUCCESS, ret);

    OutDataAnchorPtr dataSrcAnchor2 = node4->GetOutDataAnchor(0);
    InDataAnchorPtr dataDstAnchor2 = node5->GetInDataAnchor(0);
    NodePtr variableNode2;
    NodePtr assignNode2;
    NodePtr identityNode2;
    GeTensorDesc tensorDesc5(GeShape(tensorShape1), FORMAT_ND, DT_INT32);
    GeTensorDesc tensorDesc6(GeShape(tensorShape1), FORMAT_ND, DT_INT32);
    GeTensorDesc tensorDesc7(GeShape(tensorShape1), FORMAT_ND, DT_INT32);

    ret = TfVariableGraph::CreateVariable(tensorDesc5, graph, variableNode2);
    ret = TfVariableGraph::InsertVariable(dataSrcAnchor2, dataDstAnchor2, variableNode2);

    ASSERT_EQ(SUCCESS, ret);

    ret = TfVariableGraph::CreateAssign(tensorDesc5, tensorDesc6, tensorDesc7, graph, assignNode2);
    ret = TfVariableGraph::InsertAssign(dataSrcAnchor2, dataDstAnchor2, variableNode2, assignNode2);

    ASSERT_EQ(SUCCESS, ret);

    OutDataAnchorPtr dataSrcAnchor5 = node5->GetOutDataAnchor(0);
    InDataAnchorPtr dataDstAnchor5 = node6->GetInDataAnchor(0);

    ret = TfVariableGraph::CreateIdentity(tensorDesc5, tensorDesc6, graph, identityNode2);
    ret = TfVariableGraph::InsertIdentity(dataSrcAnchor5, dataDstAnchor5, identityNode2);

    ASSERT_EQ(SUCCESS, ret);

    std::vector<ge::NodePtr> tf_node_cluster;
    tf_node_cluster.emplace_back(variableNode1);
    tf_node_cluster.emplace_back(assignNode1);
    tf_node_cluster.emplace_back(node2);
    tf_node_cluster.emplace_back(identityNode1);
    tf_node_cluster.emplace_back(variableNode2);
    tf_node_cluster.emplace_back(assignNode2);
    tf_node_cluster.emplace_back(node5);
    tf_node_cluster.emplace_back(identityNode2);

    std::vector<ge::NodePtr> tmp_tf_node_cluster;
    tmp_tf_node_cluster.assign(tf_node_cluster.begin(), tf_node_cluster.end());

    std::unordered_map<std::string, std::vector<ge::NodePtr>> tf_node_cluster_map;
    tf_node_cluster_map["mytestop"] = tmp_tf_node_cluster;
    ASSERT_EQ(1, tf_node_cluster_map.size());

    tf_node_cluster_map = MyTfOptimizer.RebuildTfNodeClusterMap(graph, tf_node_cluster_map);
    ASSERT_EQ(2, tf_node_cluster_map.size());
}

TEST(TfGraphOptimizer, Optimize_FuseNodesForGraph_SUCCESS)
{
    TfOptimizer TfOptimizer;
    std::unordered_map<std::string, ge::NodePtr> tf_isolated_node_map;
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");

    vector<int64_t> tensorShape = {2, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(Op1DescPtr);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    NodePtr node2 = make_shared<Node>(Op2DescPtr, graphPtr);
    node2->Init();
    graphPtr->AddNode(node2);
    node2->AddLinkFrom(node1);

    OpDescPtr Op3DescPtr = make_shared<OpDesc>("add3", "Add");
    Op3DescPtr->AddOutputDesc("z", tensor1);
    NodePtr node3 = make_shared<Node>(Op3DescPtr, graphPtr);
    graphPtr->AddNode(node3);
    node3->AddLinkFrom(node2);

    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(node2);
    ASSERT_EQ(TfOptimizer.FuseNodesForGraph(*(graphPtr.get()), nodeList, tf_isolated_node_map), SUCCESS);
}

TEST(TfGraphOptimizer, Optimize_FuseNodesForGraph_DTSTRING_SUCCESS)
{
    TfOptimizer TfOptimizer;
    std::unordered_map<std::string, ge::NodePtr> tf_isolated_node_map;
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");

    vector<int64_t> tensorShape = {5};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("x", tensor1);
    auto node1 = graphPtr->AddNode(Op1DescPtr);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("x", tensor1);
    (void)ge::AttrUtils::SetBool(Op2DescPtr, "_aicpu_input_output_dtstring", true);
    NodePtr node2 = make_shared<Node>(Op2DescPtr, graphPtr);
    node2->Init();
    graphPtr->AddNode(node2);
    node2->AddLinkFrom(node1);

    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(node2);
    ASSERT_EQ(TfOptimizer.FuseNodesForGraph(*(graphPtr.get()), nodeList, tf_isolated_node_map), SUCCESS);
}

TEST(TfGraphOptimizer, Optimize_OpWithTFBuffer_SUCCES)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    TfOptimizer TfOptimizer;
    GetGraphOptimizerObjs(graphOptimizers);
    std::unordered_map<std::string, ge::NodePtr> tf_isolated_node_map;
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("Test0", "Test");
    opDescPtr0->AddOutputDesc("z", tensor1);
    NodePtr node0 = make_shared<Node>(opDescPtr0, graphPtr);

    domi::tensorflow::NodeDef nodeDef;
    Ir2tfBaseParser::Instance()->ParseNodeDef(*node0.get(), &nodeDef);
    std::string nodeDefStr;
    nodeDef.SerializeToString(&nodeDefStr);
    const uint8_t *nodeDefData =
                   reinterpret_cast<const uint8_t *>(nodeDefStr.data());
    ge::Buffer nodeBuffer;
    nodeBuffer = ge::Buffer::CopyFrom(nodeDefData, nodeDefStr.length());
    ge::AttrUtils::SetBytes(opDescPtr0, kTfNodeDef, nodeBuffer);
    domi::tensorflow::OpDef opDef;
    std::string opDefStr;
    opDef.set_name("Test");
    opDef.SerializeToString(&opDefStr);
    ge::AttrUtils::SetStr(opDescPtr0, kTfOpDef, opDefStr);
    node0->UpdateOpDesc(opDescPtr0);
    graphPtr->AddNode(node0);

    OpDescPtr opDescPtr1 = make_shared<OpDesc>("AddTest", "Add");
    auto node1 = graphPtr->AddNode(opDescPtr1);
    node1->AddLinkFrom(node0);
    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(node0);
    nodeList.push_back(node1);
    ASSERT_EQ(TfOptimizer.FuseNodesForGraph(*(graphPtr.get()), nodeList, tf_isolated_node_map), SUCCESS);
}

TEST(TfGraphOptimizer, Optimize_FunctionOp_FAIL)
{
    map<string, GraphOptimizerPtr> graphOptimizers;
    TfOptimizer TfOptimizer;
    GetGraphOptimizerObjs(graphOptimizers);
    std::unordered_map<std::string, ge::NodePtr> tf_isolated_node_map;
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("kFunctionOp", kFunctionOp);
    opDescPtr0->AddOutputDesc("z", tensor1);
    ge::Buffer funcBuffer;
    uint8_t *nodeDefData = new uint8_t[20];
    domi::tensorflow::FunctionDefLibrary func_lib;
    func_lib.SerializeToArray(nodeDefData, func_lib.ByteSize());
    funcBuffer = ge::Buffer::CopyFrom(nodeDefData, func_lib.ByteSize());
    ge::AttrUtils::SetBytes(opDescPtr0, kTfFuncDef, funcBuffer);
    NodePtr node0 = make_shared<Node>(opDescPtr0, graphPtr);
    graphPtr->AddNode(node0);

    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(node0);
    ASSERT_EQ(TfOptimizer.FuseNodesForGraph(*(graphPtr.get()), nodeList, tf_isolated_node_map), PARSE_FUNC_DEF_FAILED);
    delete []nodeDefData;
}

TEST(TfGraphOptimizer, CheckSubGraphSupportFuse_SUCCESS)
{
    TfOptimizer TfOptimizer;
    ge::ComputeGraphPtr sub_graph = std::make_shared<ge::ComputeGraph>("subGraph");
    SubGraphInfo sub_graph_info;
    std::unordered_map<std::string, ge::NodePtr> tf_isolated_node_map;
    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");

    vector<int64_t> tensorShape = {2, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "AddTest_REF");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(Op1DescPtr);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "AddTest_REF");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    NodePtr node2 = make_shared<Node>(Op2DescPtr, graphPtr);
    node2->Init();
    graphPtr->AddNode(node2);
    node2->AddLinkFrom(node1);

    OpDescPtr Op3DescPtr = make_shared<OpDesc>("add3", "AddTest_REF");
    Op3DescPtr->AddOutputDesc("z", tensor1);
    NodePtr node3 = make_shared<Node>(Op3DescPtr, graphPtr);
    graphPtr->AddNode(node3);
    node3->AddLinkFrom(node2);

    std::vector<ge::NodePtr> nodeList;
    nodeList.push_back(node2);
    TfOptimizer.InsertNodesToSubGraph(sub_graph, nodeList, sub_graph_info);
    ASSERT_EQ(TfOptimizer.CheckSubGraphSupportFuse(nodeList, sub_graph_info, tf_isolated_node_map), true);
}

TEST(TfGraphOptimizer, InsertTfNodeDefToOp_Deterministic)
{
    OpDescPtr op_desc_ptr = make_shared<OpDesc>("Add", "Add");
    shared_ptr<ComputeGraph> graph_ptr = make_shared<ComputeGraph>("test_graph");
    NodePtr node = make_shared<Node>(op_desc_ptr, graph_ptr);
    domi::tensorflow::NodeDef node_def;
    Ir2tfBaseParser::Instance()->ParseNodeDef(*node.get(), &node_def);
    auto ret = InsertTfNodeDefToOp(op_desc_ptr, node_def, "Add", kTfNodeDef);
    ASSERT_EQ(ret, SUCCESS);
    ge::Buffer buffer_1;
    ge::AttrUtils::GetBytes(op_desc_ptr, kTfNodeDef, buffer_1);
    std::string str1(reinterpret_cast<const char *>(buffer_1.GetData()), buffer_1.GetSize());
    ret = InsertTfNodeDefToOp(op_desc_ptr, node_def, "Add", kTfNodeDef);
    ASSERT_EQ(ret, SUCCESS);
    ge::Buffer buffer_2;
    ge::AttrUtils::GetBytes(op_desc_ptr, kTfNodeDef, buffer_2);
    std::string str2(reinterpret_cast<const char *>(buffer_2.GetData()), buffer_2.GetSize());
    ASSERT_EQ(str1, str2);
}

TEST(TfGraphOptimizer, OptimizeFusedGraphForOoLevel3_SUCCESS)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O3"}};
    const std::unordered_map<std::string, OoInfo> &registered_opt_table =
      ge::OptionRegistry::GetInstance().GetRegisteredOptTable();
    ge::GetThreadLocalContext().GetOo().Initialize(ge_options, registered_opt_table);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr0->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}

TEST(TfGraphOptimizer, OptimizeFusedGraphForOoLevel1_SUCCESS)
{
    RegisterOpCreator("TemporaryVariable", {}, {"y"});
    RegisterOpCreator("Assign", {"ref", "value"}, {"ref"});
    RegisterOpCreator("Identity", {"x"}, {"y"});
    std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O1"}};
    const std::unordered_map<std::string, OoInfo> &registered_opt_table =
      ge::OptionRegistry::GetInstance().GetRegisteredOptTable();
    ge::GetThreadLocalContext().GetOo().Initialize(ge_options, registered_opt_table);
    map<string, GraphOptimizerPtr> graphOptimizers;
    GetGraphOptimizerObjs(graphOptimizers);
    ASSERT_NE(graphOptimizers["aicpu_tf_optimizer"], nullptr);

    shared_ptr<ComputeGraph> graphPtr = make_shared<ComputeGraph>("test_graph");
    vector<int64_t> tensorShape = {1, 2, 3, 4};
    GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetListInt(tensor1, "ref_port_index", {1, 1, 1});
    tensor1.SetOriginFormat(FORMAT_NCHW);
    tensor1.SetOriginShape(GeShape(tensorShape));

    OpDescPtr opDescPtr0 = make_shared<OpDesc>("placeholder", "PlaceHolder");
    opDescPtr0->AddOutputDesc("z", tensor1);
    auto node1 = graphPtr->AddNode(opDescPtr0);

    OpDescPtr Op1DescPtr = make_shared<OpDesc>("add1", "Add");
    Op1DescPtr->AddOutputDesc("z", tensor1);
    auto node2 = graphPtr->AddNode(Op1DescPtr);
    node2->AddLinkFrom(node1);

    OpDescPtr Op2DescPtr = make_shared<OpDesc>("add2", "Add");
    Op2DescPtr->AddOutputDesc("z", tensor1);
    auto node3 = graphPtr->AddNode(Op2DescPtr);
    node3->AddLinkFrom(node2);
    map<string, string> options;
    options[SOC_VERSION] = "Ascend910";
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->Initialize(options, nullptr), SUCCESS);
    ASSERT_EQ(graphOptimizers["aicpu_tf_optimizer"]->OptimizeFusedGraph(*(graphPtr.get())), SUCCESS);
}