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
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "../../../../graph_constructor/graph_builder_utils.h"

#define private public
#define protected public
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "fusion_manager/fusion_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph/utils/op_desc_utils_ex.h"

using namespace std;
using namespace testing;
using namespace fe;
using namespace te;

using fe::FEOpsKernelInfoStore;
using std::vector;
using std::map;
using namespace ge;
using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;
using TbeOpStoreAdapterPtr = std::shared_ptr<fe::TbeOpStoreAdapter>;

class FEOpsKernelInfoStoreSingleOpCompileTest : public testing::Test{
 protected:
  static void SetUpTestCase() {
    cout << "FEOpsKernelInfoStoreSingleOpCompileTest SetUP" << endl;
  }
  static void TearDownTestCase() {
    cout << "FEOpsKernelInfoStoreSingleOpCompileTest SetUP" << endl;
  }
  // Some expensive resource shared by all tests.
  virtual void SetUp(){
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(AI_CORE_NAME);

    op_desc_ptr = make_shared<ge::OpDesc>();
    input0_desc_ptr = make_shared<ge::GeTensorDesc>();
    input1_desc_ptr = make_shared<ge::GeTensorDesc>();
    input2_desc_ptr = make_shared<ge::GeTensorDesc>();
    output0_desc_ptr = make_shared<ge::GeTensorDesc>();


    op_desc_ptr->SetName("tbe_conv");
    ge::OpDescUtilsEx::SetType(op_desc_ptr, "conv");
    ge::DataType set_dtype = ge::DT_FLOAT16;
    ge::Format set_format = ge::FORMAT_ND;
    std::vector<int64_t> shape_vec{256,256,512};
    ge::GeShape shape_desc = GeShape(shape_vec);

    input0_desc_ptr->SetDataType(set_dtype);
    input0_desc_ptr->SetFormat(set_format);
    input0_desc_ptr->SetShape(shape_desc);
    op_desc_ptr->AddInputDesc("x", input0_desc_ptr->Clone());

    std::vector<int64_t> shape_vec1{256,256,512};
    ge::GeShape shape_desc1 = GeShape(shape_vec1);
    input1_desc_ptr->SetDataType(set_dtype);
    input1_desc_ptr->SetFormat(set_format);
    input1_desc_ptr->SetShape(shape_desc1);
    op_desc_ptr->AddInputDesc("y", input1_desc_ptr->Clone());

    std::vector<int64_t> shape_vec2{256,256,512};
    ge::GeShape shape_desc2 = GeShape(shape_vec2);
    input2_desc_ptr->SetDataType(set_dtype);
    input2_desc_ptr->SetFormat(set_format);
    input2_desc_ptr->SetShape(shape_desc2);
    op_desc_ptr->AddInputDesc("x1", input2_desc_ptr->Clone());

    output0_desc_ptr->SetDataType(set_dtype);
    output0_desc_ptr->SetFormat(set_format);
    op_desc_ptr->AddOutputDesc("z", output0_desc_ptr->Clone());

    format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);
    cout << "a test Set Up" << endl;
  }
  virtual void TearDown(){
    cout << "a test Tear Down" << endl;
    fe_ops_kernel_info_store_ptr->Finalize();

  }

 public:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr;
  shared_ptr<ge::GeTensorDesc> input0_desc_ptr;
  shared_ptr<ge::GeTensorDesc> input1_desc_ptr;
  shared_ptr<ge::GeTensorDesc> input2_desc_ptr;
  shared_ptr<ge::GeTensorDesc> output0_desc_ptr;
  shared_ptr<ge::OpDesc> op_desc_ptr;
  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
  TbeOpStoreAdapterPtr tbe_adapter_ptr_;

  static ge::ComputeGraphPtr BuildSingleOp() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 3, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(n, 0, abs, 2);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");

    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpOptional() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 4, 2, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(n, 0, abs, 2);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");

    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpDynamic() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 5, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(n, 0, abs, 6);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    builder.AddDataEdge(abs, 4, netoutput, 4);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    vector<uint32_t> dynamic_input_start_idx = {1,4};
    vector<uint32_t> dynamic_input_end_idx = {2,5};
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_end", dynamic_input_end_idx);

    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpDynamic1() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 5, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(n, 0, abs, 6);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    builder.AddDataEdge(abs, 4, netoutput, 4);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    vector<uint32_t> dynamic_input_start_idx = {2,4};
    vector<uint32_t> dynamic_input_end_idx = {2,4};
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_end", dynamic_input_end_idx);

    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpDynamicWrong() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 5, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(n, 0, abs, 6);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    builder.AddDataEdge(abs, 4, netoutput, 4);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    vector<uint32_t> dynamic_input_start_idx = {1,4,6};
    vector<uint32_t> dynamic_input_end_idx = {2,5};
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_end", dynamic_input_end_idx);

    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpDynamicWrong1() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 5, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(n, 0, abs, 6);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    builder.AddDataEdge(abs, 4, netoutput, 4);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    vector<uint32_t> dynamic_input_start_idx = {3,4};
    vector<uint32_t> dynamic_input_end_idx = {2,5};
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_input_index_end", dynamic_input_end_idx);

    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttr() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST, "strValue:list_int;stride:int;axis:float");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "strValue", dynamic_input_end_idx);
    ge::AttrUtils::SetInt(abs->GetOpDesc(), "stride", 1);
    ge::AttrUtils::SetFloat(abs->GetOpDesc(), "axis", 1.0);
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<float> abc = {1.0, 2.0};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST, "strValue:list_float;stride:str;axis:list_bool");
    ge::AttrUtils::SetListFloat(abs->GetOpDesc(), "strValue", abc);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), "stride", "abvc");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong1() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST, "strValue:abc;stride:bool;axis:list_str;abc:list_list_int");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "strValue", dynamic_input_end_idx);
    ge::AttrUtils::SetBool(abs->GetOpDesc(), "stride", true);
    ge::AttrUtils::SetListStr(abs->GetOpDesc(), "axis", abc);
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong2() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
            "1:str;2:int;3:float;4:bool;5:list_str;6:list_int;7:list_float;8:list_bool;9:list_list_int;10:tensor;11:list_tensor");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong3() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "2:int;3:float;4:bool;5:list_str;6:list_int;7:list_float;8:list_bool;9:list_list_int;10:tensor;11:list_tensor");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong4() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "3:float");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong5() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "4:bool");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong6() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "5:list_str");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong7() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "6:list_int");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong8() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "7:list_float");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong9() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "8:list_bool");
    return builder.GetGraph();
  }

  static ge::ComputeGraphPtr BuildSingleOpAttrWrong10() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "9:list_list_int");
    return builder.GetGraph();
  }
  static ge::ComputeGraphPtr BuildSingleOpAttrWrong11() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "10:tensor");
    return builder.GetGraph();
  }
  static ge::ComputeGraphPtr BuildSingleOpAttrWrong12() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNodeWithImplyType("data1", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data2 = builder.AddNodeWithImplyType("data2", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data3 = builder.AddNodeWithImplyType("data3", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data4 = builder.AddNodeWithImplyType("data4", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data5 = builder.AddNodeWithImplyType("data5", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto data6 = builder.AddNodeWithImplyType("data6", fe::DATA, 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto n = builder.AddNodeWithImplyType("n", "Const", 1, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto abs = builder.AddNodeWithImplyType("abs", "Abs", 7, 5, ge::FORMAT_NHWC, ge::DT_FLOAT);
    auto netoutput = builder.AddNodeWithImplyType("netoutput", fe::NETOUTPUT, 4, 1, ge::FORMAT_NHWC, ge::DT_FLOAT);

    builder.AddDataEdge(data1, 0, abs, 0);
    builder.AddDataEdge(data2, 0, abs, 1);
    builder.AddDataEdge(data3, 0, abs, 2);
    builder.AddDataEdge(data4, 0, abs, 3);
    builder.AddDataEdge(data5, 0, abs, 4);
    builder.AddDataEdge(data6, 0, abs, 5);
    builder.AddDataEdge(abs, 0, netoutput, 0);
    builder.AddDataEdge(abs, 1, netoutput, 1);
    builder.AddDataEdge(abs, 2, netoutput, 2);
    builder.AddDataEdge(abs, 3, netoutput, 3);
    vector<uint32_t> dynamic_input_start_idx = {1};
    vector<uint32_t> dynamic_input_end_idx = {2};
    vector<string> abc = {"abc", "abc1"};
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_OPPATH, "../../abs.py");
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_start", dynamic_input_start_idx);
    ge::AttrUtils::SetListInt(abs->GetOpDesc(), "_dynamic_output_index_end", dynamic_input_end_idx);
    ge::AttrUtils::SetStr(abs->GetOpDesc(), ge::ATTR_NAME_UNREGST_ATTRLIST,
                          "11:list_tensor");
    return builder.GetGraph();
  }
};

TEST_F(FEOpsKernelInfoStoreSingleOpCompileTest, initialize_fail){
  map<string, string> options;
  fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  FEOpsStoreInfo tbe_custom { };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_custom);
  Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  fe_ops_kernel_info_store_ptr->Initialize(options);
  Status ret = fe_ops_kernel_info_store_ptr->Initialize(options);
  auto graph = std::make_shared<ge::ComputeGraph>("test");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "conv");
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);

  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_NCHW);
  in_desc1.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc1);

  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_HWCN);
  out_desc1.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("y", out_desc1);
  ge::AttrUtils::SetStr(relu_op, "unregst_oppath", "./impl/abc");
  NodePtr relu_node = graph->AddNode(relu_op);
  vector<ge::NodePtr> node_vec;
  node_vec.push_back(relu_node);
  fe_ops_kernel_info_store_ptr->CompileOp(node_vec);
  EXPECT_EQ(fe::SUCCESS, ret);
}
