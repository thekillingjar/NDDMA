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
#include "fe_llt_utils.h"
#include "common/fe_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#define private public
#define protected public
#include "format_selector/manager/format_dtype_setter.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "common/config_parser/op_cust_dtypes_config_parser.h"
#include "common/configuration.h"
#include "graph/ge_local_context.h"
using namespace std;
using namespace ge;
using namespace fe;

using FEOpsKernelInfoStorePtr = std::shared_ptr<FEOpsKernelInfoStore>;
using OpImplyTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;
using OpFormatDtypeJudgePtr = std::shared_ptr<OpFormatDtypeJudge>;
using ReflectionBuilderPtr = std::shared_ptr<ge::RefRelations>;
using FormatDtypeSetterPtr = std::shared_ptr<FormatDtypeSetter>;

std::string GetAscendPath() {
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_path = "/mnt/d/Ascend";
  if (ascend_custom_path_ptr != nullptr) {
      ascend_path = fe::GetRealPath(string(ascend_custom_path_ptr));
  } else {
      const char *ascend_home_path_ptr = std::getenv("ASCEND_HOME");
      if (ascend_home_path_ptr != nullptr) {
      ascend_path = fe::GetRealPath(string(ascend_home_path_ptr));
      } else {
      ascend_path = "/mnt/d/Ascend";
      }
  }
  return ascend_path;
}

string GetNetworkPath(const string &network_name) {
  auto custom_path = GetAscendPath();
  custom_path += "/net/";
  return custom_path + network_name;
}

class st_op_judge_by_custom_dtypes : public testing::Test
{
protected:
  void SetUp()
  {
    ReflectionBuilderPtr reflection_builder_ptr = std::make_shared<ge::RefRelations>();
    FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr =
            std::make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

    FEOpsStoreInfo tbe_custom {
            6,
            "tbe-custom",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
            ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);

    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr->Initialize(options);
    ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
    if (Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_ == nullptr) {
      Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_ = make_shared<OpCustDtypesConfigParser>();
    }
    op_cust_dtypes_parser_ptr_ =
            std::dynamic_pointer_cast<OpCustDtypesConfigParser>(Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

    op_impl_type_judge_ptr_ =
            std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr);
    format_dtype_setter_ptr_ =
            std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
    op_format_dtype_judge_ptr_ =
            std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr);
    op_format_dtype_judge_ptr_->Initialize();
    Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  }

  void TearDown() {}

protected:
  OpImplyTypeJudgePtr op_impl_type_judge_ptr_;
  FormatDtypeSetterPtr format_dtype_setter_ptr_;
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr_;
  OpCustDtypesConfigParserPtr op_cust_dtypes_parser_ptr_;

  ComputeGraphPtr CreateMatmulSingleOpGraph(const string &op_name, const string &op_type, OpDescPtr &op_desc) {
    OpDescPtr mat_mul_op = std::make_shared<OpDesc>(op_name, op_type);
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
    OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
    OpDescPtr const_op3 = std::make_shared<OpDesc>("const3", "Const");
    op_desc = mat_mul_op;

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    mat_mul_op->AddInputDesc("x1", tensor_desc);
    mat_mul_op->AddInputDesc("x2", tensor_desc);
    mat_mul_op->AddInputDesc("bias", tensor_desc);
    GeTensorDesc tensor_desc1(shape);
    tensor_desc1.SetDataType(DT_INT32);
    tensor_desc1.SetOriginFormat(FORMAT_ND);
    tensor_desc1.SetOriginDataType(DT_INT32);
    tensor_desc1.SetOriginShape(shape);
    mat_mul_op->AddInputDesc("offset", tensor_desc1);
    mat_mul_op->AddOutputDesc("x2", tensor_desc);

    data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
    const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
    const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));
    const_op3->AddOutputDesc(mat_mul_op->GetInputDesc(3));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr const_node1 = graph->AddNode(const_op1);
    ge::NodePtr const_node2 = graph->AddNode(const_op2);
    ge::NodePtr const_node3 = graph->AddNode(const_op3);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(const_node3->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(3));
    return graph;
  }

  ComputeGraphPtr CreateCustomSingleOpGraph(const string &op_name, const string &op_type, OpDescPtr &op_desc) {
    OpDescPtr mat_mul_op = std::make_shared<OpDesc>(op_name, op_type);
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
    OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
    op_desc = mat_mul_op;

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    mat_mul_op->AddInputDesc("x", tensor_desc);
    mat_mul_op->AddInputDesc("y", tensor_desc);
    mat_mul_op->AddOutputDesc("z", tensor_desc);

    data_op->AddOutputDesc(mat_mul_op->GetInputDesc(0));
    const_op1->AddOutputDesc(mat_mul_op->GetInputDesc(1));
    const_op2->AddOutputDesc(mat_mul_op->GetInputDesc(2));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    ge::NodePtr mat_mul_node = graph->AddNode(mat_mul_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr const_node1 = graph->AddNode(const_op1);
    ge::NodePtr const_node2 = graph->AddNode(const_op2);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), mat_mul_node->GetInDataAnchor(2));
    return graph;
  }

  ComputeGraphPtr  CreateFirstLayerOpGraph1(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op,
                                           OpDescPtr &dw_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Variable");
    dw_op = std::make_shared<OpDesc>("conv2ddw", "Conv2DBackpropFilterD");
    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);
    data_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));
    dw_op->AddInputDesc(conv2d_op->GetInputDesc(0));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    ge::NodePtr conv2ddw_node = graph->AddNode(dw_op);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2ddw_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    return graph;
  }

  ComputeGraphPtr CreateFirstLayerOpGraph2(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Variable");
    OpDescPtr conv2ddw_op = std::make_shared<OpDesc>("conv2ddw", "Conv2DBackpropFilterD");
    OpDescPtr test_op1 = std::make_shared<OpDesc>("test1", "Test");
    OpDescPtr test_op2 = std::make_shared<OpDesc>("test2", "Test");
    OpDescPtr test_op3 = std::make_shared<OpDesc>("test3", "Test");
    OpDescPtr test_op4 = std::make_shared<OpDesc>("test4", "Test");
    OpDescPtr test_op5 = std::make_shared<OpDesc>("test5", "Test");
    OpDescPtr test_op6 = std::make_shared<OpDesc>("test6", "Test");
    OpDescPtr test_op7 = std::make_shared<OpDesc>("test7", "Test");
    OpDescPtr test_op8 = std::make_shared<OpDesc>("test8", "Test");
    OpDescPtr test_op9 = std::make_shared<OpDesc>("test9", "Test");
    OpDescPtr test_op10 = std::make_shared<OpDesc>("test10", "Test");

    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);

    data_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));

    conv2ddw_op->AddInputDesc(conv2d_op->GetInputDesc(0));

    test_op1->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op2->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op3->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op4->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op5->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op6->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op7->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op8->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op9->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op10->AddInputDesc(conv2d_op->GetInputDesc(0));

    test_op1->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op2->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op3->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op4->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op5->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op6->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op7->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op8->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op9->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op10->AddOutputDesc(conv2d_op->GetInputDesc(0));


    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    ge::NodePtr conv2ddw_node = graph->AddNode(conv2ddw_op);
    
    ge::NodePtr test_node1 = graph->AddNode(test_op1);
    ge::NodePtr test_node2 = graph->AddNode(test_op2);
    ge::NodePtr test_node3 = graph->AddNode(test_op3);
    ge::NodePtr test_node4 = graph->AddNode(test_op4);
    ge::NodePtr test_node5 = graph->AddNode(test_op5);
    ge::NodePtr test_node6 = graph->AddNode(test_op6);
    ge::NodePtr test_node7 = graph->AddNode(test_op7);
    ge::NodePtr test_node8 = graph->AddNode(test_op8);
    ge::NodePtr test_node9 = graph->AddNode(test_op9);
    ge::NodePtr test_node10 = graph->AddNode(test_op10);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), test_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node1->GetOutDataAnchor(0), test_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node2->GetOutDataAnchor(0), test_node3->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node3->GetOutDataAnchor(0), test_node4->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node4->GetOutDataAnchor(0), test_node5->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node5->GetOutDataAnchor(0), test_node6->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node6->GetOutDataAnchor(0), test_node7->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node7->GetOutDataAnchor(0), test_node8->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node8->GetOutDataAnchor(0), test_node9->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node9->GetOutDataAnchor(0), test_node10->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node10->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2ddw_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    return graph;
  }

  ComputeGraphPtr CreateFirstLayerOpGraph3(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Variable");
    OpDescPtr conv2ddw_op = std::make_shared<OpDesc>("conv2ddw", "Conv2DBackpropFilterD");
    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);
    data_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));
    conv2ddw_op->AddInputDesc(conv2d_op->GetInputDesc(0));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    ge::NodePtr conv2ddw_node = graph->AddNode(conv2ddw_op);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2ddw_node->GetInDataAnchor(0));
    return graph;
  }
  
  ComputeGraphPtr CreateFirstLayerOpGraph4(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Test");
    OpDescPtr conv2ddw_op = std::make_shared<OpDesc>("conv2ddw", "Conv2DBackpropFilterD");
    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);
    data_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));
    conv2ddw_op->AddInputDesc(conv2d_op->GetInputDesc(0));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    ge::NodePtr conv2ddw_node = graph->AddNode(conv2ddw_op);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2ddw_node->GetInDataAnchor(0));
    return graph;
  }

  ComputeGraphPtr CreateFirstLayerOpGraph5(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Variable");
    OpDescPtr conv2ddw_op = std::make_shared<OpDesc>("conv2ddw", "Conv2DBackpropFilterD");
    OpDescPtr test_op1 = std::make_shared<OpDesc>("test1", "Test");
    OpDescPtr test_op2 = std::make_shared<OpDesc>("test2", "Test");
    OpDescPtr test_op3 = std::make_shared<OpDesc>("test3", "Test");
    OpDescPtr test_op4 = std::make_shared<OpDesc>("test4", "Test");
    OpDescPtr test_op5 = std::make_shared<OpDesc>("test5", "Test");
    OpDescPtr test_op6 = std::make_shared<OpDesc>("test6", "Test");
    OpDescPtr test_op7 = std::make_shared<OpDesc>("test7", "Test");
    OpDescPtr test_op8 = std::make_shared<OpDesc>("test8", "Test");
    OpDescPtr test_op9 = std::make_shared<OpDesc>("test9", "Test");
    OpDescPtr test_op10 = std::make_shared<OpDesc>("test10", "Test");
    OpDescPtr test_op11 = std::make_shared<OpDesc>("test11", "Test");
    OpDescPtr test_op12 = std::make_shared<OpDesc>("test12", "Test");
    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);

    data_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));

    conv2ddw_op->AddInputDesc(conv2d_op->GetInputDesc(0));

    test_op1->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op2->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op3->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op4->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op5->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op6->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op7->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op8->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op9->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op10->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op11->AddInputDesc(conv2d_op->GetInputDesc(0));
    test_op12->AddInputDesc(conv2d_op->GetInputDesc(0));

    test_op1->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op2->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op3->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op4->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op5->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op6->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op7->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op8->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op9->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op10->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op11->AddOutputDesc(conv2d_op->GetInputDesc(0));
    test_op12->AddOutputDesc(conv2d_op->GetInputDesc(0));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    ge::NodePtr conv2ddw_node = graph->AddNode(conv2ddw_op);
    
    ge::NodePtr test_node1 = graph->AddNode(test_op1);
    ge::NodePtr test_node2 = graph->AddNode(test_op2);
    ge::NodePtr test_node3 = graph->AddNode(test_op3);
    ge::NodePtr test_node4 = graph->AddNode(test_op4);
    ge::NodePtr test_node5 = graph->AddNode(test_op5);
    ge::NodePtr test_node6 = graph->AddNode(test_op6);
    ge::NodePtr test_node7 = graph->AddNode(test_op7);
    ge::NodePtr test_node8 = graph->AddNode(test_op8);
    ge::NodePtr test_node9 = graph->AddNode(test_op9);
    ge::NodePtr test_node10 = graph->AddNode(test_op10);
    ge::NodePtr test_node11 = graph->AddNode(test_op10);
    ge::NodePtr test_node12 = graph->AddNode(test_op10);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), test_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node1->GetOutDataAnchor(0), test_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node2->GetOutDataAnchor(0), test_node3->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node3->GetOutDataAnchor(0), test_node4->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node4->GetOutDataAnchor(0), test_node5->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node5->GetOutDataAnchor(0), test_node6->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), test_node7->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node7->GetOutDataAnchor(0), test_node8->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node8->GetOutDataAnchor(0), test_node9->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node9->GetOutDataAnchor(0), test_node10->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node10->GetOutDataAnchor(0), test_node11->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node11->GetOutDataAnchor(0), test_node12->GetInDataAnchor(0));
    GraphUtils::AddEdge(test_node12->GetOutDataAnchor(0), conv2ddw_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    return graph;
  }

  ComputeGraphPtr CreateFirstLayerOpGraph6(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr data_op = std::make_shared<OpDesc>("data1", "Data");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Variable");
    OpDescPtr conv2ddw_op = std::make_shared<OpDesc>("conv2ddw", "Test");
    //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);
    data_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));
    conv2ddw_op->AddInputDesc(conv2d_op->GetInputDesc(0));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr data_node = graph->AddNode(data_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    ge::NodePtr conv2ddw_node = graph->AddNode(conv2ddw_op);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv2ddw_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    return graph;
  }

  ComputeGraphPtr CreateFirstLayerOpGraph7(ge::NodePtr &conv2d_node, OpDescPtr &conv2d_op) {
    conv2d_op = std::make_shared<OpDesc>("conv2d1", "Conv2D");
    OpDescPtr bn_op = std::make_shared<OpDesc>("bninter", "BNInferenceD");
    OpDescPtr aipp_op = std::make_shared<OpDesc>("aipp1", "Aipp");
    OpDescPtr weight_op = std::make_shared<OpDesc>("weight1", "Const");
    
     //add descriptor
    vector<int64_t> dim({4, 33, 12, 16});
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetOriginFormat(FORMAT_ND);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    tensor_desc.SetOriginShape(shape);

    conv2d_op->AddInputDesc("x", tensor_desc);
    conv2d_op->AddInputDesc("y", tensor_desc);
    conv2d_op->AddOutputDesc("z", tensor_desc);
    bn_op->AddOutputDesc(conv2d_op->GetInputDesc(0));
    bn_op->AddInputDesc(conv2d_op->GetInputDesc(0));
    aipp_op->AddOutputDesc(conv2d_op->GetInputDesc(1));
    weight_op->AddOutputDesc(conv2d_op->GetInputDesc(1));

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
    conv2d_node =  graph->AddNode(conv2d_op);
    ge::NodePtr bn_node = graph->AddNode(bn_op);
    ge::NodePtr aipp_node = graph->AddNode(aipp_op);
    ge::NodePtr weight_node = graph->AddNode(weight_op);
    GraphUtils::AddEdge(aipp_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(weight_node->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    return graph;
  }

  ge::ComputeGraphPtr CreateInceptionV3NetGraph() {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("inceptionv3");
    string network_path = GetNetworkPath("inceptionv3_aipp_int8_16batch.txt");
    (void)ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
    return graph;
  }
};

TEST_F(st_op_judge_by_custom_dtypes, multi_thread_judge)
{
  auto graph = CreateInceptionV3NetGraph();
  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph2 = CreateFirstLayerOpGraph5(op_node, op_desc);
  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph2);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(st_op_judge_by_custom_dtypes, mat_mul_v2_check_support_without_custom)
{
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  Configuration::Instance(fe::AI_CORE_NAME).fp16_op_type_list_.clear();
  string op_name = "mat_mul_name";
  string op_type = "MatMulV2";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateMatmulSingleOpGraph(op_name, op_type, op_desc);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), true);

  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(st_op_judge_by_custom_dtypes, mat_mul_v2_check_support_not_pass)
{
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  string op_name = "mat_mul_name";
  string op_type = "MatMulV2";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateMatmulSingleOpGraph(op_name, op_type, op_desc);

  map<string, string> options;
  options.emplace("ge.customizeDtypes",
                  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes1.txt");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_->InitializeFromOptions(options);
  Configuration::Instance(fe::AI_CORE_NAME).config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes1.txt",
               Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), false);

  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT64);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT64);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(st_op_judge_by_custom_dtypes, mat_mul_v2_check_support_pass)
{
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  string op_name = "mat_mul_name";
  string op_type = "MatMulV2";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateMatmulSingleOpGraph(op_name, op_type, op_desc);

  map<string, string> options;
  options.emplace("ge.customizeDtypes",
  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes2.txt");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_->InitializeFromOptions(options);
  Configuration::Instance(fe::AI_CORE_NAME).config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes2.txt",
               Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), true);

  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
}

TEST_F(st_op_judge_by_custom_dtypes, mat_mul_v3_check_support_pass)
{
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  string op_name = "mat_mul_name";
  string op_type = "MatMulV3";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateMatmulSingleOpGraph(op_name, op_type, op_desc);

  map<string, string> options;
  options.emplace("ge.customizeDtypes",
  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes3.txt");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_->InitializeFromOptions(options);
  Configuration::Instance(fe::AI_CORE_NAME).config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes3.txt",
               Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), true);

  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(st_op_judge_by_custom_dtypes, mat_mul_v3_check_support_without_custom)
{
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  string op_name = "mat_mul_name";
  string op_type = "MatMulV3";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateMatmulSingleOpGraph(op_name, op_type, op_desc);

  map<string, string> options;
  options.emplace("ge.customizeDtypes",
  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes4.txt");
  Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_->InitializeFromOptions(options);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), true);

  ret = format_dtype_setter_ptr_->MultiThreadSetSupportFormatDtype(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(2).GetDataType(), DT_FLOAT);
  EXPECT_EQ(op_desc->GetInputDesc(3).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT);
}

TEST_F(st_op_judge_by_custom_dtypes, custom_op_check_support_pass1)
{
  Configuration &config = Configuration::Instance(fe::AI_CORE_NAME);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  string op_name = "custom_op_name";
  string op_type = "CustomOpA";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateCustomSingleOpGraph(op_name, op_type, op_desc);

  map<string, string> options;
  options.emplace("ge.customizeDtypes",
  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes5.txt");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_->InitializeFromOptions(options);
  Configuration::Instance(fe::AI_CORE_NAME).config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes5.txt",
               Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), true);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_INT32);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_INT32);
}

TEST_F(st_op_judge_by_custom_dtypes, custom_op_check_support_pass2)
{
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = MUST_KEEP_ORIGIN_DTYPE;
  string op_name = "custom_op_name";
  string op_type = "CustomOpA";
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr graph = CreateCustomSingleOpGraph(op_name, op_type, op_desc);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), false);

  map<string, string> options;
  options.emplace("ge.customizeDtypes",
  GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes6.txt");
  ge::GetThreadLocalContext().SetGraphOption(options);
  Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_->InitializeFromOptions(options);
  Configuration::Instance(fe::AI_CORE_NAME).config_parser_map_vec_[static_cast<size_t>(CONFIG_PARSER_PARAM::CustDtypes)]
      .emplace(GetCodeDir() + "/tests/engines/nn_engine/st/testcase/graph_optimizer/op_judge/custom_dtypes/custom_dtypes6.txt",
               Configuration::Instance(fe::AI_CORE_NAME).cust_dtypes_parser_);

  ret = op_impl_type_judge_ptr_->MultiThreadJudge(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->HasAttr(fe::FE_IMPLY_TYPE), true);

  ret = op_format_dtype_judge_ptr_->Judge(*graph);
  ret = op_format_dtype_judge_ptr_->SetFormat(*graph);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetInputDesc(1).GetDataType(), DT_FLOAT16);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv01)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  OpDescPtr dw_op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph1(op_node, op_desc, dw_op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), true);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
  EXPECT_EQ(dw_op_desc->HasAttr(IS_FIRST_LAYER_CONV), true);
  EXPECT_EQ(dw_op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv02)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph2(op_node, op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), false);
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv03)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph3(op_node, op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), false);
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv04)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph4(op_node, op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), false);
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv05)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph5(op_node, op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), false);
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv06)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph6(op_node, op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), false);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), false);
}

TEST_F(st_op_judge_by_custom_dtypes, JudgeFirstLayerConv07)
{
  Configuration::Instance(AI_CORE_NAME).SetEnableSmallChannel(true);
  op_cust_dtypes_parser_ptr_->op_name_cust_dtypes_.clear();
  op_cust_dtypes_parser_ptr_->op_type_cust_dtypes_.clear();
  OpDescPtr op_desc = nullptr;
  ge::NodePtr op_node = nullptr;
  ComputeGraphPtr graph = CreateFirstLayerOpGraph7(op_node, op_desc);
  if (op_desc == nullptr ||  op_node == nullptr) {
    EXPECT_EQ(false, true);
  }
  format_dtype_setter_ptr_->JudgeFirstLayerConv(op_node, op_desc);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV_FOR_OP), true);
  EXPECT_EQ(op_desc->HasAttr(IS_FIRST_LAYER_CONV), false);
}

