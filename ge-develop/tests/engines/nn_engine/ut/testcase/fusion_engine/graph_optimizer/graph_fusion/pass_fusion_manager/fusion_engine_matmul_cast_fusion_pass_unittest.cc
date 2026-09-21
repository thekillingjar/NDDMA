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

#include "fe_llt_utils.h"
#define protected public
#define private public
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "common/configuration.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace te;

using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;


namespace fe {
  static const char *TF_MATMUL = "MatMul";
  static const char *SUB = "Sub";
  bool CheckTbeCastSupportedStub(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
    result.isSupported = te::FULLY_SUPPORTED;
    return true;
  }
static const uint8_t MATMUL_INPUT_NUM = 2;
static const uint8_t MATMUL_OUTPUT_NUM = 1;
static const uint8_t CAST_INPUT_NUM = 1;
static const uint8_t CAST_OUTPUT_NUM = 1;
static const uint8_t SUB_INPUT_NUM = 2;
static const uint8_t SUB_OUTPUT_NUM = 1;
static const int32_t TENSORFLOW_DATATYPE_FLOAT32 = 0;
static const int32_t TENSORFLOW_DATATYPE_FLOAT16 = 19;
static const string MATMUL_DATATYPE_ATTR_KEY = "T";
static const string CAST_DATATYPE_DES_ATTR_KEY = "DstT";
class UTEST_fusion_engine_matmul_cast_fusion_unittest :public testing::Test {
protected:
  FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr;
    void SetUp()
    {
      TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
      tbe_op_store_adapter_ptr->CheckTbeSupported = CheckTbeCastSupportedStub;

      fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
      FEOpsStoreInfo tbe_builtin {
              0,
              "tbe-builtin",
              EN_IMPL_HW_TBE,
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
              "",
              false,
              false,
              false};
      vector<FEOpsStoreInfo> store_info;
      store_info.emplace_back(tbe_builtin);
      Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
      std::map<std::string, std::string> options;
      fe_ops_kernel_info_store_ptr->Initialize(options);
    }
    void TearDown()
    {

    }

protected:
    ge::NodePtr AddNode(ge::ComputeGraphPtr graph,
                        const string &name,
                        const string &type,
                        unsigned int in_anchor_num,
                        unsigned int out_anchor_num,
                        ge::DataType input_type,
                        ge::DataType output_type)
    {
        ge::GeTensorDesc input_tensor_desc(ge::GeShape({1, 2, 3}), ge::FORMAT_NHWC, input_type);
        ge::GeTensorDesc output_tensor_desc(ge::GeShape({1, 2, 3}), ge::FORMAT_NHWC, output_type);
        ge::OpDescPtr op_desc = make_shared<ge::OpDesc>(name, type);
        for (unsigned int i = 0; i < in_anchor_num; ++i) {
            op_desc->AddInputDesc(input_tensor_desc);
        }
        for (unsigned int i = 0; i < out_anchor_num; ++i) {
            op_desc->AddOutputDesc(output_tensor_desc);
        }
        ge::NodePtr node = graph->AddNode(op_desc);
        return node;
    }

    ge::ComputeGraphPtr CreateMatmulCastGraph()
    {
        // create compute graph
        ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_matmul_cast_fusion_graph");
        // create matmul node
        ge::NodePtr matmul_node = AddNode(graph, "matmul", TF_MATMUL, MATMUL_INPUT_NUM, MATMUL_OUTPUT_NUM,
                                         ge::DataType::DT_FLOAT16,
                                         ge::DataType::DT_FLOAT16);
        // create cast node
        ge::NodePtr cast_node = AddNode(graph, "cast", fe::CAST, CAST_INPUT_NUM, CAST_OUTPUT_NUM,
                                       ge::DataType::DT_FLOAT16,
                                       ge::DataType::DT_FLOAT);
        // create sub node
        ge::NodePtr sub_node = AddNode(graph, "sub", SUB, SUB_INPUT_NUM, SUB_OUTPUT_NUM,
                                      ge::DataType::DT_FLOAT,
                                      ge::DataType::DT_FLOAT);
        // link matmul node and cast node
        ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
        // linkd cast node and sub node
        ge::GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(0));
        return graph;
    }

      ge::ComputeGraphPtr CreateMatmulCastGraph2()
      {
          // create compute graph
          ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_matmul_cast_fusion_graph");
          // create matmul node
          ge::NodePtr matmul_node = AddNode(graph, "matmul", TF_MATMUL, MATMUL_INPUT_NUM, MATMUL_OUTPUT_NUM,
                                           ge::DataType::DT_FLOAT16,
                                           ge::DataType::DT_FLOAT16);
          // create cast node
          ge::NodePtr cast_node = AddNode(graph, "cast", fe::CAST, CAST_INPUT_NUM, CAST_OUTPUT_NUM,
                                         ge::DataType::DT_FLOAT16,
                                         ge::DataType::DT_FLOAT);
          // create sub node
          ge::NodePtr sub_node = AddNode(graph, "sub", SUB, SUB_INPUT_NUM, SUB_OUTPUT_NUM,
                                        ge::DataType::DT_FLOAT,
                                        ge::DataType::DT_FLOAT);
          // link matmul node and cast node
          ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0));
          // linkd cast node and sub node
          ge::GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), sub_node->GetInDataAnchor(0));
          return graph;
      }
};
}
