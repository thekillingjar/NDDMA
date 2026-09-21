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

#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "common/configuration.h"
#include "adapter/common/op_store_adapter_manager.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace fe;
using namespace te;
static const string ATTR_DATA_TYPE = "T";

namespace fe {
  static const char *TF_MATMUL = "MatMul";
  static const char *BIASADD = "BiasAdd";
  static const char *ADD = "Add";

  using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;

  bool CheckTbeSupportedStub(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
    result.isSupported = te::FULLY_SUPPORTED;
    return true;
  }

  class fusion_matmul_biasadd_pass_st : public testing::Test
  {
  public:
    FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr;
  protected:
    void SetUp()
    {
        TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
        tbe_op_store_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub;

        fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>();
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
        OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

        std::map<std::string, std::string> options;
        fe_ops_kernel_info_store_ptr->Initialize(options);
    }

    void TearDown()
    {

    }

    static NodePtr CreateConstNode(string name, GeTensorDescPtr out_desc_ptr, ComputeGraphPtr graph)
    {
        OpDescPtr constant = std::make_shared<OpDesc>(name, CONSTANT);
        //set OpDesc
        AttrUtils::SetStr(out_desc_ptr, "name", name + "Out0");
        constant->AddOutputDesc(out_desc_ptr->Clone());
        // set attr
        AttrUtils::SetInt(constant, ATTR_DATA_TYPE, DT_FLOAT);
        NodePtr node_const = graph->AddNode(constant);

        return node_const;
    }

    static NodePtr CreateOtherNode(string name, GeTensorDescPtr tensor_desc_ptr, ComputeGraphPtr graph)
    {
        OpDescPtr other_desc_ptr = std::make_shared<OpDesc>(name, "otherNode");
        //set OpDesc
        auto local_tensor_desc = tensor_desc_ptr->Clone();
        // add two input desc
        for (int i = 0; i < 2; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name", name + "In" + std::to_string(i));
            other_desc_ptr->AddInputDesc(local_tensor_desc);
        }
        // add two output desc
        for (int i = 0; i < 2; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name", name + "Out" + std::to_string(i));
            other_desc_ptr->AddOutputDesc(local_tensor_desc);
        }
        // add node from other_desc_ptr to graph
        // set attr
        AttrUtils::SetInt(other_desc_ptr, ATTR_DATA_TYPE, DT_FLOAT);
        NodePtr node_other = graph->AddNode(other_desc_ptr);

        return node_other;
    }

    static NodePtr CreateBiasAddNode(string name, GeTensorDescPtr tensor_desc_ptr, ComputeGraphPtr graph)
    {
        OpDescPtr desc_ptr = std::make_shared<OpDesc>(name, BIASADD);
        //set OpDesc
        auto local_tensor_desc = tensor_desc_ptr->Clone();
        // add two input desc
        for (int i = 0; i < 2; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name", name + "In" + std::to_string(i));
            desc_ptr->AddInputDesc(local_tensor_desc);
        }
        // add 1 output desc
        for (int i = 0; i < 1; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name", name + "Out" + std::to_string(i));
            desc_ptr->AddOutputDesc(local_tensor_desc);
        }
        // add node from desc_ptr to graph
        // set attr
        AttrUtils::SetInt(desc_ptr, ATTR_DATA_TYPE, DT_FLOAT);
        NodePtr node_bias_add = graph->AddNode(desc_ptr);

        return node_bias_add;
    }
    static NodePtr CreateAddNode(string name, GeTensorDescPtr tensor_desc_ptr,
                                 ComputeGraphPtr graph) {
        OpDescPtr desc_ptr = std::make_shared<OpDesc>(name, ADD);
        //set OpDesc
        auto local_tensor_desc = tensor_desc_ptr->Clone();
        // add two input desc
        for (int i = 0; i < 2; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name",
                              name + "In" + std::to_string(i));
            desc_ptr->AddInputDesc(local_tensor_desc);
        }
        // add 1 output desc
        for (int i = 0; i < 1; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name",
                              name + "Out" + std::to_string(i));
            desc_ptr->AddOutputDesc(local_tensor_desc);
        }
        // add node from desc_ptr to graph
        // set attr
        AttrUtils::SetInt(desc_ptr, ATTR_DATA_TYPE, DT_FLOAT);
        NodePtr node_bias_add = graph->AddNode(desc_ptr);
        std::vector<std::string> original_names;
        original_names.push_back("matmul");
        original_names.push_back("cast");
        ge::AttrUtils::SetListStr(node_bias_add->GetOpDesc(),
                                  ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                  original_names);
        ge::AttrUtils::SetStr(node_bias_add->GetOpDesc()->MutableOutputDesc(0),
                              ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, "cast");
        ge::AttrUtils::SetInt(node_bias_add->GetOpDesc()->MutableOutputDesc(0),
                              ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,
                              0);
        GraphPassUtil::SetDataDumpOriginDataType(ge::DT_FLOAT,
                                             node_bias_add->GetOpDesc()->MutableOutputDesc(
                                                     0));
        GraphPassUtil::SetDataDumpOriginFormat(ge::FORMAT_NHWC,
                                           node_bias_add->GetOpDesc()->MutableOutputDesc(
                                                   0));
        return node_bias_add;
    }
    static NodePtr CreateMatMulNode(string name, GeTensorDescPtr tensor_desc_ptr, ComputeGraphPtr graph)
    {
        OpDescPtr desc_ptr = std::make_shared<OpDesc>(name, TF_MATMUL);
        //set OpDesc
        auto local_tensor_desc = tensor_desc_ptr->Clone();
        // add two input desc
        for (int i = 0; i < 2; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name", name + "In" + std::to_string(i));
            desc_ptr->AddInputDesc(local_tensor_desc);
        }
        // add 1 output desc
        for (int i = 0; i < 1; ++i) {
            AttrUtils::SetStr(local_tensor_desc, "name", name + "Out" + std::to_string(i));
            desc_ptr->AddOutputDesc(local_tensor_desc);
        }
        // add node from desc_ptr to graph
        // set attr
        AttrUtils::SetInt(desc_ptr, ATTR_DATA_TYPE, DT_FLOAT);
        NodePtr node = graph->AddNode(desc_ptr);

        return node;
    }

    static ComputeGraphPtr CreateTestGraph(bool fusion_flag)
    {
        /*
            Const(0)    Const(1)
               \           /
                \         /
                 \       /
                  \     /
              MatMul(fusion)   Const(Bias)
                      \           /  \
                       \         /    \
                        \       /      \
                         \     /     Other(0)
                    BiasAdd(fusion)       Const(2)
                       /     \             /
                      /       \           /
                     /         \         /
                    /           \       /
                  Other(1)      MatMul(0)      Const(3)
                                  /  \            /
                                 /    \          /
                                /      \        /
                            Other(2)      BiasAdd0
                                            |
                                            |
                                          Other(3)
        */
        ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
        // new a output GeTensorDesc
        GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
        GeShape shape;
        shape.SetDim(0, 1);
        shape.SetDim(1, 128);
        shape.SetDim(2, 229);
        shape.SetDim(3, 229);
        general_ge_tensor_desc->SetShape(shape);
        general_ge_tensor_desc->SetFormat(FORMAT_NCHW);
        general_ge_tensor_desc->SetDataType(DT_FLOAT16);

        NodePtr node_const0 = CreateConstNode("test/const0", general_ge_tensor_desc, graph);
        NodePtr node_const1 = CreateConstNode("test/const1", general_ge_tensor_desc, graph);
        NodePtr node_const2 = CreateConstNode("test/const2", general_ge_tensor_desc, graph);
        NodePtr node_const3 = CreateConstNode("test/const3", general_ge_tensor_desc, graph);
        NodePtr node_bias = CreateConstNode("test/Bias", general_ge_tensor_desc, graph);

        NodePtr node_other0 = CreateOtherNode("test/other0", general_ge_tensor_desc, graph);
        NodePtr node_other1 = CreateOtherNode("test/other1", general_ge_tensor_desc, graph);
        NodePtr node_other2 = CreateOtherNode("test/other2", general_ge_tensor_desc, graph);
        NodePtr node_other3 = CreateOtherNode("test/other3", general_ge_tensor_desc, graph);

        NodePtr node_mat_mul_fusion = CreateMatMulNode("test/MatMul_fusion", general_ge_tensor_desc, graph);
        NodePtr node_mat_mul0 = CreateMatMulNode("test/MatMul0", general_ge_tensor_desc, graph);

        NodePtr node_bias_add_fusion = CreateBiasAddNode("test/BiasAdd_fusion", general_ge_tensor_desc, graph);
        NodePtr node_bias_add0 = CreateBiasAddNode("test/BiasAdd0", general_ge_tensor_desc, graph);

        /* add link of anchors */
        std::vector<OutDataAnchorPtr> srcs;
        std::vector<InDataAnchorPtr> dsts;
        // 0: add link from Const(0) to MatMul(fusion)[0]
        srcs.push_back(node_const0->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul_fusion->GetInDataAnchor(0));
        // 1: add link from Const(1) to MatMul(fusion)[1]
        srcs.push_back(node_const1->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul_fusion->GetInDataAnchor(1));
        // 2: add link from MatMul(fusion) to BiasAdd(fusion)[0]
        srcs.push_back(node_mat_mul_fusion->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add_fusion->GetInDataAnchor(0));
        // 3: add link from Const(Bias) to BiasAdd(fusion)[1]
        srcs.push_back(node_bias->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add_fusion->GetInDataAnchor(1));
        // 4: add link from Const(Bias) to Other(0)
        srcs.push_back(node_bias->GetOutDataAnchor(0));
        dsts.push_back(node_other0->GetInDataAnchor(0));
        // 5: add link from BiasAdd(fusion) to Other(1)[1]
        srcs.push_back(node_bias_add_fusion->GetOutDataAnchor(0));
        dsts.push_back(node_other1->GetInDataAnchor(1));
        // 6: add link from BiasAdd(fusion) to MatMul(0)[0]
        srcs.push_back(node_bias_add_fusion->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul0->GetInDataAnchor(0));
        // 7: add link from Const(2) to MatMul(0)[1]
        srcs.push_back(node_const2->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul0->GetInDataAnchor(1));
        // 8: add link from MatMul(0) to Other(2)[1]
        srcs.push_back(node_mat_mul0->GetOutDataAnchor(0));
        dsts.push_back(node_other2->GetInDataAnchor(1));
        // 9: add link from MatMul(0) to BiasAdd(0)[0]
        srcs.push_back(node_mat_mul0->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add0->GetInDataAnchor(0));
        // 10: add link from Const(3) to BiasAdd(0)[1]
        srcs.push_back(node_const3->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add0->GetInDataAnchor(1));
        // 11: add link from BiadAdd(0) to Other(3)
        srcs.push_back(node_bias_add0->GetOutDataAnchor(0));
        dsts.push_back(node_other3->GetInDataAnchor(0));

        int start = 0;
        int end = 11;
        if (fusion_flag) {
            start = 0;
            end = 7;
        } else {
            start = 5;
            end = 11;
        }

        // add edges
        for (int i = start; i <= end; ++i)
        {
            GraphUtils::AddEdge(srcs[i], dsts[i]);
        }

        return graph;
    }
    static ComputeGraphPtr CreateAddTestGraph() {
        /*
            Const(0)    Const(1)
               \           /
                \         /
                 \       /
                  \     /
              MatMul(fusion)   Const(Bias)
                      \           /
                       \         /
                        \       /
                         \     /
                       Add(fusion)       Const(2)
                             \             /
                              \           /
                               \         /
                                \       /
                                MatMul(0)      Const(3)
                                     \            /
                                      \          /
                                       \        /
                                          Add
                                            |
                                            |
                                          Other(3)
        */
        ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
        // new a output GeTensorDesc
        GeTensorDescPtr general_ge_tensor_desc = std::make_shared<GeTensorDesc>();
        GeTensorDesc const_ge_tensor_desc;
        GeShape shape({1, 128, 229, 229});
        general_ge_tensor_desc->SetShape(shape);
        general_ge_tensor_desc->SetFormat(FORMAT_NCHW);
        general_ge_tensor_desc->SetDataType(DT_FLOAT16);
        GeShape const_shape({128});
        const_ge_tensor_desc.SetShape(const_shape);
        const_ge_tensor_desc.SetFormat(FORMAT_NCHW);
        const_ge_tensor_desc.SetDataType(DT_FLOAT16);

        NodePtr node_const0 = CreateConstNode("test/const0", general_ge_tensor_desc,
                                             graph);
        NodePtr node_const1 = CreateConstNode("test/const1", general_ge_tensor_desc,
                                             graph);
        NodePtr node_const2 = CreateConstNode("test/const2", general_ge_tensor_desc,
                                             graph);
        NodePtr node_const3 = CreateConstNode("test/const3", general_ge_tensor_desc,
                                             graph);
        node_const3->GetOpDesc()->UpdateOutputDesc(0, const_ge_tensor_desc);

        NodePtr node_bias = CreateConstNode("test/Bias", general_ge_tensor_desc,
                                           graph);
        node_bias->GetOpDesc()->UpdateOutputDesc(0, const_ge_tensor_desc);

        NodePtr node_other0 = CreateOtherNode("test/other0", general_ge_tensor_desc,
                                             graph);
        NodePtr node_other1 = CreateOtherNode("test/other1", general_ge_tensor_desc,
                                             graph);
        NodePtr node_other2 = CreateOtherNode("test/other2", general_ge_tensor_desc,
                                             graph);
        NodePtr node_other3 = CreateOtherNode("test/other3", general_ge_tensor_desc,
                                             graph);

        NodePtr node_mat_mul_fusion = CreateMatMulNode("test/MatMul_fusion",
                                                    general_ge_tensor_desc, graph);
        NodePtr node_mat_mul0 = CreateMatMulNode("test/MatMul0",
                                               general_ge_tensor_desc, graph);

        NodePtr node_bias_add_fusion = CreateAddNode("test/BiasAdd_fusion",
                                                  general_ge_tensor_desc, graph);
        node_bias_add_fusion->GetOpDesc()->UpdateInputDesc(1, const_ge_tensor_desc);
        NodePtr node_bias_add0 = CreateAddNode("test/BiasAdd0",
                                             general_ge_tensor_desc, graph);
        node_bias_add0->GetOpDesc()->UpdateInputDesc(0, const_ge_tensor_desc);

        /* add link of anchors */
        std::vector<OutDataAnchorPtr> srcs;
        std::vector<InDataAnchorPtr> dsts;
        // 0: add link from Const(0) to MatMul(fusion)[0]
        srcs.push_back(node_const0->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul_fusion->GetInDataAnchor(0));
        // 1: add link from Const(1) to MatMul(fusion)[1]
        srcs.push_back(node_const1->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul_fusion->GetInDataAnchor(1));
        // 2: add link from MatMul(fusion) to BiasAdd(fusion)[0]
        srcs.push_back(node_mat_mul_fusion->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add_fusion->GetInDataAnchor(0));
        // 3: add link from Const(Bias) to BiasAdd(fusion)[1]
        srcs.push_back(node_bias->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add_fusion->GetInDataAnchor(1));
        // 4: add link from BiasAdd(fusion) to MatMul(0)[0]
        srcs.push_back(node_bias_add_fusion->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul0->GetInDataAnchor(0));
        // 5: add link from Const(2) to MatMul(0)[1]
        srcs.push_back(node_const2->GetOutDataAnchor(0));
        dsts.push_back(node_mat_mul0->GetInDataAnchor(1));
        // 6: add link from MatMul(0) to BiasAdd(0)[0]
        srcs.push_back(node_mat_mul0->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add0->GetInDataAnchor(1));
        // 7: add link from Const(3) to BiasAdd(0)[1]
        srcs.push_back(node_const3->GetOutDataAnchor(0));
        dsts.push_back(node_bias_add0->GetInDataAnchor(0));
        // 8: add link from BiadAdd(0) to Other(3)
        srcs.push_back(node_bias_add0->GetOutDataAnchor(0));
        dsts.push_back(node_other3->GetInDataAnchor(0));

        int start = 0;
        int end = 8;

        // add edges
        for (int i = start; i <= end; ++i) {
            GraphUtils::AddEdge(srcs[i], dsts[i]);
        }

        return graph;
    }
    static Status DumpGraph(const ComputeGraphPtr graph)
    {
        for(NodePtr node : graph->GetAllNodes()) {
            printf("node name = %s.\n", node->GetName().c_str());
            for (OutDataAnchorPtr anchor : node->GetAllOutDataAnchors()) {
                for (InDataAnchorPtr peer_in_anchor : anchor->GetPeerInDataAnchors()) {
                    printf("    node name = %s[%d], out data node name = %s[%d].\n",
                           node->GetName().c_str(),
                           anchor->GetIdx(),
                           peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                           peer_in_anchor->GetIdx());
                }
            }
            if (node->GetOutControlAnchor() != nullptr) {
                for (InControlAnchorPtr peer_in_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
                    printf("    node name = %s, out control node name = %s.\n", node->GetName().c_str(),
                           peer_in_anchor->GetOwnerNode()->GetName().c_str());
                }
            }
        }
    }
  };

/*
*  测试函数： Status ExtremumGradFusionPass::Run(ge::ComputeGraphPtr graph)
*  场景：图优化-MaximumGradFusion
*  结果： SUCCESS
*/
//  TEST_F(fusion_matmul_biasadd_pass_st, matmul_biasadd_fusion_success)
//  {
//      // GetContext().type = TENSORFLOW;
//      ComputeGraphPtr graph = CreateTestGraph(true);
//      // printf("-----------------------graph before fusion-----------------------\n");
//      // DumpGraph(graph);
//      // printf("-----------------------------------------------------------------\n");
//      MatMulBiasAddFusionPass pass;
//      fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
//      // printf("-----------------------graph after fusion-----------------------\n");
//      // DumpGraph(graph);
//      // printf("-----------------------------------------------------------------\n");
//      EXPECT_EQ(fe::SUCCESS, status);
//  }
//
//  TEST_F(fusion_matmul_biasadd_pass_st, matmul_biasadd_fusion_faild)
//  {
//      // GetContext().type = TENSORFLOW;
//      ComputeGraphPtr graph = CreateTestGraph(false);
//      // printf("-----------------------graph before fusion-----------------------\n");
//      // DumpGraph(graph);
//      // printf("-----------------------------------------------------------------\n");
//      MatMulBiasAddFusionPass pass;
//      fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
//      // printf("-----------------------graph after fusion-----------------------\n");
//      // DumpGraph(graph);
//      // printf("-----------------------------------------------------------------\n");
//      EXPECT_EQ(fe::NOT_CHANGED, status);
//  }
//
//    TEST_F(fusion_matmul_biasadd_pass_st, matmul_add_fusion_success)
//    {
//        // GetContext().type = TENSORFLOW;
//        ComputeGraphPtr graph = CreateAddTestGraph();
//        // printf("-----------------------graph before fusion-----------------------\n");
//        // DumpGraph(graph);
//        // printf("-----------------------------------------------------------------\n");
//        MatMulBiasAddFusionPass pass;
//        fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
//        // printf("-----------------------graph after fusion-----------------------\n");
//        // DumpGraph(graph);
//        // printf("-----------------------------------------------------------------\n");
//        EXPECT_EQ(fe::SUCCESS, status);
//    }
}
