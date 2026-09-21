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

#include "common/util/op_info_util.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph_constructor/pass_manager.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace ge;
using namespace fe;

using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;
using HeavyFormatPropagationPtr = std::shared_ptr<HeavyFormatPropagation>;
class UTEST_fusion_engine_heavy_format_distribution : public testing::Test
{
 protected:
  void SetUp()
  {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo heavy_op_info {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/heavy_opinfo",
        "",
        false,
        false,
        false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(heavy_op_info);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);

    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
  }

  void TearDown()
  {

  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
 protected:
    static void CreateConv3dGraph01(ComputeGraphPtr graph) {
        /*           Data(NCDHW)          Varable(NCDHW)
         *             \                   /        \
         *        Conv3D (NDC1HWC0,FRACTAL_Z_3D)   ApplyMomentum(NCDHW)
         *               |   Const(NCDHW)
         *               |   /
         *              BiasAdd(NCDHW)
         * */
        // Conv3D
        OpDescPtr data_op = std::make_shared<OpDesc>("data", fe::DATA);
        vector<int64_t> data_dims = {1, 2, 3, 4, 5};
        GeTensorDesc data_tensor_desc(GeShape(data_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
        data_tensor_desc.SetOriginShape(GeShape(data_dims));
        data_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
        data_op->AddOutputDesc(data_tensor_desc);
        data_op->AddInputDesc(data_tensor_desc);
        auto data_node = graph->AddNode(data_op);

        OpDescPtr variable_op = std::make_shared<OpDesc>("varibale", fe::VARIABLE);
        vector<int64_t> variable_dims = {6, 7, 8, 9, 10};
        GeTensorDesc variable_tensor_desc(GeShape(variable_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
        variable_tensor_desc.SetOriginShape(GeShape(variable_dims));
        variable_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
        variable_op->AddOutputDesc(variable_tensor_desc);
        variable_op->AddInputDesc(variable_tensor_desc);
        auto variable_node = graph->AddNode(variable_op);

        OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv3d", "Conv3D");
        GeTensorDesc conv_tensor_desc(GeShape({1, 3, 1, 4, 5, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
        conv_tensor_desc.SetOriginShape(GeShape(data_dims));
        conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
        GeTensorDesc conv_tensor_desc_weight(GeShape({720, 1, 16, 16}), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
        conv_tensor_desc_weight.SetOriginShape(GeShape(variable_dims));
        conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCDHW);
        conv_o_p->AddInputDesc(conv_tensor_desc);
        conv_o_p->AddInputDesc(conv_tensor_desc_weight);
        conv_o_p->AddOutputDesc(conv_tensor_desc);
        auto conv_node = graph->AddNode(conv_o_p);
        ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

        // ApplyMomentum
        OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am1", "ApplyMomentum");
        GeTensorDesc am_tensor_desc(GeShape(variable_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
        am_tensor_desc.SetOriginShape(GeShape(variable_dims));
        am_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
        apply_momentum_op->AddOutputDesc(am_tensor_desc);
        for (uint32_t i = 0;  i < 5; i++) {
            apply_momentum_op->AddInputDesc(am_tensor_desc);
        }
        auto am_node = graph->AddNode(apply_momentum_op);
        ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);


        // BiasAdd
        OpDescPtr const_op = std::make_shared<OpDesc>("const1", fe::CONSTANT);
        GeTensorDesc const_tensor_desc(GeShape(data_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
        const_tensor_desc.SetOriginShape(GeShape(data_dims));
        const_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
        const_op->AddInputDesc(const_tensor_desc);
        const_op->AddOutputDesc(const_tensor_desc);
        auto const_node = graph->AddNode(const_op);

        OpDescPtr bias_add_op = std::make_shared<OpDesc>("biassAdd", "BiasAdd");
        GeTensorDesc bias_add_tensor_desc(GeShape(data_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
        bias_add_tensor_desc.SetOriginShape(GeShape(data_dims));
        bias_add_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
        bias_add_op->AddInputDesc(bias_add_tensor_desc);
        bias_add_op->AddInputDesc(const_tensor_desc);
        bias_add_op->AddOutputDesc(bias_add_tensor_desc);
        auto biass_add_node = graph->AddNode(bias_add_op);
        ge::AttrUtils::SetInt(bias_add_op, FE_IMPLY_TYPE, 6);

        GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
        GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
        GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(0));
        GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(1));
        GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
    }


    static void CreateConv3dGraph02(ComputeGraphPtr graph) {
        /*           Data(NDCHW)          Varable(NDCHW)
         *             \                   /        \
         *        Conv3D (NDC1HWC0,FRACTAL_Z_3D)   ApplyMomentum(NDCHW)
         *               |   Const
         *               |   /
         *              BiasAdd
         * */
        // Conv3D
        OpDescPtr data_op = std::make_shared<OpDesc>("data", fe::DATA);
        vector<int64_t> data_dims = {1, 2, 3, 4, 5};
        GeTensorDesc data_tensor_desc(GeShape(data_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
        data_tensor_desc.SetOriginShape(GeShape(data_dims));
        data_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
        data_op->AddOutputDesc(data_tensor_desc);
        data_op->AddInputDesc(data_tensor_desc);
        auto data_node = graph->AddNode(data_op);

        OpDescPtr variable_op = std::make_shared<OpDesc>("variabel", fe::VARIABLE);
        vector<int64_t> variable_dims = {6, 7, 8, 9, 10};
        GeTensorDesc variable_tensor_desc(GeShape(variable_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
        variable_tensor_desc.SetOriginShape(GeShape(variable_dims));
        variable_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
        variable_op->AddOutputDesc(variable_tensor_desc);
        variable_op->AddInputDesc(variable_tensor_desc);
        auto variable_node = graph->AddNode(variable_op);

        OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv3d", "Conv3D");
        GeTensorDesc conv_tensor_desc(GeShape({1, 2, 1, 3, 4, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
        conv_tensor_desc.SetOriginShape(GeShape(data_dims));
        conv_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
        GeTensorDesc conv_tensor_desc_weight(GeShape({504, 1, 16, 16}), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
        conv_tensor_desc_weight.SetOriginShape(GeShape(variable_dims));
        conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NDHWC);
        conv_o_p->AddInputDesc(conv_tensor_desc);
        conv_o_p->AddInputDesc(conv_tensor_desc_weight);
        conv_o_p->AddOutputDesc(conv_tensor_desc);
        auto conv_node = graph->AddNode(conv_o_p);
        ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

        // ApplyMomentum
        OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am1", "ApplyMomentum");
        GeTensorDesc am_tensor_desc(GeShape(variable_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
        am_tensor_desc.SetOriginShape(GeShape(variable_dims));
        am_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
        apply_momentum_op->AddOutputDesc(am_tensor_desc);
        for (uint32_t i = 0;  i < 5; i++) {
            apply_momentum_op->AddInputDesc(am_tensor_desc);
        }
        auto am_node = graph->AddNode(apply_momentum_op);
        ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);


        // BiasAdd
        OpDescPtr const_op = std::make_shared<OpDesc>("const1", fe::CONSTANT);
        GeTensorDesc const_tensor_desc(GeShape(data_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
        const_tensor_desc.SetOriginShape(GeShape(data_dims));
        const_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
        const_op->AddInputDesc(const_tensor_desc);
        const_op->AddOutputDesc(const_tensor_desc);
        auto const_node = graph->AddNode(const_op);

        OpDescPtr bias_add_op = std::make_shared<OpDesc>("biassAdd", "BiasAdd");
        GeTensorDesc bias_add_tensor_desc(GeShape(data_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
        bias_add_tensor_desc.SetOriginShape(GeShape(data_dims));
        bias_add_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
        bias_add_op->AddInputDesc(bias_add_tensor_desc);
        bias_add_op->AddInputDesc(const_tensor_desc);
        bias_add_op->AddOutputDesc(bias_add_tensor_desc);
        auto biass_add_node = graph->AddNode(bias_add_op);
        ge::AttrUtils::SetInt(bias_add_op, FE_IMPLY_TYPE, 6);

        GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
        GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
        GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(0));
        GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(1));
        GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
    }

  static void CreateConv3dGraph03(ComputeGraphPtr graph) {
    /*           Data(NCDHW)          Varable(NCDHW)
     *             \                   /        \
     *        Conv3D (NDC1HWC0,FRACTAL_Z_3D)   ApplyMomentum(NCDHW)
     *               |   Const(NCDHW)
     *               |   /
     *              BiasAdd(NCDHW)
     * */
    // Conv3D
    OpDescPtr data_op = std::make_shared<OpDesc>("data", fe::DATA);
    vector<int64_t> data_dims = {1, 4, 5};
    GeTensorDesc data_tensor_desc(GeShape(data_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape(data_dims));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    vector<int64_t> variable_dims = {6, 7, 8, 9, 10};

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv3d", "Conv3D");
    GeTensorDesc conv_tensor_desc(GeShape({1, 4, 1, 5, 1, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape(data_dims));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({720, 1, 16, 16}), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape(variable_dims));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCDHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    // BiasAdd
    OpDescPtr const_op = std::make_shared<OpDesc>("const1", fe::CONSTANT);
    GeTensorDesc const_tensor_desc(GeShape(data_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape(data_dims));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    const_op->AddInputDesc(const_tensor_desc);
    const_op->AddOutputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr bias_add_op = std::make_shared<OpDesc>("biassAdd1", "BiasAdd");
    vector<int64_t> bias_add_dims = {55};
    GeTensorDesc bias_add_tensor_desc(GeShape(bias_add_dims), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    bias_add_tensor_desc.SetOriginShape(GeShape(bias_add_dims));
    bias_add_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    bias_add_op->AddInputDesc(bias_add_tensor_desc);
    bias_add_op->AddInputDesc(const_tensor_desc);
    bias_add_op->AddOutputDesc(bias_add_tensor_desc);
    auto biass_add_node = graph->AddNode(bias_add_op);
    ge::AttrUtils::SetInt(bias_add_op, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_op_2 = std::make_shared<OpDesc>("biassAdd2", "BiasAdd");
    vector<int64_t> bias_add_dims_2 = {55, 66};
    GeTensorDesc bias_add_tensor_desc_2(GeShape(bias_add_dims_2), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    bias_add_tensor_desc_2.SetOriginShape(GeShape(bias_add_dims_2));
    bias_add_tensor_desc_2.SetOriginFormat(ge::FORMAT_NCDHW);
    bias_add_op_2->AddInputDesc(bias_add_tensor_desc_2);
    bias_add_op_2->AddInputDesc(bias_add_tensor_desc_2);
    bias_add_op_2->AddOutputDesc(bias_add_tensor_desc_2);
    auto biass_add_node_2 = graph->AddNode(bias_add_op_2);
    ge::AttrUtils::SetInt(bias_add_op_2, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_op_3 = std::make_shared<OpDesc>("biassAdd3", "BiasAdd");
    vector<int64_t> bias_add_dims_3 = {55, 66, 77};
    GeTensorDesc bias_add_tensor_desc_3(GeShape(bias_add_dims_3), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    bias_add_tensor_desc_3.SetOriginShape(GeShape(bias_add_dims_3));
    bias_add_tensor_desc_3.SetOriginFormat(ge::FORMAT_NCDHW);
    bias_add_op_3->AddInputDesc(bias_add_tensor_desc_3);
    bias_add_op_3->AddInputDesc(bias_add_tensor_desc_3);
    bias_add_op_3->AddOutputDesc(bias_add_tensor_desc_3);
    auto biass_add_node_3 = graph->AddNode(bias_add_op_3);
    ge::AttrUtils::SetInt(bias_add_op_3, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_op_4 = std::make_shared<OpDesc>("biassAdd4", "BiasAdd");
    vector<int64_t> bias_add_dims_4 = {55, 66, 77, 88};
    GeTensorDesc bias_add_tensor_desc_4(GeShape(bias_add_dims_4), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    bias_add_tensor_desc_4.SetOriginShape(GeShape(bias_add_dims_4));
    bias_add_tensor_desc_4.SetOriginFormat(ge::FORMAT_NCDHW);
    bias_add_op_4->AddInputDesc(bias_add_tensor_desc_4);
    bias_add_op_4->AddInputDesc(bias_add_tensor_desc_4);
    bias_add_op_4->AddOutputDesc(bias_add_tensor_desc_4);
    auto biass_add_node_4 = graph->AddNode(bias_add_op_4);
    ge::AttrUtils::SetInt(bias_add_op_4, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(1));

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node_2->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node_2->GetInDataAnchor(1));

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node_3->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node_3->GetInDataAnchor(1));

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node_4->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node_4->GetInDataAnchor(1));
  }

  static void CreateConv3dGraph04(ComputeGraphPtr graph) {
    /*           Data(NCDHW)          Varable(NCDHW)
     *             \                   /        \
     *        Conv3D (NDC1HWC0,FRACTAL_Z_3D)   ApplyMomentum(NCDHW)
     *               |   Const(NCDHW)
     *               |   /
     *              BiasAdd(NCDHW)
     * */
    // Conv3D
    OpDescPtr data_op = std::make_shared<OpDesc>("data", fe::DATA);
    vector<int64_t> data_dims = {1, 4, 5};
    GeTensorDesc data_tensor_desc(GeShape(data_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape(data_dims));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    vector<int64_t> variable_dims = {6, 7, 8, 9, 10};
    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv3d", "Conv3D");
    GeTensorDesc conv_tensor_desc(GeShape({1, 1, 1, 4, 5, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape(data_dims));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    GeTensorDesc conv_tensor_desc_weight(GeShape({720, 1, 16, 16}), ge::FORMAT_FRACTAL_Z_3D, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape(variable_dims));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCDHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    // BiasAdd
    OpDescPtr const_op = std::make_shared<OpDesc>("const1", fe::CONSTANT);
    GeTensorDesc const_tensor_desc(GeShape(data_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape(data_dims));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    const_op->AddInputDesc(const_tensor_desc);
    const_op->AddOutputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr bias_add_op = std::make_shared<OpDesc>("biassAdd1", "BiasAdd");
    vector<int64_t> bias_add_dims = {55};
    GeTensorDesc bias_add_tensor_desc(GeShape(bias_add_dims), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    bias_add_tensor_desc.SetOriginShape(GeShape(bias_add_dims));
    bias_add_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    bias_add_op->AddInputDesc(bias_add_tensor_desc);
    bias_add_op->AddInputDesc(const_tensor_desc);
    bias_add_op->AddOutputDesc(bias_add_tensor_desc);
    auto biass_add_node = graph->AddNode(bias_add_op);
    ge::AttrUtils::SetInt(bias_add_op, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_op_2 = std::make_shared<OpDesc>("biassAdd2", "BiasAdd");
    vector<int64_t> bias_add_dims_2 = {55, 66};
    GeTensorDesc bias_add_tensor_desc_2(GeShape(bias_add_dims_2), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    bias_add_tensor_desc_2.SetOriginShape(GeShape(bias_add_dims_2));
    bias_add_tensor_desc_2.SetOriginFormat(ge::FORMAT_NDHWC);
    bias_add_op_2->AddInputDesc(bias_add_tensor_desc_2);
    bias_add_op_2->AddInputDesc(bias_add_tensor_desc_2);
    bias_add_op_2->AddOutputDesc(bias_add_tensor_desc_2);
    auto biass_add_node_2 = graph->AddNode(bias_add_op_2);
    ge::AttrUtils::SetInt(bias_add_op_2, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_op_3 = std::make_shared<OpDesc>("biassAdd3", "BiasAdd");
    vector<int64_t> bias_add_dims_3 = {55, 66, 77};
    GeTensorDesc bias_add_tensor_desc_3(GeShape(bias_add_dims_3), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    bias_add_tensor_desc_3.SetOriginShape(GeShape(bias_add_dims_3));
    bias_add_tensor_desc_3.SetOriginFormat(ge::FORMAT_NDHWC);
    bias_add_op_3->AddInputDesc(bias_add_tensor_desc_3);
    bias_add_op_3->AddInputDesc(bias_add_tensor_desc_3);
    bias_add_op_3->AddOutputDesc(bias_add_tensor_desc_3);
    auto biass_add_node_3 = graph->AddNode(bias_add_op_3);
    ge::AttrUtils::SetInt(bias_add_op_3, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_op_4 = std::make_shared<OpDesc>("biassAdd4", "BiasAdd");
    vector<int64_t> bias_add_dims_4 = {55, 66, 77, 88};
    GeTensorDesc bias_add_tensor_desc_4(GeShape(bias_add_dims_4), ge::FORMAT_NDHWC, ge::DT_FLOAT16);
    bias_add_tensor_desc_4.SetOriginShape(GeShape(bias_add_dims_4));
    bias_add_tensor_desc_4.SetOriginFormat(ge::FORMAT_NDHWC);
    bias_add_op_4->AddInputDesc(bias_add_tensor_desc_4);
    bias_add_op_4->AddInputDesc(bias_add_tensor_desc_4);
    bias_add_op_4->AddOutputDesc(bias_add_tensor_desc_4);
    auto biass_add_node_4 = graph->AddNode(bias_add_op_4);
    ge::AttrUtils::SetInt(bias_add_op_4, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node->GetInDataAnchor(1));

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node_2->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node_2->GetInDataAnchor(1));

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node_3->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node_3->GetInDataAnchor(1));

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), biass_add_node_4->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), biass_add_node_4->GetInDataAnchor(1));
  }

  static void CreateThreeGraph(ComputeGraphPtr graph) {
    /*              Const
     *        /                \
     *  conv (NC1HWC0)         a.m. (NCHW)
     *  After distribution, the input of a.m. will become NC1HWC0*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }

  static void CreateUnknownShapeGraph(ComputeGraphPtr graph, ge::Format format) {
    /*              Const
     *        /                \
     *  conv (NC1HWC0)         a.m. (NCHW)
     *  After distribution, the input of a.m. will become NC1HWC0*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 1, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    vector<std::pair<int64_t, int64_t>> range1({{1, 3}});
    GeTensorDesc am_tensor_desc(GeShape({-1, 4, 5, 6}), format, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({-1, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(format);
    am_tensor_desc.SetShapeRange(range1);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }

  static void CreateReduceOpGraph(ComputeGraphPtr graph, ge::Format format, ge::GeShape &shape,
      std::vector<int64_t> axis_value) {
    /*     Conv(shape {3, 1, 5, 6, 16})
     *            /                  \
     *     reduce_op (NCHW,shape 3, 4, 5, 6)
     *        /                        \
     *  cosh (NCHW,shape{3, 4, 5, 6})
     * */
    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(format);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    GeTensorDesc redcue_tensor_desc(GeShape({3, 4, 5, 6}), format, ge::DT_FLOAT16);
    redcue_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    redcue_tensor_desc.SetOriginFormat(format);
    GeTensorDesc oredcue_tensor_desc(shape, format, ge::DT_FLOAT16);
    oredcue_tensor_desc.SetOriginShape(shape);
    oredcue_tensor_desc.SetOriginFormat(format);
    reduce_o_p->AddInputDesc("x", redcue_tensor_desc);
    reduce_o_p->AddOutputDesc("y", oredcue_tensor_desc);
    auto reduce_node = graph->AddNode(reduce_o_p);
    ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, false);
    ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, axis_value);

    OpDescPtr cosh_o_p = std::make_shared<OpDesc>("cosh", "Cosh");
    GeTensorDesc cosh_tensor_desc(shape, format, ge::DT_FLOAT16);
    cosh_tensor_desc.SetOriginShape(shape);
    cosh_tensor_desc.SetOriginFormat(format);
    cosh_o_p->AddInputDesc(cosh_tensor_desc);
    cosh_o_p->AddOutputDesc(cosh_tensor_desc);
    auto cosh_node = graph->AddNode(cosh_o_p);
    ge::AttrUtils::SetInt(cosh_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr add_o_p = std::make_shared<OpDesc>("add", "Add");
    GeTensorDesc add_tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    add_tensor_desc.SetOriginShape(shape);
    add_tensor_desc.SetOriginFormat(format);
    add_o_p->AddInputDesc(add_tensor_desc);
    add_o_p->AddOutputDesc(add_tensor_desc);
    auto add_node = graph->AddNode(add_o_p);
    ge::AttrUtils::SetInt(add_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), reduce_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(reduce_node->GetOutDataAnchor(0), cosh_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cosh_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  }

  static void CreateThreeGraphWithScalar(ComputeGraphPtr graph) {
    /*              Const(Scalar)
     *        /                \
     *  conv (NC1HWC0)         a.m. (NCHW)
     *  After distribution, the input of a.m. will still be NCHW*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape());
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }


  static void CreateThreeGraphWithScalar_1(ComputeGraphPtr graph) {
    /*              Const(Scalar, shape {3,4,5,6})
     *        /                     \
     *  conv (NC1HWC0,shape{})         a.m. (NCHW)
     *  After distribution, the input of a.m. will become NC1HWC0*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3,4,5,6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3,4,5,6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape(), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }


  static void CreateThreeGraphWithScalar_2(ComputeGraphPtr graph) {
    /*              Const(shape {3,4,5,6})
     *            /                  \
     *     a.m.2 (NCHW,shape 0)       \
     *        /                        \
     *  conv (NC1HWC0,shape{})         a.m.1(NCHW)
     *  After distribution, the input of a.m. will still be NCHW */
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3,4,5,6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3,4,5,6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape(), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op2 = std::make_shared<OpDesc>("am2", "ApplyMomentum");
    GeTensorDesc am_tensor_desc2(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc2.SetOriginShape(GeShape());
    am_tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op2->AddOutputDesc(am_tensor_desc2);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op2->AddInputDesc(am_tensor_desc2);
    }
    auto am_node_2 = graph->AddNode(apply_momentum_op2);
    ge::AttrUtils::SetInt(apply_momentum_op2, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node_2->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node_2->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }


  static void CreateThreeGraphWithL2LossAndAddN(ComputeGraphPtr graph) {
     /* conv2_d_back_prop_filter(Fragz)      Conv2D(NC1HWC0)
      *          |                       /
      *        a.m.(NCHW)          L2Loss (NCHW)
      *               \           /
      *                 AddN(NCHW)
      *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    l2_loss_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc l2_loss_out_tensor_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT16);
    l2_loss_out_tensor_desc.SetOriginShape(GeShape({1}));
    l2_loss_out_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_out_tensor_desc);
    auto l2loss_node = graph->AddNode(l2_loss_op);
    ge::AttrUtils::SetInt(l2_loss_op, FE_IMPLY_TYPE, 6);

    OpDescPtr add_no_p = std::make_shared<OpDesc>("addn", "AddN");
    add_no_p->AddInputDesc(l2_loss_out_tensor_desc);
    add_no_p->AddInputDesc(l2_loss_out_tensor_desc);
    add_no_p->AddOutputDesc(l2_loss_out_tensor_desc);
    auto addn_Node = graph->AddNode(add_no_p);
    ge::AttrUtils::SetInt(add_no_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), l2loss_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), addn_Node->GetInDataAnchor(1));
    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), addn_Node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }


  static void CreateThreeGraphWithL2LossAndAddN_1(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)      Conv2D(NC1HWC0)
     *          |                       /
     *        a.m.(NCHW)          L2Loss (NCHW)
     *               \           /
     *                 AddN(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    l2_loss_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc l2_loss_out_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_out_tensor_desc.SetOriginShape(GeShape());
    l2_loss_out_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_out_tensor_desc);
    auto l2loss_node = graph->AddNode(l2_loss_op);
    ge::AttrUtils::SetInt(l2_loss_op, FE_IMPLY_TYPE, 6);

    OpDescPtr add_no_p = std::make_shared<OpDesc>("addn", "AddN");
    add_no_p->AddInputDesc(l2_loss_out_tensor_desc);
    add_no_p->AddInputDesc(l2_loss_out_tensor_desc);
    add_no_p->AddOutputDesc(l2_loss_out_tensor_desc);
    auto addn_Node = graph->AddNode(add_no_p);
    ge::AttrUtils::SetInt(add_no_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), l2loss_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), addn_Node->GetInDataAnchor(1));
    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), addn_Node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }


  static void CreateThreeGraphWithBiasAdd(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)
     *          |
     *        a.m.(NCHW)
     *               \
     *                 BiasAdd(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_o_p = std::make_shared<OpDesc>("bias_add", "BiasAddGrad");
    bias_add_o_p->AddInputDesc(am_tensor_desc);
    bias_add_o_p->AddOutputDesc(am_tensor_desc);
    auto bias_node = graph->AddNode(bias_add_o_p);
    ge::AttrUtils::SetInt(bias_add_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }

  static void CreateThreeGraphWithTransData(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)
     *          |
     *        Transdata(NCHW->Fragz)
     *               \
     *                 BiasAdd(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr trans_op = std::make_shared<OpDesc>("trans", "TransData");
    GeTensorDesc trans_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    trans_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    trans_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    trans_op->AddOutputDesc(conv_tensor_desc_weight);
    trans_op->AddInputDesc(trans_tensor_desc);

    auto trans_node = graph->AddNode(trans_op);
    ge::AttrUtils::SetInt(trans_op, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_o_p = std::make_shared<OpDesc>("bias_add", "BiasAddGrad");
    bias_add_o_p->AddInputDesc(trans_tensor_desc);
    bias_add_o_p->AddOutputDesc(trans_tensor_desc);
    auto bias_node = graph->AddNode(bias_add_o_p);
    ge::AttrUtils::SetInt(bias_add_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(trans_node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), trans_node->GetInDataAnchor(0));
  }


  static void CreateThreeGraphWithTransData_2(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)
     *          |
     *        Transdata(Fragz->NCHW)
     *               \
     *                 BiasAdd(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr trans_op = std::make_shared<OpDesc>("trans", "TransData");
    GeTensorDesc trans_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    trans_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    trans_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    trans_op->AddOutputDesc(trans_tensor_desc);
    trans_op->AddInputDesc(conv_tensor_desc_weight);

    auto trans_node = graph->AddNode(trans_op);
    ge::AttrUtils::SetInt(trans_op, FE_IMPLY_TYPE, 6);

    OpDescPtr bias_add_o_p = std::make_shared<OpDesc>("bias_add", "BiasAddGrad");
    bias_add_o_p->AddInputDesc(trans_tensor_desc);
    bias_add_o_p->AddOutputDesc(trans_tensor_desc);
    auto bias_node = graph->AddNode(bias_add_o_p);
    ge::AttrUtils::SetInt(bias_add_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(trans_node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), trans_node->GetInDataAnchor(0));
  }

  static void CreateThreeGraphWithL2LossAndMul(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)
     *          |
     *        a.m.(NCHW)          L2Loss (NCHW)
     *               \           /
     *                 Mul(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    l2_loss_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc l2_loss_out_tensor_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT16);
    l2_loss_out_tensor_desc.SetOriginShape(GeShape({1}));
    l2_loss_out_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_out_tensor_desc);
    auto l2loss_node = graph->AddNode(l2_loss_op);
    ge::AttrUtils::SetInt(l2_loss_op, FE_IMPLY_TYPE, 6);


    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr mul_o_p = std::make_shared<OpDesc>("mul", "Mul");
    mul_o_p->AddInputDesc(am_tensor_desc);
    mul_o_p->AddInputDesc(l2_loss_out_tensor_desc);
    mul_o_p->AddOutputDesc(am_tensor_desc);
    auto mul_Node = graph->AddNode(mul_o_p);
    ge::AttrUtils::SetInt(mul_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), mul_Node->GetInDataAnchor(1));
    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), mul_Node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }
  static void CreateFiveGraph(ComputeGraphPtr graph) {
    /*   Data         Const
     *   |     /                \
     *  conv (Fz)         a.m. (NCHW)     a.m.3(NCHW)
     *                          |      /
     *                         a.m.2 (NCHW)
     *  After distribution, the input of a.m. will become NC1HWC0*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);


    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am1", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op2 = std::make_shared<OpDesc>("am2", "ApplyMomentum");
    GeTensorDesc am2_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am2_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am2_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op2->AddOutputDesc(am2_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op2->AddInputDesc(am_tensor_desc);
    }
    auto am_node2 = graph->AddNode(apply_momentum_op2);
    ge::AttrUtils::SetInt(apply_momentum_op2, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op3 = std::make_shared<OpDesc>("am3", "ApplyMomentum");
    GeTensorDesc am3_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am3_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am3_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op3->AddOutputDesc(am3_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op3->AddInputDesc(am_tensor_desc);
    }
    auto am_node3 = graph->AddNode(apply_momentum_op3);
    ge::AttrUtils::SetInt(apply_momentum_op3, FE_IMPLY_TYPE, 6);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), am_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node3->GetOutDataAnchor(0), am_node2->GetInDataAnchor(1));
  }

  static void CreateSevenGraph(ComputeGraphPtr graph) {
    /*   Data         Const                a.m.4(NCHW)
     *   |     /                \            |
     *  conv (Fz)         a.m. (NCHW)     a.m.3(NCHW)
     *                          |      /
     *                         a.m.2 (NCHW)
     *                          |
     *                         a.m.5(NCHW)
     *  After distribution, the input of a.m. will become NC1HWC0*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);


    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am1", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op2 = std::make_shared<OpDesc>("am2", "ApplyMomentum");
    apply_momentum_op2->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op2->AddInputDesc(am_tensor_desc);
    }
    auto am_node2 = graph->AddNode(apply_momentum_op2);
    ge::AttrUtils::SetInt(apply_momentum_op2, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op3 = std::make_shared<OpDesc>("am3", "ApplyMomentum");
    apply_momentum_op3->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op3->AddInputDesc(am_tensor_desc);
    }
    auto am_node3 = graph->AddNode(apply_momentum_op3);
    ge::AttrUtils::SetInt(apply_momentum_op3, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op4 = std::make_shared<OpDesc>("am4", "ApplyMomentum");
    apply_momentum_op4->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op4->AddInputDesc(am_tensor_desc);
    }
    auto am_node4 = graph->AddNode(apply_momentum_op4);
    ge::AttrUtils::SetInt(apply_momentum_op4, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op5 = std::make_shared<OpDesc>("am5", "ApplyMomentum");
    apply_momentum_op5->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op5->AddInputDesc(am_tensor_desc);
    }
    auto am_node5 = graph->AddNode(apply_momentum_op5);
    ge::AttrUtils::SetInt(apply_momentum_op5, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), am_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node3->GetOutDataAnchor(0), am_node2->GetInDataAnchor(1));

    GraphUtils::AddEdge(am_node4->GetOutDataAnchor(0), am_node3->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node2->GetOutDataAnchor(0), am_node5->GetInDataAnchor(0));
  }

  static void CreateElevenGraph(ComputeGraphPtr graph) {
    /*   Data         Const
     *   |     /                \
     *  conv (Fz)         a.m. (NCHW)  a.m.3,4,5,6(NCHW)
     *                          |       ////
     *                         a.m.2 (NCHW)
   *                        /   |   \
   *                    a.m.7  a.m.8 a.m.9
     *  After distribution, the input of a.m. will become NC1HWC0*/
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW,
                                   ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW,
                                  ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0,
                                  ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);


    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}),
                                         ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am1", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW,
                                ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op2 = std::make_shared<OpDesc>("am2", "ApplyMomentum");
    apply_momentum_op2->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op2->AddInputDesc(am_tensor_desc);
    }
    auto am_node2 = graph->AddNode(apply_momentum_op2);
    ge::AttrUtils::SetInt(apply_momentum_op2, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op3 = std::make_shared<OpDesc>("am3", "ApplyMomentum");
    apply_momentum_op3->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op3->AddInputDesc(am_tensor_desc);
    }
    auto am_node3 = graph->AddNode(apply_momentum_op3);
    ge::AttrUtils::SetInt(apply_momentum_op3, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op4 = std::make_shared<OpDesc>("am4", "ApplyMomentum");
    apply_momentum_op4->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op4->AddInputDesc(am_tensor_desc);
    }
    auto am_node4 = graph->AddNode(apply_momentum_op4);
    ge::AttrUtils::SetInt(am_tensor_desc, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op5 = std::make_shared<OpDesc>("am5", "ApplyMomentum");
    apply_momentum_op5->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op5->AddInputDesc(am_tensor_desc);
    }
    auto am_node5 = graph->AddNode(apply_momentum_op5);
    ge::AttrUtils::SetInt(apply_momentum_op5, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op6 = std::make_shared<OpDesc>("am6", "ApplyMomentum");
    apply_momentum_op6->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op6->AddInputDesc(am_tensor_desc);
    }
    auto am_node6 = graph->AddNode(apply_momentum_op6);
    ge::AttrUtils::SetInt(apply_momentum_op6, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op7 = std::make_shared<OpDesc>("am7", "ApplyMomentum");
    apply_momentum_op7->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op7->AddInputDesc(am_tensor_desc);
    }
    auto am_node7 = graph->AddNode(apply_momentum_op7);
    ge::AttrUtils::SetInt(apply_momentum_op7, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op8 = std::make_shared<OpDesc>("am8", "ApplyMomentum");
    apply_momentum_op8->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op8->AddInputDesc(am_tensor_desc);
    }
    auto am_node8 = graph->AddNode(apply_momentum_op8);
    ge::AttrUtils::SetInt(apply_momentum_op8, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op9 = std::make_shared<OpDesc>("am9", "ApplyMomentum");
    apply_momentum_op9->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op9->AddInputDesc(am_tensor_desc);
    }
    auto am_node9 = graph->AddNode(apply_momentum_op9);
    ge::AttrUtils::SetInt(apply_momentum_op9, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), am_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node3->GetOutDataAnchor(0), am_node2->GetInDataAnchor(1));
    GraphUtils::AddEdge(am_node4->GetOutDataAnchor(0), am_node2->GetInDataAnchor(2));
    GraphUtils::AddEdge(am_node5->GetOutDataAnchor(0), am_node2->GetInDataAnchor(3));
    GraphUtils::AddEdge(am_node6->GetOutDataAnchor(0), am_node2->GetInDataAnchor(4));

    GraphUtils::AddEdge(am_node2->GetOutDataAnchor(0), am_node7->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node2->GetOutDataAnchor(0), am_node8->GetInDataAnchor(0));
    GraphUtils::AddEdge(am_node2->GetOutDataAnchor(0), am_node9->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_fusion_engine_heavy_format_distribution, reduceop_nchw_0_to_nc1hwc0)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto reduced_shape = ge::GeShape({4,5,6});
  CreateReduceOpGraph(graph, ge::FORMAT_NCHW, reduced_shape, {0});
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim_none = {};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim5_h_d_reduce = {1, 1, 5, 6, 16};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "ReduceOp") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Cosh") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, reduceop_nhwc_12_to_nc1hwc0)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto reduced_shape = ge::GeShape({3, 1, 1, 6});
  CreateReduceOpGraph(graph, ge::FORMAT_NHWC, reduced_shape, {1,2});

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim_none = {};
  vector<int64_t> result_dim5_hd_conv = { 3, 1, 5, 6, 16};
  vector<int64_t> result_dim5_h_d = {3, 1, 4, 5, 16};
  vector<int64_t> result_dim5_h_d_reduce = {3, 1, 1, 1, 16};
  vector<int64_t> result_dim5_h_d_cos = {3, 1, 1, 1, 16};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_hd_conv, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "ReduceOp") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Cosh") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(result_dim5_h_d_cos, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d_cos, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, reduceop_hwcn_12_to_nc1hwc0)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto reduced_shape = ge::GeShape({3,6});
  CreateReduceOpGraph(graph, ge::FORMAT_HWCN, reduced_shape, {1,2});
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim_none = {};
  vector<int64_t> result_dim5_hd_conv = { 3, 1, 5, 6, 16};
  vector<int64_t> result_dim5_h_d = {4, 1, 1, 5, 16};
  vector<int64_t> result_dim5_h_d_reduce = {4, 1, 1, 1, 16};
//  for(auto node : graph->GetDirectNode()) {
//    if (node->GetType() == "Conv2D") {
//      auto opdesc = node->GetOpDesc();
//
//      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
//      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
//      EXPECT_EQ(result_dim5_hd_conv, opdesc->GetOutputDesc(0).GetShape().GetDims());
//    }
//    if (node->GetType() == "ReduceOp") {
//      auto opdesc = node->GetOpDesc();
//      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
//      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
//      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());
//
//      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
//      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
//      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
//    }
//    if (node->GetType() == "Cosh") {
//      auto opdesc = node->GetOpDesc();
//      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
//      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
//      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetInputDesc(0).GetShape().GetDims());
//      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
//      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
//    }
//  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, reduceop_chwn_12_to_nc1hwc0)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto reduced_shape = ge::GeShape({3, 1, 1, 6});
  CreateReduceOpGraph(graph, ge::FORMAT_CHWN, reduced_shape, {1,2});
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim_none = {};
  vector<int64_t> result_dim5_hd_conv = { 3, 1, 5, 6, 16};
  vector<int64_t> result_dim5_h_d = {6, 1, 4, 5, 16};
  vector<int64_t> result_dim5_h_d_reduce = {6, 1, 1, 1, 16};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_hd_conv, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "ReduceOp") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Cosh") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d_reduce, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraph(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "ApplyMomentumA") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim = {3, 4, 5, 6};
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_01_option)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraph(graph);
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "Conv2D") {
      GeTensorDesc data_desc_invalid({}, FORMAT_RESERVED, DT_UNDEFINED);
      auto opdesc = node->GetOpDesc();
      opdesc->UpdateInputDesc(0, data_desc_invalid);

    }
  }
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateFiveGraph(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  for(auto node : graph->GetDirectNode()) {
    vector<int64_t> result_original_dim = {3, 4, 5, 6};
    vector<int64_t> result_dim = {30, 1, 16, 16};
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateSevenGraph(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  for(auto node : graph->GetDirectNode()) {
    vector<int64_t> result_original_dim = {3, 4, 5, 6};
    vector<int64_t> result_dim = {30, 1, 16, 16};
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am3") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }

    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am5") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_04)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateElevenGraph(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  for(auto node : graph->GetDirectNode()) {
    vector<int64_t> result_original_dim = {3, 4, 5, 6};
    vector<int64_t> result_dim = {30, 1, 16, 16};
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(1).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(2).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(2).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(2).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(3).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(3).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(3).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(4).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(4).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(4).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am3" || node->GetName() == "am5") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }

    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am6") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_05)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateElevenGraph(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  for(auto node : graph->GetDirectNode()) {
    vector<int64_t> result_original_dim = {3, 4, 5, 6};
    vector<int64_t> result_dim = {30, 1, 16, 16};
    if (node->GetName() == "am1") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(1).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(2).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(2).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(2).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(3).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(3).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(3).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(4).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(4).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(4).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am3" || node->GetName() == "am5") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }

    if (node->GetName() == "am4") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am6") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am7"|| node->GetName() == "am8"||node->GetName() == "am9") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_06)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithScalar(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "ApplyMomentum") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim_none = {};
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_none, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> result_dim = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_06_1)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithScalar_1(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim_none = {};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetType() == "ApplyMomentum") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_none, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_06_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithScalar_2(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim_none = {};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "am2") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_none, opdesc->GetInputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Const") {
      auto opdesc = node->GetOpDesc();

      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_none, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}



TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_07)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithL2LossAndAddN(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> result_dim5_hd_l2_loss = {1,1,1,1,16};
  vector<int64_t> scalar = {1};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2DBackpropInput") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
    if (node->GetType() == "L2Loss") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(scalar, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
    if (node->GetType() == "AddN") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetInputDesc(1).GetFormat());
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(scalar, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_08)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithL2LossAndAddN_1(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> result_dim5_hd_l2_loss = {1,1,1,1,16};
  vector<int64_t> scalar = {};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2DBackpropInput") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
    if (node->GetType() == "L2Loss") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(scalar, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
    if (node->GetType() == "AddN") {
      auto opdesc = node->GetOpDesc();
      vector<int64_t> dim_fz_add_n = {1,1,16,16};
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(scalar, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_09)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithL2LossAndMul(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> scalar = {1};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2DBackpropInput") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }

    if (node->GetType() == "L2Loss") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(scalar, opdesc->GetOutputDesc(0).GetShape().GetDims());

    }
    if (node->GetType() == "Mul") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetInputDesc(1).GetFormat());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(scalar, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_10)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithBiasAdd(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> scalar = {1};
  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "am") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2DBackpropInput") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }

    if (node->GetType() == "BiasAddGrad") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_11)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithTransData(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> scalar = {1};
  for(auto node : graph->GetDirectNode()) {

    if (node->GetType() == "Conv2DBackpropInput") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "trans") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "BiasAddGrad") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_12)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithTransData_2(graph);

  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> scalar = {1};
  for(auto node : graph->GetDirectNode()) {

    if (node->GetType() == "Conv2DBackpropInput") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "trans") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "BiasAddGrad") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_unknown_shape)
{
  for (auto format : FE_ORIGIN_FORMAT_VECTOR) {
    if (format == ge::FORMAT_ND) {
      continue;
    }
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateUnknownShapeGraph(graph, format);

    HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
    heavt_format_propagator->Initialize();
    Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
    for(auto node : graph->GetDirectNode()) {
      if (node->GetType() == "ApplyMomentumA") {
        auto opdesc = node->GetOpDesc();
        vector<int64_t> result_dim = {-1, 4, 5, 6};
        EXPECT_EQ(format, opdesc->GetInputDesc(0).GetFormat());
        EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
        EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      }
      if (node->GetType() == "Const") {
        auto opdesc = node->GetOpDesc();
        vector<int64_t> result_dim = {3, 4, 5, 6};
        EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
        EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
        EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
      }
      if (node->GetType() == "Conv2D") {
        auto opdesc = node->GetOpDesc();
        vector<int64_t> result_dim = {3, 1, 5, 6, 16};
        EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
        EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
        EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
        EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
      }
    }
    EXPECT_EQ(fe::SUCCESS, ret);
  }
}

// NCDHW-->NDC1HWC0/FRACTAL_3D
TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_3d_01)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateConv3dGraph01(graph);
    HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
    heavt_format_propagator->Initialize();
    Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));

    vector<int64_t> result6d_dim = {1, 3, 1, 4, 5, 16};
    vector<int64_t> result_fz3d_dim = {720, 1, 16, 16};

    for(auto node : graph->GetDirectNode()) {
        if (node->GetType() == "BiasAdd") {
            auto opdesc = node->GetOpDesc();
            EXPECT_EQ(ge::FORMAT_NDC1HWC0, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
            EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
            EXPECT_EQ(result6d_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
        }

        if (node->GetType() == "ApplyMomentum") {
            auto opdesc = node->GetOpDesc();
            EXPECT_EQ(ge::FORMAT_FRACTAL_Z_3D, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
            EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
            EXPECT_EQ(result_fz3d_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
        }
    }

    EXPECT_EQ(fe::SUCCESS, ret);
}

// NDHWC-->NDC1HWC0/FRACTAL_3D
TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_3d_02)
{
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateConv3dGraph02(graph);
    HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
    Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));

    vector<int64_t> result6d_dim = {1, 2, 1, 3, 4, 16};
    vector<int64_t> result_fz3d_dim = {504, 1, 16, 16};

    for(auto node : graph->GetDirectNode()) {
        if (node->GetType() == "BiasAdd") {
            auto opdesc = node->GetOpDesc();
            EXPECT_EQ(ge::FORMAT_NDC1HWC0, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
            EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
            EXPECT_EQ(result6d_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
        }

        if (node->GetType() == "ApplyMomentum") {
            auto opdesc = node->GetOpDesc();
            EXPECT_EQ(ge::FORMAT_FRACTAL_Z_3D, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
            EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
            EXPECT_EQ(result_fz3d_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
        }
    }
    EXPECT_EQ(fe::SUCCESS, ret);
}

/* Need expand dims*/
TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_3d_03)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateConv3dGraph03(graph);
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> result6d_dim_1 = {1, 1, 4, 1, 1, 16}; //ori {55} C
  vector<int64_t> result6d_dim_2 = {1, 66, 4, 1, 1, 16}; //ori {55, 66} CD
  vector<int64_t> result6d_dim_3 = {1, 66, 4, 77, 1, 16}; //ori {55, 66, 77} CDH
  vector<int64_t> result6d_dim_4 = {1, 66, 4, 77, 88, 16}; //ori {55, 66, 77, 88} CDHW

  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "biasAdd1") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_1, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "biasAdd2") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_2, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "biasAdd3") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_3, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "biasAdd4") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_4, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}

/* Need expand dims*/
TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_3d_04)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateConv3dGraph03(graph);
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));

  vector<int64_t> result6d_dim_1 = {1, 1, 4, 1, 1, 16}; //ori {55} C
  vector<int64_t> result6d_dim_2 = {1, 1, 5, 1, 55, 16}; //ori {55, 66} WC
  vector<int64_t> result6d_dim_3 = {1, 1, 5, 55, 66, 16}; //ori {55, 66, 77} HWC
  vector<int64_t> result6d_dim_4 = {1, 55, 6, 66, 77, 16}; //ori {55, 66, 77, 88} DHWC

  for(auto node : graph->GetDirectNode()) {
    if (node->GetName() == "biasAdd1") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_1, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "biasAdd2") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_2, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "biasAdd3") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_3, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetName() == "biasAdd4") {
      auto opdesc = node->GetOpDesc();
      EXPECT_EQ(ge::FORMAT_NDC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result6d_dim_4, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_find_samename_variablenodes_fail)
{
  NodePtr node = nullptr;
  vector<ge::NodePtr> var_nodes;
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  heavt_format_propagator->FindSameNameVariableNodes(node, var_nodes);
  EXPECT_EQ(var_nodes.size(), 0);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_get_nextnode_info_fail)
{
  HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavt_format_propagator->Initialize();
  std::deque<NodeInfoPtr> next_node_queue;
  ge::OutDataAnchorPtr peer_out_anchor = nullptr;
  const NodeInfoPtr curr_node_info;
  const TensorInfoPtr tensor_info_ptr;
  heavt_format_propagator->GetNextNodesInfoBackWards(next_node_queue, peer_out_anchor, curr_node_info, tensor_info_ptr);
  EXPECT_EQ(next_node_queue.size(), 0);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, refresh_heavy_op_c0_val) {
  ge::GeShape shape_of_nchw({5, 20, 10, 10});
  ge::GeShape shape_of_5hd({5, 2, 10, 10, 16});
  ge::GeTensorDesc tensor_desc1(shape_of_nchw, ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::GeTensorDesc tensor_desc2(shape_of_5hd, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  ge::Format format = static_cast<ge::Format>(ge::GetFormatFromC0(ge::FORMAT_NC1HWC0, 5));
  ge::GeTensorDesc tensor_desc3(shape_of_5hd, format, ge::DT_FLOAT);

  ge::OpDescPtr data_op1 = std::make_shared<ge::OpDesc>("data1", "Data");
  data_op1->AddOutputDesc(tensor_desc1);
  AttrUtils::SetBool(data_op1, "_is_heavy_op", true);
  HeavyFormatPropagation::RefreshHeavyOpC0Val(data_op1);
  EXPECT_EQ(ge::HasC0Format(data_op1->GetOutputDescPtr(0)->GetFormat()), false);

  ge::OpDescPtr data_op2 = std::make_shared<ge::OpDesc>("data2", "Data");
  data_op2->AddOutputDesc(tensor_desc2);
  AttrUtils::SetBool(data_op2, "_is_heavy_op", true);
  HeavyFormatPropagation::RefreshHeavyOpC0Val(data_op2);
  EXPECT_EQ(ge::HasC0Format(data_op2->GetOutputDescPtr(0)->GetFormat()), true);

  ge::OpDescPtr data_op3 = std::make_shared<ge::OpDesc>("data3", "Data");
  data_op3->AddOutputDesc(tensor_desc3);
  AttrUtils::SetBool(data_op3, "_is_heavy_op", true);
  HeavyFormatPropagation::RefreshHeavyOpC0Val(data_op3);
  EXPECT_EQ(ge::HasC0Format(data_op3->GetOutputDescPtr(0)->GetFormat()), true);

  ge::OpDescPtr netout_op1 = std::make_shared<ge::OpDesc>("net_output", "NetOutput");
  netout_op1->AddInputDesc(tensor_desc1);
  AttrUtils::SetBool(netout_op1, "_is_heavy_op", true);
  HeavyFormatPropagation::RefreshHeavyOpC0Val(netout_op1);
  EXPECT_EQ(ge::HasC0Format(netout_op1->GetInputDescPtr(0)->GetFormat()), false);

  ge::OpDescPtr netout_op2 = std::make_shared<ge::OpDesc>("net_output", "NetOutput");
  netout_op2->AddInputDesc(tensor_desc2);
  AttrUtils::SetBool(netout_op2, "_is_heavy_op", true);
  HeavyFormatPropagation::RefreshHeavyOpC0Val(netout_op2);
  EXPECT_EQ(ge::HasC0Format(netout_op2->GetInputDescPtr(0)->GetFormat()), true);
}

//TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_unknown_shape)
//{
//  for (auto format : FE_ORIGIN_FORMAT_VECTOR) {
//    if (format == ge::FORMAT_ND) {
//      continue;
//    }
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateUnknownShapeGraph(graph, format);
//
//    HeavyFormatPropagationPtr heavt_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
//    heavt_format_propagator->Initialize();
//    Status ret = heavt_format_propagator->PropagateHeavyFormat(*(graph.get()));
//    for(auto node : graph->GetDirectNode()) {
//      if (node->GetType() == "ApplyMomentumA") {
//        auto opdesc = node->GetOpDesc();
//        vector<int64_t> result_dim = {-1, 4, 5, 6};
//        EXPECT_EQ(format, opdesc->GetInputDesc(0).GetFormat());
//        EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
//        EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
//      }
//      if (node->GetType() == "Const") {
//        auto opdesc = node->GetOpDesc();
//        vector<int64_t> result_dim = {3, 4, 5, 6};
//        EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
//        EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
//        EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
//      }
//      if (node->GetType() == "Conv2D") {
//        auto opdesc = node->GetOpDesc();
//        vector<int64_t> result_dim = {3, 1, 5, 6, 16};
//        EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
//        EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
//        EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
//        EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
//      }
//    }
//    EXPECT_EQ(fe::SUCCESS, ret);
//  }
//}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_heavy_format)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mm_in_nchw_out_nchw = std::make_shared<OpDesc>("mm_in_nchw_out_nchw", "MatMulV2");
  OpDescPtr mm_in_fz_out_nchw = std::make_shared<OpDesc>("mm_in_fz_out_nchw", "MatMulV2");
  OpDescPtr mm_in_nchw_out_fz = std::make_shared<OpDesc>("mm_in_nchw_out_fz", "MatMulV2");
  OpDescPtr mm_ffts = std::make_shared<OpDesc>("mm_ffts", "MatMulV2");

  // add descriptor
  ge::GeShape shape({2,4,9,16});
  GeTensorDesc tensor_desc1(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape);
  mm_in_nchw_out_nchw->AddInputDesc(tensor_desc1);
  mm_in_nchw_out_nchw->AddInputDesc(tensor_desc1);
  mm_in_nchw_out_nchw->AddOutputDesc(tensor_desc1);
  NodePtr mm_in_nchw_out_nchw_node = graph->AddNode(mm_in_nchw_out_nchw);

  GeTensorDesc tensor_desc2(shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape);
  mm_in_fz_out_nchw->AddInputDesc(tensor_desc2);
  mm_in_fz_out_nchw->AddInputDesc(tensor_desc2);
  mm_in_fz_out_nchw->AddOutputDesc(tensor_desc1);
  NodePtr mm_in_fz_out_nchw_node = graph->AddNode(mm_in_fz_out_nchw);

  mm_in_nchw_out_fz->AddInputDesc(tensor_desc1);
  mm_in_nchw_out_fz->AddInputDesc(tensor_desc1);
  mm_in_nchw_out_fz->AddOutputDesc(tensor_desc2);
  NodePtr mm_in_nchw_out_fz_node = graph->AddNode(mm_in_nchw_out_fz);


  mm_ffts->AddInputDesc(tensor_desc1);
  mm_ffts->AddInputDesc(tensor_desc1);
  mm_ffts->AddOutputDesc(tensor_desc1);
  NodePtr mm_ffts_node = graph->AddNode(mm_ffts);

  bool is_heavy_format_mm = IsHeavyFormatMatmul(mm_in_nchw_out_nchw_node);
  EXPECT_EQ(is_heavy_format_mm, false);
  is_heavy_format_mm = IsHeavyFormatMatmul(mm_in_fz_out_nchw_node);
  EXPECT_EQ(is_heavy_format_mm, true);
  is_heavy_format_mm = IsHeavyFormatMatmul(mm_in_nchw_out_fz_node);
  EXPECT_EQ(is_heavy_format_mm, true);
  is_heavy_format_mm = IsHeavyFormatMatmul(mm_ffts_node);
  EXPECT_EQ(is_heavy_format_mm, false);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_ffts)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mm_ffts = std::make_shared<OpDesc>("mm_ffts", "MatMulV2");

  // add descriptor
  ge::GeShape shape({2,4,9,16});
  GeTensorDesc tensor_desc1(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape);
  mm_ffts->AddInputDesc(tensor_desc1);
  mm_ffts->AddInputDesc(tensor_desc1);
  mm_ffts->AddOutputDesc(tensor_desc1);
  NodePtr mm_ffts_node = graph->AddNode(mm_ffts);
  (void)ge::AttrUtils::SetBool(graph, "_ffts_switch", true);

  bool is_heavy_format_mm = IsHeavyFormatMatmul(mm_ffts_node);
  EXPECT_EQ(is_heavy_format_mm, false);
  bool is_allow_nz_mm = IsAllowNzMatmul(mm_ffts_node);
  EXPECT_EQ(is_allow_nz_mm, false);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, heavy_format_distribution_allow_nz)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr mm = std::make_shared<OpDesc>("mm", "MatMulV2");

  // add descriptor
  ge::GeShape shape({2,4,9,16});
  GeTensorDesc tensor_desc1(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape);
  mm->AddInputDesc(tensor_desc1);
  mm->AddInputDesc(tensor_desc1);
  mm->AddOutputDesc(tensor_desc1);
  NodePtr mm_node = graph->AddNode(mm);
  (void)ge::AttrUtils::SetBool(mm, "allow_nz", true);

  bool is_allow_nz_mm = IsAllowNzMatmul(mm_node);
  EXPECT_EQ(is_allow_nz_mm, true);
}

TEST_F(UTEST_fusion_engine_heavy_format_distribution, is_heavy_format_consistent_with_ori)
{
  OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
  GeTensorDescPtr conv_tensor_desc = std::make_shared<GeTensorDesc>(GeShape({3, 1, 6, 16}), ge::FORMAT_NHWC, ge::DT_FLOAT16);
  conv_tensor_desc->SetOriginShape(GeShape({3, 4, 5, 6}));
  conv_tensor_desc->SetOriginFormat(ge::FORMAT_NHWC);
  bool bres = HeavyFormatSelector::IsHeavyFormatConsistentWithOriFormat(conv_tensor_desc, ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16, conv_o_p);
  EXPECT_EQ(bres, false);
  bres = HeavyFormatSelector::IsHeavyFormatConsistentWithOriFormat(conv_tensor_desc, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, conv_o_p);
  EXPECT_EQ(bres, true);
  GeTensorDescPtr conv_tensor_desc1 = std::make_shared<GeTensorDesc>(GeShape({3, 1, 6, 16}), ge::FORMAT_NHWC, ge::DT_INT64);
  conv_tensor_desc1->SetOriginShape(GeShape({3, 4, 5, 6}));
  conv_tensor_desc1->SetOriginFormat(ge::FORMAT_NHWC);
  bres = HeavyFormatSelector::IsHeavyFormatConsistentWithOriFormat(conv_tensor_desc1, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16, conv_o_p);
  EXPECT_EQ(bres, false);
}