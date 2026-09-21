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
#include "common/aicore_util_attr_define.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
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
class STEST_fusion_engine_heavy_format_distribution_fzg : public testing::Test
{
protected:
  bool static SelectTbeOpFormatStub(const te::TbeOpInfo &tbe_op_info, string &op_format_dtype_str) {
    std::vector<te::TbeOpParam> inputs;
    tbe_op_info.GetInputs(inputs);
    if (inputs.empty()) {
      return false;
    }
    vector<te::TbeOpTensor> tensors;
    inputs[0].GetTensors(tensors);
    if (tensors.empty()) {
      return false;
    }
    int32_t sub_format = 0;
    tensors[0].GetSubFormat(sub_format);
      if (sub_format == 0 || sub_format == 2) {
          op_format_dtype_str =
                  "{\"input0\":{\"name\":\"x\",\"format\":\"NCHW,FRACTAL_Z\", \"sub_format\":\"0\", "
                  "\"dtype\":\"float16,float16\"},\"output0\":{\"name\":\"y\",\"format\":\"NCHW,FRACTAL_Z\", \"sub_format\":\"0\", "
                  "\"dtype\":\"float16,float16\"}}";
      }

    if (sub_format == 3) {
        op_format_dtype_str =
                "{\"input0\":{\"name\":\"x\",\"format\":\"NCHW\", "
                "\"dtype\":\"float16\"},\"output0\":{\"name\":\"y\",\"format\":\"NCHW\", "
                "\"dtype\":\"float16\"}}";

    }
    return true;
  }

    void SetUp()
    {
      TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
      tbe_op_store_adapter_ptr->SelectTbeOpFormat = SelectTbeOpFormatStub;
      std::map<std::string, std::string> options;
      fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
      FEOpsStoreInfo heavy_op_info {
              6,
              "tbe-builtin",
              EN_IMPL_HW_TBE,
              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_group",
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
    static void CreateThreeGraphWithL2LossAndMul(ComputeGraphPtr graph) {
      /* conv2_d_back_prop_filter(Fzg)
       *          |
       *        a.m.(NCHW)          L2Loss (NCHW)
       *               \           /
       *                 Mul(NCHW)
       */
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

      auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z,4));
      GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), new_format, ge::DT_FLOAT16);
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
       *  conv (Fzg)         a.m.1 (NCHW)     a.m.3(NCHW)
       *                          |      /
       *                         a.m.2 (NCHW)
       */
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

      auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 3));
      GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), new_format, ge::DT_FLOAT16);
      conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
      conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);

      conv_o_p->AddInputDesc(conv_tensor_desc);
      conv_o_p->AddInputDesc(conv_tensor_desc_weight);
      conv_o_p->AddOutputDesc(conv_tensor_desc);
      auto conv_node = graph->AddNode(conv_o_p);
      ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

      OpDescPtr apply_momentum_op1 = std::make_shared<OpDesc>("am1", "ApplyMomentum");
      GeTensorDesc am1_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
      am1_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
      am1_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
      apply_momentum_op1->AddOutputDesc(am1_tensor_desc);
      for (uint32_t i = 0;  i < 5; i++) {
        apply_momentum_op1->AddInputDesc(am1_tensor_desc);
      }
      auto am_node = graph->AddNode(apply_momentum_op1);
      ge::AttrUtils::SetInt(apply_momentum_op1, FE_IMPLY_TYPE, 6);

      OpDescPtr apply_momentum_op2 = std::make_shared<OpDesc>("am2", "ApplyMomentum");
      GeTensorDesc am2_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
      am2_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
      am2_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
      apply_momentum_op2->AddOutputDesc(am2_tensor_desc);
      for (uint32_t i = 0;  i < 5; i++) {
        apply_momentum_op2->AddInputDesc(am1_tensor_desc);
      }
      auto am_node2 = graph->AddNode(apply_momentum_op2);
      ge::AttrUtils::SetInt(apply_momentum_op2, FE_IMPLY_TYPE, 6);

      OpDescPtr apply_momentum_op3 = std::make_shared<OpDesc>("am3", "ApplyMomentum");
      GeTensorDesc am3_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
      am3_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
      am3_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
      apply_momentum_op3->AddOutputDesc(am3_tensor_desc);
      for (uint32_t i = 0;  i < 5; i++) {
        apply_momentum_op3->AddInputDesc(am1_tensor_desc);
      }
      auto am_node3 = graph->AddNode(apply_momentum_op3);
      ge::AttrUtils::SetInt(apply_momentum_op3, FE_IMPLY_TYPE, 6);
      GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
      GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));

      GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), am_node2->GetInDataAnchor(0));
      GraphUtils::AddEdge(am_node3->GetOutDataAnchor(0), am_node2->GetInDataAnchor(1));
    }

  //  pad and max support fz(first)/fz(second)
  static void CreateSecCallFormatSelectorGraph1(ComputeGraphPtr graph) {
    /*   Data          Const
     *   |         /     \     \
     *  Conv2D (Fzg)      \    PadD (selectOpFormat, NCHW)
     *                     \            |
     *                    Maximum (broadcast, NCHW)
     *                            |
     *                    ReduceMeanD (reduce, NCHW)
     */
    GeShape origin_shape({32, 16, 1, 1});
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(origin_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(origin_shape);
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_tensor_desc.SetOriginDataType(ge::DT_FLOAT16);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(origin_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(origin_shape);
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({32, 1, 1, 1, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(origin_shape);
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 2));
    GeTensorDesc conv_tensor_desc_weight(GeShape({16, 2, 16, 16}), new_format, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(origin_shape);
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);

    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    // PadD
    OpDescPtr padd = std::make_shared<OpDesc>("padd", "PadD");
    padd->AddInputDesc(const_tensor_desc);
    padd->AddOutputDesc(const_tensor_desc);
    auto padd_node = graph->AddNode(padd);
    ge::AttrUtils::SetInt(padd, FE_IMPLY_TYPE, 6);

    // Maximum
    map<string, vector<ge::Format>> max_format_map;
    OpDescPtr max = std::make_shared<OpDesc>("max", "Maximum");
    max->AddInputDesc(const_tensor_desc);
    GeShape max_origin_shape({32, 16, 1, 2});
    GeTensorDesc max_input2_desc(max_origin_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    max_input2_desc.SetOriginShape(max_origin_shape);
    max_input2_desc.SetOriginFormat(ge::FORMAT_NCHW);
    max_input2_desc.SetOriginDataType(ge::DT_FLOAT16);
    max->AddInputDesc(max_input2_desc);
    max->AddOutputDesc(const_tensor_desc);
    auto max_node = graph->AddNode(max);
    ge::AttrUtils::SetInt(max, FE_IMPLY_TYPE, 6);

    // ReduceMeanD
    OpDescPtr reduce_mean = std::make_shared<OpDesc>("reduce_mean", "ReduceMeanD");
    reduce_mean->AddInputDesc(const_tensor_desc);
    reduce_mean->AddOutputDesc(const_tensor_desc);
    auto reduce_mean_node = graph->AddNode(reduce_mean);
    ge::AttrUtils::SetInt(reduce_mean, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetListInt(reduce_mean, "axes", {1});
    ge::AttrUtils::SetBool(reduce_mean, "keep_dims", false);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), padd_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(padd_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(max_node->GetOutDataAnchor(0), reduce_mean_node->GetInDataAnchor(0));
  }

  //  pad support fz(first)/nchw(second), max support fz(first)/fz(second)
  static void CreateSecCallFormatSelectorGraph2(ComputeGraphPtr graph) {
    /*   Data          Const
     *   |         /     \     \
     *  Conv2D (Fzg)      \    PadD (selectOpFormat, NCHW)
     *                     \            |
     *                    Maximum (broadcast, NCHW)
     *                            |
     *                    ReduceMeanD (reduce, NCHW)
     */
    GeShape origin_shape({32, 16, 1, 1});
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(origin_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(origin_shape);
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(origin_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(origin_shape);
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({32, 1, 1, 1, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(origin_shape);
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    auto new_format = static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 3));
    GeTensorDesc conv_tensor_desc_weight(GeShape({1, 2, 16, 16}), new_format, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(origin_shape);
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);

    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    // PadD
    OpDescPtr padd = std::make_shared<OpDesc>("padd", "PadD");
    padd->AddInputDesc(const_tensor_desc);
    padd->AddOutputDesc(const_tensor_desc);
    auto padd_node = graph->AddNode(padd);
    ge::AttrUtils::SetInt(padd, FE_IMPLY_TYPE, 6);

    // Maximum
    map<string, vector<ge::Format>> max_format_map;
    OpDescPtr max = std::make_shared<OpDesc>("max", "Maximum");
    max->AddInputDesc(const_tensor_desc);
    GeShape max_origin_shape({32, 16, 1, 2});
    GeTensorDesc max_input2_desc(max_origin_shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    max_input2_desc.SetOriginShape(max_origin_shape);
    max_input2_desc.SetOriginFormat(ge::FORMAT_NCHW);
    max->AddInputDesc(max_input2_desc);
    max->AddOutputDesc(const_tensor_desc);
    auto max_node = graph->AddNode(max);
    ge::AttrUtils::SetInt(max, FE_IMPLY_TYPE, 6);

    // ReduceMeanD
    OpDescPtr reduce_mean = std::make_shared<OpDesc>("reduce_mean", "ReduceMeanD");
    reduce_mean->AddInputDesc(const_tensor_desc);
    reduce_mean->AddOutputDesc(const_tensor_desc);
    auto reduce_mean_node = graph->AddNode(reduce_mean);
    ge::AttrUtils::SetInt(reduce_mean, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetListInt(reduce_mean, "axes", {1});
    ge::AttrUtils::SetBool(reduce_mean, "keep_dims", false);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), padd_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(padd_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), max_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(max_node->GetOutDataAnchor(0), reduce_mean_node->GetInDataAnchor(0));
  }
};


TEST_F(STEST_fusion_engine_heavy_format_distribution_fzg, heavy_format_distribution_fzg_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateFiveGraph(graph);

  HeavyFormatPropagationPtr heavy_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavy_format_propagator->Initialize();
  Status ret = heavy_format_propagator->PropagateHeavyFormat(*(graph.get()));
  for(auto node : graph->GetDirectNode()) {
    OpDescPtr opdesc = node->GetOpDesc();
    vector<int64_t> result_original_dim = {3, 4, 5, 6};
    vector<int64_t> result_dim = {30, 1, 16, 16};
    int64_t result_fe_group = 3;
    if (node->GetName() == "am1") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), result_fe_group);
    }
    if (node->GetName() == "am2") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), result_fe_group);

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetOutputDesc(0).GetFormat()), result_fe_group);
    }
    if (node->GetName() == "am3") {
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());

      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetOutputDesc(0).GetFormat()), result_fe_group);
    }
    if (node->GetType() == "Const") {
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Conv2D") {
      vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(1).GetFormat()));
      EXPECT_EQ(ge::FORMAT_NC1HWC0, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim5_h_d, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(result_dim, opdesc->GetInputDesc(1).GetShape().GetDims());
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_fusion_engine_heavy_format_distribution_fzg, heavy_format_distribution_fzg_02)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithL2LossAndMul(graph);

  HeavyFormatPropagationPtr heavy_format_propagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavy_format_propagator->Initialize();
  Status ret = heavy_format_propagator->PropagateHeavyFormat(*(graph.get()));
  vector<int64_t> result_dim = {3, 4, 5, 6};
  vector<int64_t> result_dim5_h_d = {3, 1, 5, 6, 16};
  vector<int64_t> result_dim_fz = {30, 1, 16, 16};
  vector<int64_t> scalar = {1};
  int64_t result_fe_group = 4;
  for(auto node : graph->GetDirectNode()) {
    OpDescPtr opdesc = node->GetOpDesc();
    if (node->GetName() == "am") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), result_fe_group);
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetOutputDesc(0).GetFormat()), result_fe_group);
    }
    if (node->GetType() == "Conv2DBackpropInput") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }

    if (node->GetType() == "L2Loss") {
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetInputDesc(0).GetFormat());
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(scalar, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
    if (node->GetType() == "Mul") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::FORMAT_ND, opdesc->GetInputDesc(1).GetFormat());
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetOutputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_dim_fz, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(scalar, opdesc->GetInputDesc(1).GetShape().GetDims());
      EXPECT_EQ(result_dim_fz, opdesc->GetOutputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), result_fe_group);
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetOutputDesc(0).GetFormat()), result_fe_group);
    }
  }

  EXPECT_EQ(fe::SUCCESS, ret);
}

// second call formatSelector: const share
TEST_F(STEST_fusion_engine_heavy_format_distribution_fzg, heavy_format_distribution_fzg_03) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateSecCallFormatSelectorGraph1(graph);
  FormatDtypeQuerierPtr format_dtype_querier_ptr = std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);
  FormatDtypeSetterPtr format_dtype_setter_ptr =
      std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  Status ret = format_dtype_setter_ptr->MultiThreadSetSupportFormatDtype(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);

  HeavyFormatPropagationPtr heavy_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavy_format_propagator->Initialize();
  ret = heavy_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);

  for (const auto &node : graph->GetDirectNode()) {
    OpDescPtr opdesc = node->GetOpDesc();
    vector<int64_t> result_original_dim = {32, 16, 1, 1};
    vector<int64_t> result_fz_dim = {2, 1, 16, 16};
    int64_t result_fe_group = 2;

    if (node->GetType() == "PadD") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), result_fe_group);
      EXPECT_EQ(result_fz_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }

    if (node->GetType() == "Maximum") {
      EXPECT_EQ(ge::FORMAT_FRACTAL_Z, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), 2);
      EXPECT_EQ(result_fz_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }

    if (node->GetType() == "ReduceMeanD") {
      EXPECT_EQ(ge::FORMAT_NCHW, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), 0);
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
    }

    if (node->GetType() == "Const") {
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}

// second call formatSelector
/*
TEST_F(STEST_fusion_engine_heavy_format_distribution_fzg, heavy_format_distribution_fzg_04) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateSecCallFormatSelectorGraph2(graph);
  FormatDtypeQuerierPtr format_dtype_querier_ptr = std::make_shared<FormatDtypeQuerier>(AI_CORE_NAME);
  FormatDtypeSetterPtr format_dtype_setter_ptr =
      std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  Status ret = format_dtype_setter_ptr->MultiThreadSetSupportFormatDtype(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);

  HeavyFormatPropagationPtr heavy_format_propagator =
      std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  heavy_format_propagator->Initialize();
  ret = heavy_format_propagator->PropagateHeavyFormat(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);

  for (const auto &node : graph->GetDirectNode()) {
    OpDescPtr opdesc = node->GetOpDesc();
    vector<int64_t> result_original_dim = {32, 16, 1, 1};
    vector<int64_t> result_fz_dim = {16, 2, 16, 16};
    int64_t result_fe_group = 4;

    if (node->GetType() == "PadD") {
      EXPECT_EQ(ge::FORMAT_NCHW, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), 0);
    }

    if (node->GetType() == "Maximum") {
      EXPECT_EQ(ge::FORMAT_NCHW, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), 0);
    }

    if (node->GetType() == "ReduceMeanD") {
      EXPECT_EQ(ge::FORMAT_NCHW, ge::GetPrimaryFormat(opdesc->GetInputDesc(0).GetFormat()));
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetInputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetInputDesc(0).GetShape().GetDims());
      EXPECT_EQ(ge::GetSubFormat(opdesc->GetInputDesc(0).GetFormat()), 0);
    }

    if (node->GetType() == "Const") {
      EXPECT_EQ(ge::FORMAT_NCHW, opdesc->GetOutputDesc(0).GetFormat());
      EXPECT_EQ(ge::DT_FLOAT16, opdesc->GetOutputDesc(0).GetDataType());
      EXPECT_EQ(result_original_dim, opdesc->GetOutputDesc(0).GetShape().GetDims());
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}*/

/* Test whether the sub-format will be updated. */
TEST_F(STEST_fusion_engine_heavy_format_distribution_fzg, test_format_updater) {
  HeavyFormatSupportFormatsUpdater updater(nullptr, nullptr);
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("test", "Test");
  GeTensorDesc tensor_in(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_in.SetOriginShape(GeShape({3, 4, 5, 6}));
  tensor_in.SetOriginFormat(ge::FORMAT_NCHW);
  GeTensorDesc tensor_out(GeShape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensor_out.SetOriginShape(GeShape({1}));
  tensor_out.SetOriginFormat(ge::FORMAT_NCHW);
  op_desc_ptr->AddInputDesc(tensor_in);
  op_desc_ptr->AddOutputDesc(tensor_out);
  HeavyFormatInfo heavy_format_info;
  heavy_format_info.sub_format = 160;

  auto in_sub_format = GetSubFormat(op_desc_ptr->MutableInputDesc(0)->GetFormat());
  EXPECT_EQ(in_sub_format, 0);

  auto out_sub_format = GetSubFormat(op_desc_ptr->MutableOutputDesc(0)->GetFormat());
  EXPECT_EQ(out_sub_format, 0);

  updater.UpdateSubFormatForTensors(op_desc_ptr, heavy_format_info);

  in_sub_format = GetSubFormat(op_desc_ptr->MutableInputDesc(0)->GetFormat());
  EXPECT_EQ(in_sub_format, 160);

  out_sub_format = GetSubFormat(op_desc_ptr->MutableOutputDesc(0)->GetFormat());
  EXPECT_EQ(out_sub_format, 160);
}
