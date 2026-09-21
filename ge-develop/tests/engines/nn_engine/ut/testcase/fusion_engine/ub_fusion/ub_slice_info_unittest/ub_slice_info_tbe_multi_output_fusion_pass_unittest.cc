/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <map>
#include <memory>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "proto/om.pb.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "common/configuration.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/ub_fusion/buffer_fusion_pass_runner.h"
#include "ub_pass_slice_info/ub_pass_slice_info_manager.h"
#include "../fusion_stub.hpp"
#undef protected
#undef private
using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;

using OpSetterPtr = std::shared_ptr<OpSetter>;
class TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST : public testing::Test {
public:
  using AttrDefMap = ::google::protobuf::Map<::std::string, AttrDef>;

protected:
  static void SetUpTestCase() { std::cout << "UB fusion SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "UB fusion TearDown" << std::endl; }

  void SetUp()
  {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo tbe_custom {
            6,
            "tbe-custom",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_slice_op_info/slice_success",
            ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    fe_ops_kernel_info_store_ptr_->Initialize(options);
  }

  virtual void TearDown() {}

  void SetPattern(ge::OpDescPtr opdef, string optype) {
    auto key_pattern = "_pattern";
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
  }

  void SetTvmType(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE,static_cast<int64_t>(domi::ImplyType::TVM));
  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;

  /************************
   *    ElemWise_1-->ElemWise_2-->ElemWise_3-->ElemWise_4
   *                                 |
   *                                 V
   *                             ElemWise_5
   *************************/
  // {"_fusion_op_slice_info":{"l1FusionEnable":1,"minTbeL1Space":0,"reduceMaps":[],"splitMaps":[
  //   {"inputList":[{"axis":[3],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[3],"idx":0},{"axis":[3],"idx":1}]},
  //   {"inputList":[{"axis":[2],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[2],"idx":0},{"axis":[2],"idx":1}]},
  //   {"inputList":[{"axis":[1],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[1],"idx":0},{"axis":[1],"idx":1}]},
  //   {"inputList":[{"axis":[0],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[0],"idx":0},{"axis":[0],"idx":1}]}]}}
  void BuildGraphForTbeMultiOutputFusionPassOKCase1(ComputeGraphPtr &graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr eltwise1 = std::make_shared<OpDesc>("eltwise1", "Eltwise");
    OpDescPtr eltwise2 = std::make_shared<OpDesc>("eltwise2", "Eltwise");
    OpDescPtr eltwise3 = std::make_shared<OpDesc>("eltwise3", "Eltwise");
    OpDescPtr eltwise4 = std::make_shared<OpDesc>("eltwise4", "Eltwise");
    OpDescPtr eltwise5 = std::make_shared<OpDesc>("eltwise5", "Eltwise");
    OpDescPtr end1 = std::make_shared<OpDesc>("end1", "End");
    OpDescPtr end2 = std::make_shared<OpDesc>("end2", "End");
    SetPattern(eltwise1, "ElemWise");
    SetPattern(eltwise2, "ElemWise");
    SetPattern(eltwise3, "ElemWise");
    SetPattern(eltwise4, "ElemWise");
    SetPattern(eltwise5, "ElemWise");
    SetTvmType(eltwise1);
    SetTvmType(eltwise2);
    SetTvmType(eltwise3);
    SetTvmType(eltwise4);
    SetTvmType(eltwise5);
    ge::AttrUtils::SetInt(eltwise1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise3, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise4, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise5, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    // add descriptor
    GeTensorDesc tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0,ge::DT_FLOAT16);
    tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    data->AddOutputDesc(tensor_desc);
    eltwise1->AddInputDesc(tensor_desc);
    eltwise1->AddOutputDesc(tensor_desc);
    eltwise2->AddInputDesc(tensor_desc);
    eltwise2->AddOutputDesc(tensor_desc);
    eltwise3->AddInputDesc(tensor_desc);
    eltwise3->AddOutputDesc(tensor_desc);
    eltwise4->AddInputDesc(tensor_desc);
    eltwise4->AddOutputDesc(tensor_desc);
    eltwise5->AddInputDesc(tensor_desc);
    eltwise5->AddOutputDesc(tensor_desc);
    end1->AddInputDesc(tensor_desc);
    end1->AddOutputDesc(tensor_desc);
    end2->AddInputDesc(tensor_desc);
    end2->AddOutputDesc(tensor_desc);
    NodePtr data_node = graph->AddNode(data);
    NodePtr eltwise1_node = graph->AddNode(eltwise1);
    NodePtr eltwise2_node = graph->AddNode(eltwise2);
    NodePtr eltwise3_node = graph->AddNode(eltwise3);
    NodePtr eltwise4_node = graph->AddNode(eltwise4);
    NodePtr eltwise5_node = graph->AddNode(eltwise5);
    NodePtr end1_node = graph->AddNode(end1);
    NodePtr end2_node = graph->AddNode(end2);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(eltwise1_node->GetName(), std::move(buffer));
    eltwise1_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), eltwise1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise1_node->GetOutDataAnchor(0), eltwise2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise2_node->GetOutDataAnchor(0), eltwise3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), eltwise4_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), eltwise5_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise4_node->GetOutDataAnchor(0), end1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise5_node->GetOutDataAnchor(0), end2_node->GetInDataAnchor(0));
  }

  /************************
   *   ElemWise_3-->ElemWise_4
   *      |
   *      V
   *  ElemWise_5
   *************************/
  // {"_fusion_op_slice_info":{"l1FusionEnable":1,"minTbeL1Space":0,"reduceMaps":[],"splitMaps":[
  //   {"inputList":[{"axis":[3],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[3],"idx":0},{"axis":[3],"idx":1}]},
  //   {"inputList":[{"axis":[2],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[2],"idx":0},{"axis":[2],"idx":1}]},
  //   {"inputList":[{"axis":[1],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[1],"idx":0},{"axis":[1],"idx":1}]},
  //   {"inputList":[{"axis":[0],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[0],"idx":0},{"axis":[0],"idx":1}]}]}}
  void BuildGraphForTbeMultiOutputFusionPassOKCase2(ComputeGraphPtr &graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr eltwise3 = std::make_shared<OpDesc>("eltwise3", "Eltwise");
    OpDescPtr eltwise4 = std::make_shared<OpDesc>("eltwise4", "Eltwise");
    OpDescPtr eltwise5 = std::make_shared<OpDesc>("eltwise5", "Eltwise");
    OpDescPtr end1 = std::make_shared<OpDesc>("end1", "End");
    OpDescPtr end2 = std::make_shared<OpDesc>("end2", "End");
    SetPattern(eltwise3, "ElemWise");
    SetPattern(eltwise4, "ElemWise");
    SetPattern(eltwise5, "ElemWise");
    SetTvmType(eltwise3);
    SetTvmType(eltwise4);
    SetTvmType(eltwise5);
    ge::AttrUtils::SetInt(eltwise3, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise4, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise5, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    // add descriptor
    GeTensorDesc tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0,ge::DT_FLOAT16);
    tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    data->AddOutputDesc(tensor_desc);
    eltwise3->AddInputDesc(tensor_desc);
    eltwise3->AddOutputDesc(tensor_desc);
    eltwise4->AddInputDesc(tensor_desc);
    eltwise4->AddOutputDesc(tensor_desc);
    eltwise5->AddInputDesc(tensor_desc);
    eltwise5->AddOutputDesc(tensor_desc);
    end1->AddInputDesc(tensor_desc);
    end1->AddOutputDesc(tensor_desc);
    end2->AddInputDesc(tensor_desc);
    end2->AddOutputDesc(tensor_desc);
    NodePtr data_node = graph->AddNode(data);;
    NodePtr eltwise3_node = graph->AddNode(eltwise3);
    NodePtr eltwise4_node = graph->AddNode(eltwise4);
    NodePtr eltwise5_node = graph->AddNode(eltwise5);
    NodePtr end1_node = graph->AddNode(end1);
    NodePtr end2_node = graph->AddNode(end2);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(eltwise3_node->GetName(), std::move(buffer));
    eltwise3_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), eltwise3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), eltwise4_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), eltwise5_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise4_node->GetOutDataAnchor(0), end1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise5_node->GetOutDataAnchor(0), end2_node->GetInDataAnchor(0));
  }

  /************************
   *    (ElemWise_1)-->(ElemWise_2)-->ElemWise_3-->ElemWise_4
   *                                      |
   *                                      V
   *                                 OtherOutput
   *************************/
  //  {"_fusion_op_slice_info":{"l1FusionEnable":1,"minTbeL1Space":0,"reduceMaps":[],"splitMaps":[
  //    {"inputList":[{"axis":[3],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[3],"idx":0},{"axis":[3],"idx":1}]},
  //    {"inputList":[{"axis":[2],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[2],"idx":0},{"axis":[2],"idx":1}]},
  //    {"inputList":[{"axis":[1],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[1],"idx":0},{"axis":[1],"idx":1}]},
  //    {"inputList":[{"axis":[0],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[0],"idx":0},{"axis":[0],"idx":1}]}]}}
   void BuildGraphForTbeMultiOutputFusionPassOKCase3(ComputeGraphPtr &graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr eltwise1 = std::make_shared<OpDesc>("eltwise1", "Eltwise");
    OpDescPtr eltwise2 = std::make_shared<OpDesc>("eltwise2", "Eltwise");
    OpDescPtr eltwise3 = std::make_shared<OpDesc>("eltwise3", "Eltwise");
    OpDescPtr eltwise4 = std::make_shared<OpDesc>("eltwise4", "Eltwise");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr end1 = std::make_shared<OpDesc>("end1", "End");
    OpDescPtr end2 = std::make_shared<OpDesc>("end2", "End");
    SetPattern(eltwise1, "ElemWise");
    SetPattern(eltwise2, "ElemWise");
    SetPattern(eltwise3, "ElemWise");
    SetPattern(eltwise4, "ElemWise");
    SetPattern(conv, "Convolution");
    SetTvmType(eltwise1);
    SetTvmType(eltwise2);
    SetTvmType(eltwise3);
    SetTvmType(eltwise4);
    SetTvmType(conv);
    ge::AttrUtils::SetInt(eltwise1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise2, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise3, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise4, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(conv, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    // add descriptor
    GeTensorDesc tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0,ge::DT_FLOAT16);
    tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_bias(GeShape({512}), ge::FORMAT_NCHW, ge::DT_INT32);
    conv_tensor_desc_bias.SetOriginShape(GeShape({512}));
    conv_tensor_desc_bias.SetOriginFormat(ge::FORMAT_NCHW);
    data->AddOutputDesc(tensor_desc);
    eltwise1->AddInputDesc(tensor_desc);
    eltwise1->AddOutputDesc(tensor_desc);
    eltwise2->AddInputDesc(tensor_desc);
    eltwise2->AddOutputDesc(tensor_desc);
    eltwise3->AddInputDesc(tensor_desc);
    eltwise3->AddOutputDesc(tensor_desc);
    eltwise4->AddInputDesc(tensor_desc);
    eltwise4->AddOutputDesc(tensor_desc);
    conv->AddInputDesc(tensor_desc);
    conv->AddInputDesc(conv_tensor_desc_weight);
    conv->AddInputDesc(conv_tensor_desc_bias);
    conv->AddOutputDesc(tensor_desc);
    end1->AddInputDesc(tensor_desc);
    end1->AddOutputDesc(tensor_desc);
    end2->AddInputDesc(tensor_desc);
    end2->AddOutputDesc(tensor_desc);
    string conv_op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [0], \"tailOverLap\": [0]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [0], \"tailOverLap\": [0]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}, {\"inputList\": [{\"idx\": 1, \"axis\": [1], \"headOverLap\": [-1], \"tailOverLap\": [-1]}, {\"idx\": 2, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [1]}]}], \"reduceMaps\": [], \"l1FusionEnable\": 2, \"minTbeL1Space\": 0}}";
    AttrUtils::SetStr(conv, OP_SLICE_INFO, conv_op_slice_info);

    NodePtr data_node = graph->AddNode(data);
    NodePtr eltwise1_node = graph->AddNode(eltwise1);
    NodePtr eltwise2_node = graph->AddNode(eltwise2);
    NodePtr eltwise3_node = graph->AddNode(eltwise3);
    NodePtr eltwise4_node = graph->AddNode(eltwise4);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr end1_node = graph->AddNode(end1);
    NodePtr end2_node = graph->AddNode(end2);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(eltwise1_node->GetName(), std::move(buffer));
    eltwise1_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), eltwise1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise1_node->GetOutDataAnchor(0), eltwise2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise2_node->GetOutDataAnchor(0), eltwise3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), eltwise4_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise4_node->GetOutDataAnchor(0), end1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), end2_node->GetInDataAnchor(0));
  }

  /************************
  *                     ElemWise_3-->ElemWise_4
  *                                      |
  *                                      V
  *                                 OtherOutput
  *************************/
  // {"_fusion_op_slice_info":{"l1FusionEnable":1,"minTbeL1Space":0,"reduceMaps":[],"splitMaps":[
  //   {"inputList":[{"axis":[3],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[3],"idx":0},{"axis":[3],"idx":1}]},
  //   {"inputList":[{"axis":[2],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[2],"idx":0},{"axis":[2],"idx":1}]},
  //   {"inputList":[{"axis":[1],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[1],"idx":0},{"axis":[1],"idx":1}]},
  //   {"inputList":[{"axis":[0],"headOverLap":[],"idx":0,"tailOverLap":[]}],"outputList":[{"axis":[0],"idx":0},{"axis":[0],"idx":1}]}]}}
  void BuildGraphForTbeMultiOutputFusionPassOKCase4(ComputeGraphPtr &graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr eltwise3 = std::make_shared<OpDesc>("eltwise3", "Eltwise");
    OpDescPtr eltwise4 = std::make_shared<OpDesc>("eltwise4", "Eltwise");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr end1 = std::make_shared<OpDesc>("end1", "End");
    OpDescPtr end2 = std::make_shared<OpDesc>("end2", "End");
    SetPattern(eltwise3, "ElemWise");
    SetPattern(eltwise4, "ElemWise");
    SetPattern(conv, "Convolution");
    SetTvmType(eltwise3);
    SetTvmType(eltwise4);
    SetTvmType(conv);
    ge::AttrUtils::SetInt(eltwise3, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise4, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(conv, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    // add descriptor
    GeTensorDesc tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0,ge::DT_FLOAT16);
    tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_bias(GeShape({512}), ge::FORMAT_NCHW, ge::DT_INT32);
    conv_tensor_desc_bias.SetOriginShape(GeShape({512}));
    conv_tensor_desc_bias.SetOriginFormat(ge::FORMAT_NCHW);

    data->AddOutputDesc(tensor_desc);
    eltwise3->AddInputDesc(tensor_desc);
    eltwise3->AddOutputDesc(tensor_desc);
    eltwise4->AddInputDesc(tensor_desc);
    eltwise4->AddOutputDesc(tensor_desc);
    conv->AddInputDesc(tensor_desc);
    conv->AddInputDesc(conv_tensor_desc_weight);
    conv->AddInputDesc(conv_tensor_desc_bias);
    conv->AddOutputDesc(tensor_desc);
    end1->AddInputDesc(tensor_desc);
    end1->AddOutputDesc(tensor_desc);
    end2->AddInputDesc(tensor_desc);
    end2->AddOutputDesc(tensor_desc);
    string conv_op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [0], \"tailOverLap\": [0]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [0], \"tailOverLap\": [0]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}, {\"inputList\": [{\"idx\": 1, \"axis\": [1], \"headOverLap\": [-1], \"tailOverLap\": [-1]}, {\"idx\": 2, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [1]}]}], \"reduceMaps\": [], \"l1FusionEnable\": 2, \"minTbeL1Space\": 0}}";
    AttrUtils::SetStr(conv, OP_SLICE_INFO, conv_op_slice_info);

    NodePtr data_node = graph->AddNode(data);;
    NodePtr eltwise3_node = graph->AddNode(eltwise3);
    NodePtr eltwise4_node = graph->AddNode(eltwise4);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr end1_node = graph->AddNode(end1);
    NodePtr end2_node = graph->AddNode(end2);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));

    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(eltwise3_node->GetName(), std::move(buffer));
    eltwise3_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), eltwise3_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), eltwise4_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise3_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise4_node->GetOutDataAnchor(0), end1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), end2_node->GetInDataAnchor(0));
  }
};

TEST_F(TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST, pass_case1) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraphForTbeMultiOutputFusionPassOKCase1(graph_out);
  graph_out->TopologicalSorting();
  graph_out->Dump();
  // set op slice info for each op
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph_out.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
          std::make_shared<FusionPriorityManager>("engineName", nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
          std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);
  uint32_t id = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case1 UB fusion before" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr << "id:" << id << endl;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    id++;
  }
  sub_graph_optimizer_ptr->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*graph_out);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph_out);

  string fusion_op_slice_info;
  string op_slice_info;
  int find = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case1 UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr  << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (node->GetOpDesc()->GetType() == "Eltwise" && node->GetOpDesc()->GetName() == "eltwise1eltwise2eltwise3eltwise5eltwise4") {
      AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
      cerr << "op slice info is :   " << endl << op_slice_info << endl;
      AttrUtils::GetStr(node->GetOpDesc(), FUSION_OP_SLICE_INFO, fusion_op_slice_info);
      cerr << "fusion op slice info is :   " << endl << fusion_op_slice_info << endl;
      find = 1;
    }
  }
  EXPECT_EQ(find, 1);
}

TEST_F(TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST, pass_case2) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraphForTbeMultiOutputFusionPassOKCase2(graph_out);
  graph_out->TopologicalSorting();
  graph_out->Dump();
  // set op slice info for each op
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph_out.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
          std::make_shared<FusionPriorityManager>("engineName", nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
          std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);
  uint32_t id = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case2 UB fusion before" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr << "id:" << id << endl;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    id++;
  }
  sub_graph_optimizer_ptr->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*graph_out);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph_out);

  string fusion_op_slice_info;
  string op_slice_info;
  int find = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case2 UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr  << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
    cerr << "op slice info is :   " << endl << op_slice_info << endl;
    AttrUtils::GetStr(node->GetOpDesc(), FUSION_OP_SLICE_INFO, fusion_op_slice_info);
    cerr << "fusion op slice info is :   " << endl << fusion_op_slice_info << endl;
    if (node->GetOpDesc()->GetType() == "Eltwise" && node->GetName() == "eltwise3eltwise5eltwise4") {
      find = 1;
    }
  }
  EXPECT_EQ(find, 1);
}

TEST_F(TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST, pass_case3) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraphForTbeMultiOutputFusionPassOKCase3(graph_out);
  graph_out->TopologicalSorting();
  graph_out->Dump();
  // set op slice info for each op
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph_out.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
          std::make_shared<FusionPriorityManager>("engineName", nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
          std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);
  uint32_t id = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case3 UB fusion before" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr << "id:" << id << endl;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    id++;
  }
  sub_graph_optimizer_ptr->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*graph_out);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph_out);

  string fusion_op_slice_info;
  string op_slice_info;
  int find = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case3 UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr  << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (node->GetOpDesc()->GetType() == "Eltwise" && node->GetName() == "eltwise1eltwise2eltwise3eltwise4") {
      AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
      cerr << "op slice info is :   " << endl << op_slice_info << endl;
      AttrUtils::GetStr(node->GetOpDesc(), FUSION_OP_SLICE_INFO, fusion_op_slice_info);
      cerr << "fusion op slice info is :   " << endl << fusion_op_slice_info << endl;
      find = 1;
    }
  }
  EXPECT_EQ(find, 1);
}

TEST_F(TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST, pass_case4) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraphForTbeMultiOutputFusionPassOKCase4(graph_out);
  graph_out->TopologicalSorting();
  graph_out->Dump();
  // set op slice info for each op
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph_out.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
          std::make_shared<FusionPriorityManager>("engineName", nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
          std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);
  uint32_t id = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case4 UB fusion before" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr << "id:" << id << endl;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    id++;
  }
  sub_graph_optimizer_ptr->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*graph_out);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph_out);

  string fusion_op_slice_info;
  string op_slice_info;
  int find = 0;
  cerr << endl;
  cerr << "TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST::pass_case4 UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr  << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (node->GetOpDesc()->GetType() == "Eltwise" && node->GetName() == "eltwise3eltwise4") {
      AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
      cerr << "op slice info is :   " << endl << op_slice_info << endl;
      AttrUtils::GetStr(node->GetOpDesc(), FUSION_OP_SLICE_INFO, fusion_op_slice_info);
      cerr << "fusion op slice info is :   " << endl << fusion_op_slice_info << endl;
      find = 1;
    }
  }
  EXPECT_EQ(find, 1);
}

TEST_F(TBE_MULTI_OUTPUT_FUSION_SLICE_INFO_UNITTEST, pass_case5) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
  OpDescPtr fixpipeops1 = std::make_shared<OpDesc>("fixpipe1", "FixPipe");
  OpDescPtr fixpipeops2 = std::make_shared<OpDesc>("fixpipe2", "FixPipe");
  OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
  OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", fe::DATA);
  OpDescPtr data3 = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr data4 = std::make_shared<OpDesc>("DATA4", fe::DATA);

  ge::GeTensorDesc tensor1(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  ge::GeTensorDesc tensor2(GeShape({3, 4, 5, 6}), ge::FORMAT_NHWC, ge::DT_FLOAT);

  ge::GeTensorDesc data1_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  data1_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
  data1_weight.SetOriginFormat(ge::FORMAT_NC1HWC0);
  ge::GeTensorDesc data2_weight(GeShape(), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  data2_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
  data2_weight.SetOriginFormat(ge::FORMAT_NC1HWC0);
  ge::GeTensorDesc data3_weight(GeShape(), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  data3_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
  data3_weight.SetOriginFormat(ge::FORMAT_NC1HWC0);
  ge::GeTensorDesc data4_weight(GeShape(), ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  data4_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
  data4_weight.SetOriginFormat(ge::FORMAT_NC1HWC0);

  data1->AddOutputDesc(data1_weight);
  data2->AddOutputDesc(data2_weight);
  data3->AddOutputDesc(data3_weight);
  data4->AddOutputDesc(data4_weight);

  tensor1.SetOriginShape(GeShape({3, 4, 5, 6}));
  tensor1.SetOriginFormat(ge::FORMAT_NHWC);
  tensor2.SetOriginShape(GeShape({3, 4, 5, 6}));
  tensor2.SetOriginFormat(ge::FORMAT_NHWC);
  conv1->AddInputDesc("x1", tensor1);
  conv1->AddInputDesc("x2", tensor1);
  conv1->AddOutputDesc("y", tensor1);
  fixpipeops1->AddInputDesc("x1", tensor1);
  fixpipeops1->AddInputDesc("x2", tensor1);
  fixpipeops1->AddOutputDesc("y", tensor1);
  fixpipeops2->AddInputDesc("x1", tensor1);
  fixpipeops2->AddInputDesc("x2", tensor1);
  fixpipeops2->AddOutputDesc("y", tensor2);
  SetPattern(conv1, "Convolution");
  SetPattern(fixpipeops1, "fixpipe");
  SetPattern(fixpipeops2, "fixpipe");  
  NodePtr conv_node1 = graph_out->AddNode(conv1);
  NodePtr fixpipenode1 = graph_out->AddNode(fixpipeops1);
  NodePtr fixpipenode2 = graph_out->AddNode(fixpipeops2);
  NodePtr data_node1 = graph_out->AddNode(data1);
  NodePtr data_node2 = graph_out->AddNode(data2);
  NodePtr data_node3 = graph_out->AddNode(data3);
  NodePtr data_node4 = graph_out->AddNode(data4);
  string conv_op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}]}}";
  string fixpipe1_op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}]}}";
  string fixpipe2_op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [1]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}]}}";
  AttrUtils::SetStr(conv1, OP_SLICE_INFO, conv_op_slice_info);
  AttrUtils::SetStr(fixpipeops1, OP_SLICE_INFO, fixpipe1_op_slice_info);
  AttrUtils::SetStr(fixpipeops2, OP_SLICE_INFO, fixpipe2_op_slice_info);

  GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), conv_node1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), conv_node1->GetInDataAnchor(1));
  GraphUtils::AddEdge(data_node3->GetOutDataAnchor(0), fixpipenode1->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node1->GetOutDataAnchor(0),  fixpipenode1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data_node4->GetOutDataAnchor(0),  fixpipenode2->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node1->GetOutDataAnchor(0),  fixpipenode2->GetInDataAnchor(0));

  graph_out->TopologicalSorting();
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::vector<ge::NodePtr> fusion_nodes = {conv_node1, fixpipenode1, fixpipenode2};
  OpCalcInfo op_calc_info;
  EXPECT_EQ(UbPassSliceInfoManager::CalcSliceInfoForFusionOp(fusion_nodes, op_calc_info), fe::FAILED);
  string op_slice_info_str;
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  std::cout << "fusion op slice info is : " << std::endl << op_slice_info_str << std::endl;
}
