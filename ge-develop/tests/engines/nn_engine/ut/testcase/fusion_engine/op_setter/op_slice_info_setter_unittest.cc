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
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "common/op_slice_util.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

using namespace std;
using namespace ge;
using namespace fe;
using OpSetterPtr = std::shared_ptr<OpSetter>;


class UTEST_OP_SLICE_INFO_SETTER : public testing::Test
{
protected:
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
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

  void TearDown() {}

static void CreateOneOpGraph(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  pack_op->MutableInputDesc(0)->SetFormat(ge::FORMAT_FRACTAL_NZ);
  pack_op->MutableOutputDesc(0)->SetFormat(ge::FORMAT_FRACTAL_NZ);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pack_op1 = std::make_shared<OpDesc>("pack1", "Pack");
  pack_op1->AddInputDesc("x", out_desc);
  pack_op1->AddOutputDesc("y", out_desc);
  pack_op1->MutableInputDesc(0)->SetFormat(ge::FORMAT_NC1HWC0);
  pack_op1->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NC1HWC0);
  ge::AttrUtils::SetInt(pack_op1, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node1 = graph->AddNode(pack_op1);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}

static void CreateOneOpGraphElem(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  relu_op->AddOutputDesc("yy", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}

static void CreateOneOpGraphElem1(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddOutputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphElem2(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphElem3(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NHWC);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphElem4(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  vector<int64_t> dim1 = {2, 12, 2};
  GeShape shape1(dim1);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  GeTensorDesc in_desc(shape1);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}

static void CreateOneOpGraphElem5(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NC1HWC0);
  out_desc.SetFormat(FORMAT_NC1HWC0);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphElem6(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_FRACTAL_Z);
  out_desc.SetFormat(FORMAT_FRACTAL_Z);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphElem7(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  vector<int64_t> dim1;
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape1);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddInputDesc("xx", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphElem8(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  vector<int64_t> dim1(4, 1);
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}

static void CreateOneOpGraphElem9(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_ND);
  in_desc.SetFormat(FORMAT_ND);
  in_desc.SetDataType(DT_FLOAT16);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);
}

static void CreateOneOpGraphOther(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Other");

  // add descriptor
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  vector<int64_t> dim1(4, 1);
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);
}
static void CreateOneOpGraphOther1(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Segment");

  // add descriptor
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  vector<int64_t> dim1(4, 1);
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);
}
static void CreateOneOpGraphOther2(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Resize");

  // add descriptor
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  vector<int64_t> dim1(4, 1);
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);
}
static void CreateOneOpGraphOther3(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Scatter");

  // add descriptor
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  vector<int64_t> dim1(4, 1);
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", in_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);
}

static void CreateOneOpGraphReduce(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphReduce1(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  prelu_op->AddOutputDesc("yy", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphReduce2(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddInputDesc("xx", out_desc);
  prelu_op->AddInputDesc("xxx", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  vector<int32_t> axes = {1};
  ge::AttrUtils::SetListInt(prelu_op, "axes", axes);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphReduce3(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  GeTensorDesc in_desc(shape);
  in_desc.SetOriginFormat(FORMAT_NHWC);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", in_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
static void CreateOneOpGraphReduce4(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);
  vector<int64_t> dim1 = {2, 12, 8};
  GeShape shape1(dim1);
  GeTensorDesc in_desc(shape1);
  in_desc.SetOriginFormat(FORMAT_NCHW);
  in_desc.SetFormat(FORMAT_NCHW);
  in_desc.SetDataType(DT_FLOAT16);
  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", in_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}

static void CreateOneOpGraphreturn(ComputeGraphPtr graph) {
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  // add descriptor
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetOriginFormat(FORMAT_NCHW);
  out_desc.SetFormat(FORMAT_NCHW);
  out_desc.SetDataType(DT_FLOAT16);
  relu_op->AddInputDesc("x", out_desc);
  relu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");
  NodePtr relu_node = graph->AddNode(relu_op);

  OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
  conv_op->AddInputDesc("x", out_desc);
  conv_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");

  ge::AttrUtils::SetInt(conv_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr conv_node = graph->AddNode(conv_op);

  OpDescPtr relu6_op = std::make_shared<OpDesc>("relu6", "Relu6");
  relu6_op->AddInputDesc("x", out_desc);
  relu6_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");
  ge::AttrUtils::SetInt(relu6_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr relu6_node = graph->AddNode(relu6_op);

  OpDescPtr convback_op = std::make_shared<OpDesc>("convback", "Conv2DBackpropInput");
  convback_op->AddInputDesc("x", out_desc);
  convback_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");
  ge::AttrUtils::SetInt(convback_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr convback_node = graph->AddNode(convback_op);

  OpDescPtr prelu_op = std::make_shared<OpDesc>("prelu", "PRelu");
  prelu_op->AddInputDesc("x", out_desc);
  prelu_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");
  ge::AttrUtils::SetInt(prelu_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr prelu_node = graph->AddNode(prelu_op);

  OpDescPtr pack_op = std::make_shared<OpDesc>("pack", "Pack");
  pack_op->AddInputDesc("x", out_desc);
  pack_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");
  ge::AttrUtils::SetInt(pack_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pack_node = graph->AddNode(pack_op);

  OpDescPtr pad_op = std::make_shared<OpDesc>("padd", "PadD");
  pad_op->AddInputDesc("x", out_desc);
  pad_op->AddOutputDesc("y", out_desc);
  ge::AttrUtils::SetStr(relu_op, "_op_slice_info", "abc");
  ge::AttrUtils::SetInt(pad_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
  NodePtr pad_node = graph->AddNode(pad_op);
}
};


TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateOneOpGraphElem(graph);

  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_1) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem1(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_2) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem2(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_3) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem3(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_4) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem4(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_5) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem5(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_6) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem6(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_7) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem7(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_8) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphElem8(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}
TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elem_9) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateOneOpGraphElem9(graph);

  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_other) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphOther(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_other_1) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphOther1(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_other_2) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphOther2(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_other_3) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphOther3(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_reduce) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphReduce(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_reduce1) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphReduce1(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_reduce2) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphReduce2(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_reduce3) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphReduce3(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_reduce4) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphReduce4(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_return_success) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraphreturn(graph);

OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_return_success1) {
ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
CreateOneOpGraph(graph);
OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateOneOpGraphreturn(graph);

  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  OpCalcInfo opCalcInfo;
  opCalcInfo.Initialize();
  std::string opCalcInfoStr;
  AxisReduceMap axisReduceMap;
  axisReduceMap.Initialize();
  InputReduceInfo inputReduceInfo;
  inputReduceInfo.Initialize();
  OutputReduceInfo outputReduceInfo;
  outputReduceInfo.Initialize();
  inputReduceInfo.SetIndex(0);
  std::vector<int64_t> axes = {0,1,2};
  inputReduceInfo.SetAxis(axes);
  axisReduceMap.AddInputReduceInfo(inputReduceInfo);
  outputReduceInfo.SetIndex(0);
  outputReduceInfo.SetIsAtomic(false);
  outputReduceInfo.SetReduceType(REDUCE_MEAN);
  axisReduceMap.AddOutputReduceInfo(outputReduceInfo);
  opCalcInfo.AddAxisReduceMap(axisReduceMap);
  SetOpSliceInfoToJson(opCalcInfo, opCalcInfoStr);
  GetOpSliceInfoFromJson(opCalcInfo, opCalcInfoStr);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddOutputDesc("y", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim = {1, 3, -1, 3};
  GeShape dynamic_shape(dynamic_dim);
  tensor_desc.SetOriginShape(dynamic_shape);
  tensor_desc.SetShape(dynamic_shape);
  op_desc_ptr->UpdateInputDesc("x", tensor_desc);
  op_desc_ptr->UpdateOutputDesc("y", tensor_desc);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_EQ(op_slice_info, op_slice_info_1);

  vector<int64_t> dynamic_dim_2 = {-2};
  GeShape dynamic_shape_2(dynamic_dim_2);
  tensor_desc.SetOriginShape(dynamic_shape_2);
  tensor_desc.SetShape(dynamic_shape_2);
  op_desc_ptr->UpdateInputDesc("x", tensor_desc);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_2;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_2);
  EXPECT_EQ(op_slice_info, op_slice_info_2);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim = {1, 3, -1, 5};
  GeShape dynamic_shape(dynamic_dim);
  tensor_desc.SetOriginShape(dynamic_shape);
  tensor_desc.SetShape(dynamic_shape);
  op_desc_ptr->UpdateInputDesc("x", tensor_desc);
  op_desc_ptr->UpdateInputDesc("y", tensor_desc);
  op_desc_ptr->UpdateOutputDesc("z", tensor_desc);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_NE(op_slice_info, op_slice_info_1);

  vector<int64_t> dynamic_dim_1 = {1, 3, 4, 5};
  GeShape dynamic_shape_1(dynamic_dim_1);
  vector<int64_t> dynamic_dim_2 = {-2};
  GeShape dynamic_shape_2(dynamic_dim_2);
  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_1);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape_2);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim_1 = {1, 3, -1, 5};
  GeShape dynamic_shape_1(dynamic_dim_1);
  vector<int64_t> dynamic_dim_2 = {1, -1, 3, 5};
  GeShape dynamic_shape_2(dynamic_dim_2);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_1);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape_2);
  op_desc_ptr->MutableOutputDesc("z")->SetShape(dynamic_shape_2);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_EQ(op_slice_info, op_slice_info_1);

  dynamic_dim_1 = {-2};
  GeShape dynamic_shape_3(dynamic_dim_1);
  dynamic_dim_2 = {1, -1, 3, 5};
  GeShape dynamic_shape_4(dynamic_dim_2);
  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_3);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape_4);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_2;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_2);
  EXPECT_NE(op_slice_info, op_slice_info_2);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_4) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("reduce_sum_d", "ReduceSumD");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);

  std::vector<int64_t> axes_vec = {1};
  (void)ge::AttrUtils::SetListInt(op_desc_ptr, "axes", axes_vec);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, SLICE_PATTERN_REDUCE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim_1 = {1, 3, -1, 5};
  GeShape dynamic_shape_1(dynamic_dim_1);
  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_1);
  op_desc_ptr->MutableOutputDesc("z")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLICE_PATTERN_REDUCE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_NE(op_slice_info, op_slice_info_1);

  vector<int64_t> dynamic_dim_2 = {1, -1, 3, 5};
  GeShape dynamic_shape_2(dynamic_dim_2);
  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_2);
  op_desc_ptr->MutableOutputDesc("z")->SetShape(dynamic_shape_2);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLICE_PATTERN_REDUCE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_2;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_2);
  EXPECT_EQ(op_slice_info, op_slice_info_2);

  vector<int64_t> dynamic_dim_3 = {1, 2, 3, -1};
  GeShape dynamic_shape_3(dynamic_dim_3);
  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_3);
  op_desc_ptr->MutableOutputDesc("z")->SetShape(dynamic_shape_3);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLICE_PATTERN_REDUCE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_3;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_3);
  EXPECT_NE(op_slice_info, op_slice_info_3);

  vector<int64_t> dynamic_dim_4 = {-2};
  GeShape dynamic_shape_4(dynamic_dim_4);
  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_4);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLICE_PATTERN_REDUCE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_4;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_4);
  EXPECT_NE(op_slice_info, op_slice_info_4);

  op_desc_ptr->MutableOutputDesc("z")->SetShape(shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLICE_PATTERN_REDUCE, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_5;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_5);
  EXPECT_NE(op_slice_info, op_slice_info_5);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_5) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pooling", "Pooling");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc);
  op_desc_ptr->AddInputDesc("z", tensor_desc);
  op_desc_ptr->AddOutputDesc("out", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim = {1, 3, -1, 5};
  GeShape dynamic_shape(dynamic_dim);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape);
  op_desc_ptr->MutableInputDesc("z")->SetShape(dynamic_shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_EQ(op_slice_info, op_slice_info_1);

  vector<int64_t> dynamic_dim_1 = {-2};
  GeShape dynamic_shape_1(dynamic_dim_1);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_1);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_2;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_2);
  EXPECT_EQ(op_slice_info, op_slice_info_2);

  op_desc_ptr->MutableOutputDesc("out")->SetShape(dynamic_shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_3;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_3);
  EXPECT_EQ(op_slice_info, op_slice_info_3);

  op_desc_ptr->MutableOutputDesc("out")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_4;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_4);
  EXPECT_NE(op_slice_info, op_slice_info_4);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_6) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("conv2dbackpropinput", "Conv2DBackpropInput");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc);
  op_desc_ptr->AddInputDesc("z", tensor_desc);
  op_desc_ptr->AddOutputDesc("out", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW_DECONV, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim = {1, 3, -1, 5};
  GeShape dynamic_shape(dynamic_dim);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape);
  op_desc_ptr->MutableInputDesc("z")->SetShape(dynamic_shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW_DECONV, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_EQ(op_slice_info, op_slice_info_1);

  vector<int64_t> dynamic_dim_1 = {-2};
  GeShape dynamic_shape_1(dynamic_dim_1);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_1);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW_DECONV, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_2;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_2);
  EXPECT_EQ(op_slice_info, op_slice_info_2);

  op_desc_ptr->MutableOutputDesc("out")->SetShape(dynamic_shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW_DECONV, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_3;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_3);
  EXPECT_EQ(op_slice_info, op_slice_info_3);

  op_desc_ptr->MutableOutputDesc("out")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, SLIDING_WINDOW_DECONV, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_4;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_4);
  EXPECT_NE(op_slice_info, op_slice_info_4);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_dynamic_7) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("matmul_v2", "MatMulV2");
  vector<int64_t> dim(4, 1);
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc);
  op_desc_ptr->AddInputDesc("z", tensor_desc);
  op_desc_ptr->AddInputDesc("b", tensor_desc);
  op_desc_ptr->AddOutputDesc("out", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, CUBE_MATMUL, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);

  vector<int64_t> dynamic_dim = {1, 3, -1, 5};
  GeShape dynamic_shape(dynamic_dim);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape);
  op_desc_ptr->MutableInputDesc("z")->SetShape(dynamic_shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, CUBE_MATMUL, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_1;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_1);
  EXPECT_EQ(op_slice_info, op_slice_info_1);

  vector<int64_t> dynamic_dim_1 = {-2};
  GeShape dynamic_shape_1(dynamic_dim_1);

  op_desc_ptr->MutableInputDesc("x")->SetShape(dynamic_shape_1);
  op_desc_ptr->MutableInputDesc("y")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, CUBE_MATMUL, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_2;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_2);
  EXPECT_EQ(op_slice_info, op_slice_info_2);

  op_desc_ptr->MutableOutputDesc("out")->SetShape(dynamic_shape);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, CUBE_MATMUL, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_3;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_3);
  EXPECT_EQ(op_slice_info, op_slice_info_3);

  op_desc_ptr->MutableOutputDesc("out")->SetShape(dynamic_shape_1);
  op_desc_ptr->DelAttr("_op_slice_info");
  ret = op_setter_ptr->SetOpSliceInfo(node, CUBE_MATMUL, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info_4;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info_4);
  EXPECT_NE(op_slice_info, op_slice_info_4);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elemwisebroadcast_case1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {1, 2, 3, 4};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim1 = {4};
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1, FORMAT_NCHW, DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(DT_FLOAT);

  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc1);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elemwisebroadcast_case2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {1, 2, 3, 4};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim1 = {2, 3, 1, 4};
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1, FORMAT_NCHW, DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim2 = {2, 3, 1};
  GeShape shape2(dim2);
  GeTensorDesc tensor_desc2(shape2, FORMAT_NCHW, DT_FLOAT);
  tensor_desc2.SetOriginShape(shape2);
  tensor_desc2.SetOriginFormat(FORMAT_NCHW);
  tensor_desc2.SetOriginDataType(DT_FLOAT);

  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc1);
  op_desc_ptr->AddOutputDesc("z", tensor_desc2);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elemwisebroadcast_case3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {4};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim1 = {1, 2, 3, 4};
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1, FORMAT_NCHW, DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(DT_FLOAT);

  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc1);
  op_desc_ptr->AddOutputDesc("z", tensor_desc1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_op_slice_info_elemwisebroadcast_fail1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {4};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim1 = {1, 2, 3, 4};
  GeShape shape1(dim1);
  GeTensorDesc tensor_desc1(shape1, FORMAT_NCHW, DT_FLOAT);
  tensor_desc1.SetOriginShape(shape1);
  tensor_desc1.SetOriginFormat(FORMAT_NCHW);
  tensor_desc1.SetOriginDataType(DT_FLOAT);

  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddInputDesc("y", tensor_desc1);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr node = graph->AddNode(op_desc_ptr);

  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  fe::Status ret = op_setter_ptr->SetOpSliceInfo(node, ELEMENT_WISE_BROADCAST, true);
  EXPECT_EQ(fe::SUCCESS, ret);
  string op_slice_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, fe::OP_SLICE_INFO, op_slice_info);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_elemwise_slice_info_test) {
  ge::OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {4};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_NC1HWC0, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);

  size_t dim_size = 1;
  bool has_scalar = true;
  ge::Format op_output_format = Format::FORMAT_NC1HWC0;
  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  Status ret = op_setter_ptr->CheckElemwiseInputAndOutputNum(op_desc_ptr, has_scalar, dim_size, op_output_format);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<AxisSplitMap> axis_split_maps;
  bool sup_sw;
  bool is_filter_dynamic;
  ret = op_setter_ptr->SetElemWiseSliceInfo(op_desc_ptr, axis_split_maps, has_scalar, sup_sw, is_filter_dynamic);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_OP_SLICE_INFO_SETTER, set_elemwise_slice_info_test1) {
  ge::OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {4};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape, FORMAT_FRACTAL_NZ, DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginFormat(FORMAT_FRACTAL_NZ);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_desc);
  op_desc_ptr->AddOutputDesc("z", tensor_desc);

  size_t dim_size = 1;
  bool has_scalar = true;
  ge::Format op_output_format = Format::FORMAT_FRACTAL_NZ;
  auto op_setter_ptr = std::make_shared<OpSliceUtil>();
  Status ret = op_setter_ptr->CheckElemwiseInputAndOutputNum(op_desc_ptr, has_scalar, dim_size, op_output_format);
  EXPECT_EQ(ret, fe::SUCCESS);
  std::vector<AxisSplitMap> axis_split_maps;
  bool sup_sw;
  bool is_filter_dynamic;
  ret = op_setter_ptr->SetElemWiseSliceInfo(op_desc_ptr, axis_split_maps, has_scalar, sup_sw, is_filter_dynamic);
  EXPECT_EQ(ret, fe::FAILED);
}
