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
#include "graph_optimizer/dsa_optimizer/dsa_graph_optimizer.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"

#include "graph/utils/graph_utils.h"
#include "common/util/op_info_util.h"
#include "common/fe_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/dsa_adapter/dsa_op_store_adapter.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "./ge_context.h"
#include "./ge_local_context.h"
#undef protected
#undef private

using namespace testing;
using namespace fe;
using namespace ge;
using DSAOpStoreAdapterPtr = std::shared_ptr<fe::DSAOpStoreAdapter>;
using DSAGraphOptimizerPtr = std::shared_ptr<fe::DSAGraphOptimizer>;
using OpStoreAdapterPtr = std::shared_ptr<fe::OpStoreAdapter>;
class UTEST_fusion_engine_dsa_graph_optimizer : public testing::Test {
 public:
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
  FormatDtypeSetterPtr format_dtype_setter_ptr_;
  OpImplyTypeJudgePtr op_impl_type_judge_ptr_;
  DSAGraphOptimizerPtr dsa_graph_optimizer_;
  shared_ptr<fe::SubOpInfoStore> sub_ops_kernel_ptr;
  shared_ptr<fe::SubOpsStore> sub_ops_store_ptr;
 protected:
  void SetUp() {
    PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::DsaWorkspaceSize)] = 48;
    ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::kDsaCoreName);
    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
    dsa_graph_optimizer_ = make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, fe::kDsaCoreName);
    FEOpsStoreInfo TBE_OPINFO_STUB = {
            12,
            "dsa-builtin",
            EN_IMPL_HW_DSA,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/dsa_opinfo",
            ""
    };
    sub_ops_store_ptr = make_shared<fe::SubOpsStore>(fe::kDsaCoreName);
    sub_ops_store_ptr->SetSubStoreInfo(TBE_OPINFO_STUB);
    sub_ops_store_ptr->InitializeSubStore();

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(TBE_OPINFO_STUB);
    Configuration::Instance(fe::kDsaCoreName).ops_store_info_vector_ = (store_info);

    sub_ops_kernel_ptr = std::make_shared<fe::SubOpInfoStore>(TBE_OPINFO_STUB);
    sub_ops_kernel_ptr->Initialize(fe::kDsaCoreName);
    OpsKernelManager::Instance(fe::kDsaCoreName).sub_ops_kernel_map_.emplace("dsa-builtin", sub_ops_kernel_ptr);

    std::map<std::string, std::string> options;
    options.insert(std::pair<std::string, std::string>("ge.shape_generalized_build_mode", SHAPE_GENERALIZED));
    ge::GetThreadLocalContext().SetGlobalOption(options);

    std::map<std::string, std::string> options1;
    OpsKernelManager::Instance(fe::kDsaCoreName).Finalize();
    ops_kernel_info_store_ptr_->Initialize(options1);
  }

  void TearDown() {
    sub_ops_store_ptr->FinalizeSubStore();
    sub_ops_store_ptr.reset();
    sub_ops_kernel_ptr->Finalize();
    sub_ops_kernel_ptr.reset();
    ops_kernel_info_store_ptr_->Finalize();
  }

  static void CreateGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("dsa", "DSAGenBitMask");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
    OpDescPtr out = std::make_shared<OpDesc>("OUT", fe::NETOUTPUT);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_ND);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetDataType(DT_INT32);
    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("count", in_desc1);
    bn_op->AddInputDesc("dropout", in_desc2);
    data->AddOutputDesc("y", in_desc1);
    data1->AddOutputDesc("y", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_ND);
    out_desc2.SetOriginFormat(FORMAT_ND);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("out", out_desc2);
    out->AddInputDesc("x", out_desc2);

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr out_node = graph->AddNode(out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  static void CreateGraphNODSA(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("dsa", "NO_OPS");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
    OpDescPtr out = std::make_shared<OpDesc>("OUT", fe::NETOUTPUT);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_ND);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetDataType(DT_INT32);
    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("count", in_desc1);
    bn_op->AddInputDesc("dropout", in_desc2);
    data->AddOutputDesc("y", in_desc1);
    data1->AddOutputDesc("y", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_ND);
    out_desc2.SetOriginFormat(FORMAT_ND);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("out", out_desc2);
    out->AddInputDesc("x", out_desc2);

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr out_node = graph->AddNode(out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  static void CreateDSAGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("dsa", "DSAGenBitMask");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
    OpDescPtr out = std::make_shared<OpDesc>("OUT", fe::NETOUTPUT);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_ND);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetDataType(DT_INT32);
    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("count", in_desc1);
    bn_op->AddInputDesc("dropout", in_desc2);
    data->AddOutputDesc("y", in_desc1);
    data1->AddOutputDesc("y", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_ND);
    out_desc2.SetOriginFormat(FORMAT_ND);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("out", out_desc2);
    (void) ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, EN_IMPL_HW_DSA);
    out->AddInputDesc("x", out_desc2);

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr out_node = graph->AddNode(out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }
  static void SetWeight(ge::NodePtr node, int index, float w) {
    float data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT16);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    map<int, ge::GeTensorPtr> weights = {{index, tensor}};
    ge::OpDescUtils::SetWeights(*node, weights);
  }
  static void SetWeight(ge::NodePtr node, int index, int64_t w) {
    int64_t data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT64);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    map<int, ge::GeTensorPtr> weights = {{index, tensor}};
    ge::OpDescUtils::SetWeights(*node, weights);
  }
  static void CreateTbeGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("dsa", "DSAGenBitMask");
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
    OpDescPtr out = std::make_shared<OpDesc>("OUT", fe::NETOUTPUT);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_ND);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetDataType(DT_INT32);
    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_ND);
    in_desc2.SetOriginFormat(FORMAT_ND);
    in_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("count", in_desc1);
    bn_op->AddInputDesc("dropout", in_desc2);
    data->AddOutputDesc("y", in_desc1);
    data1->AddOutputDesc("y", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_ND);
    out_desc2.SetOriginFormat(FORMAT_ND);
    out_desc2.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("out", out_desc2);
    (void) ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    out->AddInputDesc("x", out_desc2);

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr out_node = graph->AddNode(out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), bn_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, get_attributes_success)
{
  GraphOptimizerAttribute attrs;
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);
  Status status = dsa_graph_optimizer_->GetAttributes(attrs);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, dsa_op_store_adapter_finalize)
{
  std::map<std::string, std::string> options;
  DSAOpStoreAdapterPtr dsa_adapter_ptr = std::make_shared<DSAOpStoreAdapter>();
  dsa_adapter_ptr->init_flag = true;
  dsa_adapter_ptr->Initialize(options);
  dsa_adapter_ptr->init_flag = false;
  dsa_adapter_ptr->Finalize();
  dsa_adapter_ptr->init_flag = true;
  EXPECT_EQ(fe::SUCCESS, dsa_adapter_ptr->Finalize());
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, initialize_success)
{
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);
  std::map<std::string, std::string> options;
  dsa_graph_optimizer_->init_flag_ = true;
  Status status = dsa_graph_optimizer_->Initialize(options, nullptr);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, finalize_success)
{
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);
  dsa_graph_optimizer_->init_flag_ = true;
  Status status = dsa_graph_optimizer_->Finalize();
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, finalize_success1)
{
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);
  dsa_graph_optimizer_->init_flag_ = false;
  Status status = dsa_graph_optimizer_->Finalize();
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_original_graph_init_again)
{
  std::map<std::string, std::string> options;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

  dsa_graph_optimizer_->init_flag_ = false;
  Status status = dsa_graph_optimizer_->OptimizeOriginalGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_original_graph_fail1)
{
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "allow_fp32_to_fp16";
  std::map<std::string, std::string> options;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateGraphNODSA(graph);
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

  (void)dsa_graph_optimizer_->Initialize(options, nullptr);
  Status status = dsa_graph_optimizer_->OptimizeOriginalGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

// TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_original_graph)
// {
//   ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "allow_fp32_to_fp16";
//   std::map<std::string, std::string> options;
//   auto graph = std::make_shared<ComputeGraph>("test");
//   CreateGraph(graph);
//   auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

//   (void)dsa_graph_optimizer_->Initialize(options, nullptr);
//   Status status = dsa_graph_optimizer_->OptimizeOriginalGraph(*(graph.get()));
//   EXPECT_EQ(fe::SUCCESS, status);
//   for (auto node : graph->GetAllNodes()) {
//     if (node->GetType() == "DSAGenBitMask") {
//       int64_t fe_impl_type = -1;
//       (void)ge::AttrUtils::GetInt(node->GetOpDesc(), FE_IMPLY_TYPE, fe_impl_type);
//       EXPECT_EQ(fe_impl_type, static_cast<OpImplType>(EN_IMPL_HW_DSA));
//       std::string op_slice_info;
//       (void)ge::AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
//       std::cout << "Node DSAGenBitMask slice info is：" << op_slice_info << endl;
//     }
//   }
// }

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_whole_graph)
{
  std::map<std::string, std::string> options;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
  GeTensorDesc src_tensor_desc(GeShape({1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({1}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);

  auto src_node = graph->AddNode(src_op);
  SetWeight(src_node, 0, 512L);
  SetWeight(src_node, 2, 0.24f);
  (void)ge::AttrUtils::SetInt(src_node->GetOpDesc(), FE_IMPLY_TYPE, EN_IMPL_HW_DSA);
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

  (void)dsa_graph_optimizer_->Initialize(options, nullptr);
  Status status = dsa_graph_optimizer_->OptimizeWholeGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

// TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_original_graph1)
// {
//   ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = "allow_fp32_to_fp16";
//   std::map<std::string, std::string> options;
//   auto graph = std::make_shared<ComputeGraph>("test");
//   CreateGraph(graph);

//   auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

//   (void)dsa_graph_optimizer_->Initialize(options, nullptr);
//   Status status = dsa_graph_optimizer_->OptimizeOriginalGraph(*(graph.get()));
//   EXPECT_EQ(fe::SUCCESS, status);
//   for (auto node : graph->GetAllNodes()) {
//     if (node->GetType() == "DSAGenBitMask") {
//       int64_t fe_impl_type = -1;
//       (void)ge::AttrUtils::GetInt(node->GetOpDesc(), FE_IMPLY_TYPE, fe_impl_type);
//       EXPECT_EQ(fe_impl_type, static_cast<OpImplType>(EN_IMPL_HW_DSA));
//       std::string op_slice_info;
//       (void)ge::AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
//       std::cout << "Node DSAGenBitMask slice info is：" << op_slice_info << endl;
//     }
//   }
// }

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_original_graph2)
{
  std::map<std::string, std::string> options;
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateTbeGraph(graph);
  auto format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(kDsaCoreName);
  Status status = format_dtype_setter_ptr_->SetSupportFormatDtype(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_fused_graph)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateDSAGraph(graph);
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

  std::map<std::string, std::string> options;
  (void)dsa_graph_optimizer_->Initialize(options, nullptr);
  Status status = dsa_graph_optimizer_->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "DSAGenBitMask") {
      std::vector<int32_t> memory_no_reuse;
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, memory_no_reuse);
      std::vector<int32_t> memory_no_reuse1 = {1, 1};
      EXPECT_EQ(memory_no_reuse1, memory_no_reuse);
      std::vector<int64_t> workspace_bytes = {48, 48};
      EXPECT_EQ(node->GetOpDesc()->GetWorkspaceBytes(), workspace_bytes);
    }
  }
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, optimize_fused_graph_failed)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateDSAGraph(graph);
  auto dsa_graph_optimizer_ = std::make_shared<DSAGraphOptimizer>(ops_kernel_info_store_ptr_, kDsaCoreName);

  std::map<std::string, std::string> options;
  (void)dsa_graph_optimizer_->Initialize(options, nullptr);
  dsa_graph_optimizer_->init_flag_ = false;
  Status status = dsa_graph_optimizer_->OptimizeFusedGraph(*(graph.get()));
  EXPECT_EQ(fe::FAILED, status);
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, check_supported)
{
  auto graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);

  std::map<std::string, std::string> options;
  (void)ops_kernel_info_store_ptr_->Initialize(options);
  for (auto node : graph->GetAllNodes()) {
    if (node->GetType() == "DSAGenBitMask") {
      std::string reason;
      bool res = ops_kernel_info_store_ptr_->CheckSupported(node->GetOpDesc(), reason);
      EXPECT_EQ(true, res);
    }
  }
}

TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, fe_util_ut1)
{
  std::string engine_name = AI_CORE_NAME;
  OpImplType op_impl_type = EN_IMPL_HW_DSA;
  bool ret = IsInvalidImplType(engine_name, op_impl_type);
  EXPECT_EQ(ret, true);
}
TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, fe_util_ut2)
{
  std::string engine_name = kDsaCoreName;
  OpImplType op_impl_type = EN_IMPL_HW_TBE;
  bool ret = IsInvalidImplType(engine_name, op_impl_type);
  EXPECT_EQ(ret, true);
}
TEST_F(UTEST_fusion_engine_dsa_graph_optimizer, fe_util_ut3)
{
  std::string engine_name = kDsaCoreName;
  OpImplType op_impl_type = EN_IMPL_HW_DSA;
  bool ret = IsInvalidImplType(engine_name, op_impl_type);
  EXPECT_EQ(ret, false);
}
