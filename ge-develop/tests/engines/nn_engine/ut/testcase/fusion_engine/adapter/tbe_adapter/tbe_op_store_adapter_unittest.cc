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
#include "all_ops_stub.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_context.h"
#include "./ge_local_context.h"

#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/plugin_manager.h"
#include "common/util/op_info_util.h"
#include "ops_store/op_kernel_info.h"
#include "common/fe_op_info_common.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/aicore_util_types.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/ffts_plus_type.h"
#undef protected
#undef private

using namespace std;
using namespace ge;
using namespace fe;
using namespace te;


class UTEST_tbe_op_store_adapter : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }

protected:
  static bool GetOpSpecificInfoSub(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfoString) {
    nlohmann::json op_specific_info;
    op_specific_info["rangeLimit"] = "limited";
    opSpecificInfoString = op_specific_info.dump();
    return true;
  }

  static bool GetOpSpecificInfoSub1(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfoString) {
    nlohmann::json op_specific_info;
    op_specific_info["rangeLimit"] = "unlimited";
    opSpecificInfoString = op_specific_info.dump();
    return true;
  }

  static bool GetOpSpecificInfoSub2(const te::TbeOpInfo &tbeOpInfo, std::string &opSpecificInfoString) {
    nlohmann::json op_specific_info;
    op_specific_info["rangeLimit"] = "dynamic";
    opSpecificInfoString = op_specific_info.dump();
    return true;
  }
};

TEST_F(UTEST_tbe_op_store_adapter, parse_hardwareinfos_01)
{
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  std::map<string, string> ge_options = {{"ge.aicoreNum", "16"},
                                         {"ge.hardwareInfo", "ai_core_cnt:12;vector_core_cnt:16"}};
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::HardwareInfo)] = "ai_core_cnt:12;vector_core_cnt:16";
  config.ParseHardwareInfo();
  std::map<string, string> options = {{"ge.aicoreNum", "16"}};
  tbe_adapter_ptr->ParseHardwareInfos(options);
  EXPECT_EQ(options["ge.aicoreNum"], "16");
  EXPECT_EQ(options["ai_core_cnt"], "12");
  EXPECT_EQ(options["vector_core_cnt"], "16");
}

TEST_F(UTEST_tbe_op_store_adapter, parse_hardwareinfos_02)
{
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  std::map<string, string> ge_options = {{"ge.aicoreNum", "16"},
                                         {"ge.hardwareInfo", "ai_core_cnt:12;vector_core_cnt:16"}};
  Configuration& config = Configuration::Instance(fe::AI_CORE_NAME);
  config.config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::HardwareInfo)] = "ai_core_cnt:12;vector_core_cnt:16";
  config.ParseHardwareInfo();
  std::map<string, string> options = {};
  tbe_adapter_ptr->ParseHardwareInfos(options);
  EXPECT_EQ(options["ge.aicoreNum"], "12");
  EXPECT_EQ(options["ai_core_cnt"], "12");
  EXPECT_EQ(options["vector_core_cnt"], "16");
}

TEST_F(UTEST_tbe_op_store_adapter, GetRangeLimitedType_1)
{
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V300);
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend310B1";
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310B1");
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::OpDescPtr op_desc = make_shared<ge::OpDesc>("relu", "Relu");
  ge::NodePtr node = graph->AddNode(op_desc);

  te::TbeOpInfo info("test",
                     "test1",
                     "DynamicCompileStatic",
                     fe::AI_CORE_NAME);
  bool is_limited = false;
  Status ret = tbe_adapter_ptr->GetRangeLimitType(node, info, is_limited);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(is_limited, true);

  tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub1;
  ret = tbe_adapter_ptr->GetRangeLimitType(node, info, is_limited);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(is_limited, false);

  tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub2;
  ret = tbe_adapter_ptr->GetRangeLimitType(node, info, is_limited);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(is_limited, false);
}

TEST_F(UTEST_tbe_op_store_adapter, UpdateTensorByMixPrecisionMode_1) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V300);
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend310B1";
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310B1");
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  OpKernelInfoPtr op_kernel_info = std::make_shared<OpKernelInfo>("test");

  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");

  // add descriptor
  vector<int64_t> dims = {288, 32, 16, 16};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_Z);
  in_desc1.SetDataType(ge::DT_FLOAT);
  bn_op->AddInputDesc("x", in_desc1);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NHWC);
  out_desc1.SetDataType(ge::DT_FLOAT);
  bn_op->AddOutputDesc("y", out_desc1);
  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  NodePtr bn_node = graph->AddNode(bn_op);

  std::pair<std::vector<size_t>, std::vector<size_t>> in_out_changed_idx_vec;
  tbe_adapter_ptr->UpdateTensorByMixPrecisionMode(bn_node, op_kernel_info, in_out_changed_idx_vec);
  bool res = bn_op->MutableInputDesc(0)->GetDataType() == ge::DT_FLOAT16;
  EXPECT_EQ(res, true);
}

TEST_F(UTEST_tbe_op_store_adapter, UpdateTensorByMixPrecisionMode_2) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V300);
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend310B1";
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310B1");
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = FORCE_FP16;
  OpKernelInfoPtr op_kernel_info = std::make_shared<OpKernelInfo>("test");

  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
  (void)ge::AttrUtils::SetInt(bn_op, KEEP_DTYPE, 1);
  // add descriptor
  vector<int64_t> dims = {288, 32, 16, 16};
  GeShape shape(dims);
  GeTensorDesc in_desc1(shape);
  in_desc1.SetFormat(FORMAT_FRACTAL_Z);
  in_desc1.SetDataType(ge::DT_FLOAT);
  bn_op->AddInputDesc("x", in_desc1);
  GeTensorDesc out_desc1(shape);
  out_desc1.SetFormat(FORMAT_NHWC);
  out_desc1.SetDataType(ge::DT_FLOAT);
  bn_op->AddOutputDesc("y", out_desc1);
  ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
  NodePtr bn_node = graph->AddNode(bn_op);

  std::pair<std::vector<size_t>, std::vector<size_t>> in_out_changed_idx_vec;
  tbe_adapter_ptr->UpdateTensorByMixPrecisionMode(bn_node, op_kernel_info, in_out_changed_idx_vec);
  bool res = bn_op->MutableInputDesc(0)->GetDataType() == ge::DT_FLOAT;
  EXPECT_EQ(res, true);
}

TEST_F(UTEST_tbe_op_store_adapter, DealOpDebugDir_success) {
  PlatformUtils::Instance().soc_version_ = "Ascend310B1";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V300);
  PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend310B1";
  PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310B1");
  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->GetOpSpecificInfo = GetOpSpecificInfoSub;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::OpDescPtr op_desc = make_shared<ge::OpDesc>("relu", "Relu");
  ge::NodePtr node = graph->AddNode(op_desc);

  std::map<std::string, std::string> options;
  char *path = new(std::nothrow) char[1024];
  getcwd(path, 1024);
  tbe_adapter_ptr->DealOpDebugDir(options);
  string value = options[ge::DEBUG_DIR];
  int is_same_value = value.compare(string(path));
  delete[] path;
  EXPECT_EQ(is_same_value, 0);
}

TEST_F(UTEST_tbe_op_store_adapter, task_fusion_case1) {
  Configuration::Instance(AI_CORE_NAME).InitLibPath();
  vector<int64_t> dims = {3, 4, 5, 6};
  ge::GeShape shape(dims);
  ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  OpDescPtr add_op = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");

  relu_op->AddInputDesc(tensor_desc);
  relu_op->AddOutputDesc(tensor_desc);
  sigmoid_op->AddInputDesc(tensor_desc);
  sigmoid_op->AddOutputDesc(tensor_desc);
  add_op->AddInputDesc(tensor_desc);
  add_op->AddInputDesc(tensor_desc);
  add_op->AddOutputDesc(tensor_desc);

  AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(sigmoid_op, "_fe_imply_type", 6);
  AttrUtils::SetInt(add_op, "_fe_imply_type", 6);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu_node = graph->AddNode(relu_op);
  NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
  NodePtr add_node = graph->AddNode(add_op);

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());

  std::vector<ge::Node *> nodes;
  nodes.push_back(add_node.get());
  nodes.push_back(relu_node.get());
  nodes.push_back(sigmoid_node.get());
  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.emplace(5, nodes);

  CompileResultMap compile_ret_map;
  Status ret = tbe_adapter_ptr->TaskFusion(fusion_nodes_map, compile_ret_map);
  EXPECT_EQ(ret, fe::SUCCESS);
  EXPECT_EQ(compile_ret_map.size(), 1);
}

TEST_F(UTEST_tbe_op_store_adapter, wait_task_debug_message) {
  te::FinComTask fin_task1;
  fin_task1.teNodeOpDesc = std::make_shared<ge::OpDesc>("op1", "test");
  fin_task1.taskId = 112;
  fin_task1.status = 1;

  te::FinComTask fin_task2;
  fin_task2.teNodeOpDesc = std::make_shared<ge::OpDesc>("op1", "test");
  fin_task2.taskId = 345;
  fin_task2.status = 1;

  te::FinComTask fin_task3;
  fin_task3.teNodeOpDesc = std::make_shared<ge::OpDesc>("op1", "test");
  fin_task3.taskId = 996;
  fin_task3.status = 1; 

  std::vector<te::FinComTask> fin_tasks = {fin_task1, fin_task2, fin_task3};
  std::string res = GetStrByFinComTaskVec(fin_tasks);
  EXPECT_EQ(res, "[112],[345],[996]");

  std::map<uint64_t, int64_t> task_scope_id_map;
  task_scope_id_map[666] = 2;
  task_scope_id_map[222] = 3;
  res = GetStrTaskIdByMap(task_scope_id_map);
  EXPECT_EQ(res, "[222],[666]");
}

TEST_F(UTEST_tbe_op_store_adapter, trans_string_to_dtype) {
  ge::DataType ge_dtype;
  const string dtype_str = "float";
  TransStringToDtype(dtype_str, ge_dtype);
  EXPECT_EQ(ge_dtype, ge::DT_FLOAT);
}

TEST_F(UTEST_tbe_op_store_adapter, trans_dtype_to_string) {
  const ge::DataType ge_dtype = ge::DT_FLOAT;
  string dtype_str;
  TransDtypeToString(ge_dtype, dtype_str);
  EXPECT_EQ(dtype_str, "float32");
}

TEST_F(UTEST_tbe_op_store_adapter, trans_dtype_to_string01) {
  const ge::DataType ge_dtype = ge::DT_FLOAT8_E8M0;
  string dtype_str;
  TransDtypeToString(ge_dtype, dtype_str);
  EXPECT_EQ(dtype_str, "float8_e8m0");
}

TEST_F(UTEST_tbe_op_store_adapter, test_set_extra_params) {
  std::string op_name    = "conv";
  std::string op_module   = "";
  std::string op_type     = "tbe";
  std::string core_type   = "AIcoreEngine";
  OpDescPtr conv_op = std::make_shared<OpDesc>("Conv", "conv");

  TbeInfoAssembler tbe;
  TbeOpInfo op_info(op_name, op_module, op_type, core_type);
  ge::AttrUtils::SetStr(*(conv_op.get()), "_op_aicore_num", "10");
  ge::AttrUtils::SetStr(*(conv_op.get()), "_op_vectorcore_num", "10");
  tbe.SetCustCoreNum(*(conv_op.get()), op_info);
}