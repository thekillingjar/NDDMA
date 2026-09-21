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
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "adapter/common/get_attr_by_type.h"
#include "common/fe_op_info_common.h"
#include "common/fe_log.h"
#include "common/sgt_slice_type.h"
#include "common/aicore_util_types.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/plugin_manager.h"
#include "common/scope_allocator.h"
#include "common/util/op_info_util.h"
#include "platform/platform_info.h"
#include "ops_store/op_kernel_info.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/op_compiler/op_compiler_normal.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "register/op_impl_registry.h"
#undef protected
#undef private

#include "adapter/common/op_store_adapter_manager.h"
#include "graph/ge_local_context.h"
#include "graph/tuning_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_kernel_bin.h"
#include "all_ops_stub.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace te;

#define KERNEL_NUM  2

namespace stest {

using TbeInfoAssemblerPtr = std::shared_ptr<TbeInfoAssembler>;
bool CheckTbeSupportedStub(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.isSupported = te::FULLY_SUPPORTED;
  return true;
}

class STEST_FE_TBE_COMPILER : public testing::Test {
 protected:

  void SetUp() {
    PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B");
    PlatformInfoManager::Instance().opti_compilation_infos_.SetAICoreNum(8);
    fe::OpsKernelManager::Instance(AI_CORE_NAME).is_init_ = true;

    tbe_op_store_adapter_ptr_ = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_op_store_adapter_ptr_->CheckTbeSupported = CheckTbeSupportedStub;
    graph_comm_ptr_ = std::make_shared<GraphComm>(fe::AI_CORE_NAME);
    std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr = std::make_shared<FusionPriorityManager>
            (fe::AI_CORE_NAME, nullptr);
    ub_fusion_ptr_ = std::make_shared<BufferFusion>(graph_comm_ptr_, fusion_priority_mgr_ptr, nullptr);
    ub_fusion_ptr_->SetEngineName(fe::AI_CORE_NAME);
    PlatformInfo platform_info;
    platform_info.ai_core_spec.cube_vector_split = 0;
    PlatformInfoManager::Instance().platform_info_map_.emplace("Ascend910B1", platform_info);
    FEOpsStoreInfo CCE_CUSTOM_OPINFO_STUB{
        0,
        "cce-custom",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_custom_opinfo",
        "",
        false,
        false,
        false
    };
    FEOpsStoreInfo TIK_CUSTOM_OPINFO_STUB = {
        1,
        "tik-custom",
        EN_IMPL_CUSTOM_TIK,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_custom_opinfo",
        "",
        false,
        false,
        false
    };
    FEOpsStoreInfo TBE_CUSTOM_OPINFO_STUB = {
        2,
        "tbe-custom",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
        "",
        false,
        false,
        false
    };
    FEOpsStoreInfo CCE_CONSTANT_OPINFO_STUB = {
        3,
        "cce-constant",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_constant_opinfo",
        "",
        false,
        false,
        false
    };
    FEOpsStoreInfo CCE_GENERAL_OPINFO_STUB = {
        4,
        "cce-general",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/cce_general_opinfo",
        "",
        false,
        false,
        false
    };
    FEOpsStoreInfo TIK_OPINFO_STUB = {
        5,
        "tik-builtin",
        EN_IMPL_HW_TIK,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tik_opinfo",
        "",
        false,
        false,
        false
    };
    FEOpsStoreInfo TBE_OPINFO_STUB = {
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
        "",
        false,
        true,
        false
    };
    FEOpsStoreInfo RL_OPINFO_STUB = {
        7,
        "rl-built",
        EN_IMPL_RL,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/rl_opinfo",
        "",
        false,
        false,
        false
    };

    cfg_info_.push_back(CCE_CUSTOM_OPINFO_STUB);
    cfg_info_.push_back(TIK_CUSTOM_OPINFO_STUB);
    cfg_info_.push_back(TBE_CUSTOM_OPINFO_STUB);
    cfg_info_.push_back(CCE_CONSTANT_OPINFO_STUB);
    cfg_info_.push_back(CCE_GENERAL_OPINFO_STUB);
    cfg_info_.push_back(TIK_OPINFO_STUB);
    cfg_info_.push_back(TBE_OPINFO_STUB);
    cfg_info_.push_back(RL_OPINFO_STUB);
  }

  void TearDown() {
  }

  ge::NodePtr AddNode(ge::ComputeGraphPtr graph,
                      const string &name,
                      const string &type,
                      unsigned int in_anchor_num,
                      unsigned int out_anchor_num,
                      ge::DataType input_type,
                      ge::DataType output_type) {
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

  /*
* batchnorm
*    |
*   relu
*/
  static void CreateGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

    // add descriptor
    vector<int64_t> dims = {288, 32, 16, 16};
    GeShape shape(dims);
    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_FRACTAL_Z);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc1);
    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_NHWC);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc1);
    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_NCHW);
    in_desc2.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc2);
    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_HWCN);
    out_desc2.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc2);
    ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  }
  void SetPattern(ge::OpDescPtr opdef, const string &optype) {
    auto key_pattern = "_pattern";
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
  }
  void SetTvmType(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  }

 protected:
  TbeOpStoreAdapterPtr tbe_op_store_adapter_ptr_;
  std::shared_ptr<GraphComm> graph_comm_ptr_;
  std::shared_ptr<BufferFusion> ub_fusion_ptr_;
  std::vector<FEOpsStoreInfo> cfg_info_;
};

bool QueryInputNeedCompileStub(FEOpsKernelInfoStore *This, const ge::OpDesc &op_desc, const string tensor_name) {
  map<string, bool> map_bool = {{"w1",         true},
                                {"x",          false},
                                {"w2",         true},
                                {"optional",   false},
                                {"dynamicIn1", false},
                                {"dynamicIn2", false}};
  return map_bool[tensor_name];
}

Status
QueryParamTypeStub(FEOpsKernelInfoStore *This, std::string name, std::string op_type, fe::OpParamType &param_type) {
  map<string, fe::OpParamType> type = {{"w1", fe::OpParamType::REQUIRED}, \
                                         {"x", fe::OpParamType::REQUIRED}, \
                                         {"w2", fe::OpParamType::REQUIRED}, \
                                         {"optional", fe::OpParamType::OPTIONAL}, \
                                         {"dynamicIn2", fe::OpParamType::DYNAMIC}, \
                                         {"dynamicOut1", fe::OpParamType::DYNAMIC}, \
                                         {"dynamicOut2", fe::OpParamType::DYNAMIC}};

  param_type = type[name];
  return fe::SUCCESS;
}

te::OpBuildResCode
SgtTeFusionStub(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>
&to_be_del, uint64_t taskid, uint64_t tid, uint64_t sgt_thread_index, const std::string op_compile_strategy) {
  string json_file_path = "./kernel_meta/";
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_SUCC;
}

te::OpBuildResCode
SgtTeFusionStub2(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>
&to_be_del, uint64_t taskid, uint64_t tid, uint64_t sgt_thread_index, const std::string op_compile_strategy) {
  string json_file_path = "./kernel_meta/";
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_FAIL;
}

te::OpBuildResCode TeFusionStub(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>
&to_be_del, uint64_t taskid, uint64_t tid, const std::string op_compile_strategy) {
  string json_file_path = "./kernel_meta/";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_SUCC;
}

te::OpBuildResCode TeFusionVStub(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>
&to_be_del, uint64_t taskid, uint64_t tid, uint64_t sgtThreadIndex, const std::string op_compile_strategy) {
  string json_file_path = "./kernel_meta/";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_SUCC;
}

te::OpBuildResCode
TeFusionStub2(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr> &to_be_del,
              uint64_t taskid, uint64_t tid, const std::string op_compile_strategy) {
  string json_file_path = "";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_SUCC;
}

te::OpBuildResCode
TeFusionStub3(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr> &to_be_del,
              uint64_t taskid, uint64_t tid, const std::string op_compile_strategy) {
  string json_file_path = "";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_BUILD_FAIL;
}

te::OpBuildResCode
TeFusionStub4(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr> &to_be_del,
              uint64_t taskid, uint64_t tid, const std::string op_compile_strategy) {
  string json_file_path = "./kernel_meta/";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  int64_t compile_info = 1000;
  string compile_info_dummy = "compile_info_json,compile_info_key";
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  AttrUtils::SetStr(op_desc_ptr, COMPILE_INFO_JSON, compile_info_dummy);
  AttrUtils::SetStr(op_desc_ptr, COMPILE_INFO_KEY, compile_info_dummy);
  return te::OP_BUILD_SUCC;
}

te::OpBuildResCode
TeFusionStub5(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr> &to_be_del,
              uint64_t taskid, uint64_t tid, const std::string op_compile_strategy) {
  string json_file_path = "./kernel_meta/";
  //OpDescPtr op_desc_ptr = output_node->GetOpDesc();
  AttrUtils::SetStr(op_desc_ptr, "json_file_path", json_file_path);
  return te::OP_DYNSHAPE_NOT_SUPPORT;
}

bool TbeFinalizeStub() {
  return true;
}

te::LX_QUERY_STATUS get_tbe_opinfo_stub(const te::TbeOpInfo &info, std::string &op_info) {
  return te::LX_QUERY_NOT_FOUND;
}

te::LX_QUERY_STATUS get_tbe_opinfo_stub_succ(const te::TbeOpInfo &info, std::string &op_info) {
  return te::LX_QUERY_SUCC;
}

bool pre_build_te_op_stub(TbeOpInfo &info, uint64_t taskid, uint64_t graphid) {
  std::string pattern = "eltwise";
  info.SetPattern(pattern);
  return true;
}

bool pre_build_te_op_stub2(TbeOpInfo &info, uint64_t taskid, uint64_t graphid) {
  std::string pattern = "";
  info.SetPattern(pattern);
  return true;
}

bool TbeInitializeStub(const std::map<std::string, std::string> &options, bool *support) {
  return true;
}

bool CheckTbeSupportedStub1(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.isSupported = te::PARTIALLY_SUPPORTED;
  return true;
}

bool CheckTbeSupportedStub2(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.isSupported = te::NOT_SUPPORTED;
  result.reason = "Not supported stub.";
  return true;
}

bool pre_build_te_op_stub_failed(TbeOpInfo &info, uint64_t taskid, uint64_t graphid) {
  // std::string pattern = "eltwise";
  // info.SetPattern(pattern);
  return false;
}

bool CheckTbeSupportedStub_OnlyFp16WillPass(TbeOpInfo &info, te::CheckSupportedResult &is_support,
                                            string &reason) {
  std::vector<TbeOpParam> inputs;
  std::vector<TbeOpParam> outputs;
  info.GetInputs(inputs);
  info.GetOutputs(outputs);
  for (auto &input: inputs) {
    std::vector<TbeOpTensor> tensors;
    input.GetTensors(tensors);
    for (auto &tensor: tensors) {
      string dtype;
      tensor.GetType(dtype);
      FE_LOGI("Dtype is %s", dtype.c_str());
      if (dtype != "float16") {
        is_support = te::NOT_SUPPORTED;
        reason = "Inputs only supported fp16.";
        return false;
      }
    }
  }

  for (auto &output: outputs) {
    std::vector<TbeOpTensor> tensors;
    output.GetTensors(tensors);
    for (auto &tensor: tensors) {
      string dtype;
      tensor.GetType(dtype);
      FE_LOGI("Dtype is %s", dtype.c_str());
      if (dtype != "float16") {
        is_support = te::NOT_SUPPORTED;
        reason = "Outputs only supported fp16.";
        return false;
      }
    }
  }
  is_support = te::FULLY_SUPPORTED;
  return true;
}

bool WaitAllFinishedStub(uint64_t tid, vector<te::FinComTask> &fin_task) {
  te::FinComTask fin_com_task;
  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
  fin_com_task.taskId = GetAtomicId() - 1;
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "jsonFilePath");
  fin_task.push_back(fin_com_task);
  return true;
}

bool WaitAllFinishedStub1(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  te::FinComTask fin_com_task;
  fin_com_task.taskId = GetAtomicId()-1;
  fin_com_task.status = 1;
  fin_task.push_back(fin_com_task);
  return true;
}

bool WaitAllFinishedStubNoJsonPath(uint64_t tid, vector<te::FinComTask> &fin_task) {
  te::FinComTask fin_com_task;
  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
  fin_com_task.taskId = GetAtomicId() - 1;
  fin_com_task.status = 0;

  fin_task.push_back(fin_com_task);
  return true;
}

bool WaitAllFinishedFailStub(uint64_t tid, vector<te::FinComTask> &fin_task) {
  te::FinComTask fin_com_task;
  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
  fin_com_task.taskId = GetAtomicId() - 1;
  fin_com_task.status = -1;
  fin_task.push_back(fin_com_task);
  return true;
}


TEST_F(STEST_FE_TBE_COMPILER, case_tbe_check_support_sucess) {
  OpDescPtr matmul_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  vector<int64_t> dim_data = {1, 3, 5, 5};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NHWC, DT_FLOAT);
  matmul_desc->AddInputDesc("x1", data_desc);
  matmul_desc->AddInputDesc("x2", data_desc);
  matmul_desc->AddOutputDesc("y", data_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(matmul_desc);

  FEOpsStoreInfo tbe_opinfo{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
  };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_opinfo);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  fe::OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  map<string, string> options;
  fe_ops_kernel_info_store_ptr->Initialize(options);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      tbe_opinfo.fe_ops_store_name, matmul_desc->GetType());

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  string reason;
  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub;
  CheckSupportParam check_param;
  check_param.op_kernel_ptr = op_kernel_info_ptr;
  bool is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, true);

  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub1;
  is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, true);
  bool is_part_support = false;
  ge::AttrUtils::GetBool(matmul_desc, STR_PARTIALLY_SUPPORTED, is_part_support);
  EXPECT_EQ(is_part_support, true);
}


TEST_F(STEST_FE_TBE_COMPILER, case_tbe_check_support_true_fail) {
  OpDescPtr matmul_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  vector<int64_t> dim_data = {1, 3, 5, 5};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NHWC, DT_FLOAT);
  matmul_desc->AddInputDesc("x1", data_desc);
  matmul_desc->AddInputDesc("x2", data_desc);
  matmul_desc->AddOutputDesc("y", data_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(matmul_desc);

  FEOpsStoreInfo tbe_opinfo{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
  };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_opinfo);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  map<string, string> options;
  fe_ops_kernel_info_store_ptr->Initialize(options);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      tbe_opinfo.fe_ops_store_name, matmul_desc->GetType());

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());

  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub1;
  string reason;
  CheckSupportParam check_param;
  check_param.op_kernel_ptr = op_kernel_info_ptr;
  bool is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, true);
  bool is_part_support = false;
  ge::AttrUtils::GetBool(matmul_desc, STR_PARTIALLY_SUPPORTED, is_part_support);
  EXPECT_EQ(is_part_support, true);

  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub2;
  is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, false);
  EXPECT_EQ(ge::AttrUtils::GetBool(matmul_desc, STR_PARTIALLY_SUPPORTED, is_part_support), false);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_need_check_support_no_flag_success) {
  OpDescPtr matmul_desc = std::make_shared<OpDesc>("matmul2", "MatMul2");
  vector<int64_t> dim_data = {1, 3, 5, 5};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NHWC, DT_FLOAT);
  matmul_desc->AddInputDesc("x1", data_desc);
  matmul_desc->AddInputDesc("x2", data_desc);
  matmul_desc->AddOutputDesc("y", data_desc);

  FEOpsStoreInfo tbe_opinfo{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
  };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_opinfo);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  map<string, string> options;
  fe_ops_kernel_info_store_ptr->Initialize(options);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      tbe_opinfo.fe_ops_store_name, matmul_desc->GetType());

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub;
  string reason;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(matmul_desc);
  CheckSupportParam check_param;
  check_param.op_kernel_ptr = op_kernel_info_ptr;
  bool is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, true);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_need_check_support_false_success) {
  OpDescPtr matmul_desc = std::make_shared<OpDesc>("matmul3", "MatMul3");
  vector<int64_t> dim_data = {1, 3, 5, 5};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NHWC, DT_FLOAT);
  matmul_desc->AddInputDesc("x1", data_desc);
  matmul_desc->AddInputDesc("x2", data_desc);
  matmul_desc->AddOutputDesc("y", data_desc);

  FEOpsStoreInfo tbe_opinfo{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
  };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_opinfo);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  map<string, string> options;
  fe_ops_kernel_info_store_ptr->Initialize(options);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      tbe_opinfo.fe_ops_store_name, matmul_desc->GetType());

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub;
  string reason;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(matmul_desc);
  CheckSupportParam check_param;
  check_param.op_kernel_ptr = op_kernel_info_ptr;
  bool is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, true);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_no_check_support_funtion_fail) {
  OpDescPtr matmul_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  vector<int64_t> dim_data = {1, 3, 5, 5};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NHWC, DT_FLOAT);
  matmul_desc->AddInputDesc("x1", data_desc);
  matmul_desc->AddInputDesc("x2", data_desc);
  matmul_desc->AddOutputDesc("y", data_desc);

  FEOpsStoreInfo tbe_opinfo{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
  };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_opinfo);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  map<string, string> options;
  fe_ops_kernel_info_store_ptr->Initialize(options);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      tbe_opinfo.fe_ops_store_name, matmul_desc->GetType());

  TbeOpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  tbe_adapter_ptr->Initialize(std::map<std::string, std::string>());
  tbe_adapter_ptr->CheckTbeSupported = CheckTbeSupportedStub2;
  string reason;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(matmul_desc);
  CheckSupportParam check_param;
  check_param.op_kernel_ptr = op_kernel_info_ptr;
  bool is_su = tbe_adapter_ptr->CheckSupport(test_node, check_param, false, reason);
  EXPECT_EQ(is_su, false);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_op_compiler_success) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusion = TeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);

  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;

  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_compiler_null_sgt_slice_op_success) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusionV = TeFusionVStub;
  compile_tbe_op.TeFusion = TeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);

  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo compile_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {compile_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  to_del_nodes.push_back(Node2);

  Status ret = compile_tbe_op.CompileMultiKernelSliceOp(fusion_nodes_map, compile_ret_map, compile_failed_nodes,
                                                        to_del_nodes);

  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_op_compiler_get_json_file_path_failed) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusion = TeFusionStub2;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStubNoJsonPath;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);

  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));
  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_op_compile_fusion_op_failed) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  ge::AttrUtils::SetInt(weight_op_desc1, FE_IMPLY_TYPE, EN_IMPL_PLUGIN_TBE),
      ge::AttrUtils::SetInt(weight_op_desc2, FE_IMPLY_TYPE, EN_IMPL_PLUGIN_TBE),
      weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));
  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_op_compile_l1_fusion_op_failed) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusion = TeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  ge::AttrUtils::SetInt(weight_op_desc1, FE_IMPLY_TYPE, EN_IMPL_PLUGIN_TBE),
      ge::AttrUtils::SetInt(weight_op_desc2, FE_IMPLY_TYPE, EN_IMPL_PLUGIN_TBE),
      weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  ge::AttrUtils::SetBool(weight_op_desc1, NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));
  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_string_success) {
  ge::GeAttrValue attr_value_s;
  ge::GeAttrValue attr_value_int_list;
  ge::GeAttrValue value_bool;
  ge::GeAttrValue value_int;
  ge::GeAttrValue value_float;
  attr_value_s.SetValue<std::vector<std::string>>({"abc", "def"});
  attr_value_int_list.SetValue<vector<vector<int64_t>>>({{1, 2}, {1, 3}});
  value_bool.SetValue<vector<bool>>({false, true});
  value_int.SetValue<vector<int64_t>>({1, 2});
  value_float.SetValue<vector<float>>({(float)1.2, (float)2.1});
  vector<string> valueS = {"abc", "bn"};
  vector<int64_t> valueI = {1, 2};
  vector<float> value_f = {1.0, 2.1};
  IMPL_OP(conv)
    .PrivateAttr("x2", value_bool)
    .PrivateAttr("x3", (int64_t)1)
    .PrivateAttr("x4", "abc")
    .PrivateAttr("x5", (float)1.1)
    .PrivateAttr("x6", attr_value_s)
    .PrivateAttr("x7", value_int)
    .PrivateAttr("x8", value_float)
    .PrivateAttr("x9", attr_value_int_list);
  
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<std::string>("abc");
  op->SetAttr("x1", attr_value);

  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_STRING;
  attr_info_ptr->is_required_ = true;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";
  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_float_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<float>(1.1);
  op->SetAttr("x1", attr_value);

  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_FLOAT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";
  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_int_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<int64_t>(1);
  op->SetAttr("x1", attr_value);

  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_INT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";
  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_bool_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<bool>(true);
  op->SetAttr("x1", attr_value);

  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_BOOL;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_listint_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<std::vector<int64_t>>({1, 2});
  op->SetAttr("x1", attr_value);

  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_INT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_failed6) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_TENSOR;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_failed7) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_TENSOR;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_failed8) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_failed9) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_LIST_INT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  vector<vector<int64_t> > vecvec_int_value1;
  vector<int64_t> vec_b;
  vector<int64_t> vec_c;
  vec_b.push_back(0);
  vec_b.push_back(1);
  vec_c.push_back(2);
  vec_c.push_back(4);
  vecvec_int_value1.push_back(vec_b);
  vecvec_int_value1.push_back(vec_c);
  ge::AttrUtils::SetListListInt(*(op.get()), "x1", vecvec_int_value1);
  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_listfloat_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<std::vector<float>>({1.1, 1.2});
  op->SetAttr("x1", attr_value);
  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_FLOAT;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_listbool_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<std::vector<bool>>({true, false});
  op->SetAttr("x1", attr_value);

  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_BOOL;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_feed_attrs_to_tbe_opinfo_liststr_success) {
  OpDescPtr op = std::make_shared<OpDesc>("conv", "conv");
  ge::GeAttrValue attr_value;
  attr_value.SetValue<std::vector<std::string>>({"abc", "def"});
  op->SetAttr("x1", attr_value);
  std::shared_ptr<OpKernelInfo> op_kernel_info_ptr = make_shared<OpKernelInfo>("x1");
  std::shared_ptr<AttrInfo> attr_info_ptr = make_shared<AttrInfo>("x1");
  attr_info_ptr->attr_name_ = "x1";
  attr_info_ptr->dtype_ = ge::GeAttrValue::ValueType::VT_LIST_STRING;
  attr_info_ptr->is_required_ = false;
  attr_info_ptr->is_default_value_defined_ = false;
  op_kernel_info_ptr->attrs_info_.push_back(attr_info_ptr);

  string op_name = "conv";
  string op_dsl_file_path = GetCodeDir() + "/tests/engines/nn_engine/stub/cce_reductionLayer_1_10_float16__1_SUMSQ_1_0.o";
  string opFuncName = "tbe";

  TbeOpInfo op_info(op_name, op_dsl_file_path, opFuncName, AI_CORE_NAME);

  TbeInfoAssembler feed_attrs_to_tbe_op_info;
  Status ret = feed_attrs_to_tbe_op_info.FeedAttrsToTbeOpInfo(*(op.get()), op_kernel_info_ptr, op_info);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_parallel_compiler_success) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusion = TeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);

  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);

  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  compile_tbe_op.support_parallel_compile = true;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_parallel_compiler_unknown_shape_success) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub4;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, -1, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, -1, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  ge::AttrUtils::SetInt(weight_op_desc1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(weight_op_desc2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  compile_tbe_op.support_parallel_compile = false;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_parallel_compiler_unknown_shape_success_not_support) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub5;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, -1, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, -1, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  ge::AttrUtils::SetInt(weight_op_desc1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  ge::AttrUtils::SetInt(weight_op_desc2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  compile_tbe_op.support_parallel_compile = false;
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TbeOpInfoPtr PreCompSetTbeOpInfoStub(TbeOpStoreAdapter *This, PreCompileNodePara &comp_para) {
  TbeOpInfoPtr tbe_op_info_ptr = make_shared<te::TbeOpInfo>("", "", "", "");
  return tbe_op_info_ptr;
}

TEST_F(STEST_FE_TBE_COMPILER, case_get_buffer_optimize_rollback_node_fail) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node1);
  buff_fus_compile_failed_nodes.push_back(Node2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(2, 1));
  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  vector<ge::NodePtr> l1_failed_nodes;
  Status ret = compile_tbe_op.ProcessLxFusionFailCompileTasks(task_para, l1_failed_nodes,
                                                              buff_fus_compile_failed_nodes);
  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_get_buffer_optimize_rollback_node_suc) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);

  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node1);
  buff_fus_compile_failed_nodes.push_back(Node2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));

  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  vector<ge::NodePtr> l1_failed_nodes;
  Status ret = compile_tbe_op.ProcessLxFusionFailCompileTasks(task_para, l1_failed_nodes, buff_fus_compile_failed_nodes);
  //3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_set_sgt_tensor_slice_info_to_nodes_fail) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);
  std::vector<ge::Node *> vector_node_ptr;
  for (auto &node: graph->GetDirectNode()) {
    auto op_desc_ptr = node->GetOpDesc();

    if (op_desc_ptr->GetName() == "relu") {
      vector_node_ptr.emplace_back(node.get());
    }
  }

  Status ret = compile_tbe_op.SetSgtTensorSliceInfoToNodes(vector_node_ptr, 0);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_set_sgt_tensor_slice_info_to_nodes_suc) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);
  std::vector<ge::Node *> vector_node_ptr;
  ffts::ThreadSliceMapPtr tsmp_ptr;
  for (auto &node: graph->GetDirectNode()) {
    auto op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr->GetName() == "relu") {
      AttrUtils::SetInt(node->GetOpDesc(), kThreadScopeId, 1);
      ffts::ThreadSliceMap subgraphInfo;
      vector<vector<vector<ffts::DimRange>>> inputTensorSlice;
      vector<vector<vector<ffts::DimRange>>> oriInputTensorSlice;
      for (size_t i = 0; i < 2; i++) {
        vector<int64_t> vec1 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr1;
        for (size_t j = 0; j < vec1.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec1[j];
          dr.higher = vec1[j + 1];
          vdr1.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr1);
        vector<int64_t> vec2 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr2;
        for (size_t j = 0; j < vec2.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec2[j];
          dr.higher = vec2[j + 1];
          vdr2.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> oriThreadSlice;
        oriThreadSlice.push_back(vdr2);
        inputTensorSlice.push_back(threadSlice);
        oriInputTensorSlice.push_back(oriThreadSlice);
      }

      vector<vector<vector<ffts::DimRange>>> outputTensorSlice;
      vector<vector<vector<ffts::DimRange>>> oriOutputTensorSlice;
      for (size_t i = 0; i < 2; i++) {
        vector<int64_t> vec3 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr3;
        for (size_t j = 0; j < vec3.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec3[j];
          dr.higher = vec3[j + 1];
          vdr3.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr3);
        vector<int64_t> vec4 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr4;
        for (size_t j = 0; j < vec4.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec4[j];
          dr.higher = vec4[j + 1];
          vdr4.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> oriThreadSlice;
        oriThreadSlice.push_back(vdr4);
        outputTensorSlice.push_back(threadSlice);
        oriOutputTensorSlice.push_back(oriThreadSlice);
      }

      subgraphInfo.input_tensor_slice = inputTensorSlice;
      subgraphInfo.ori_input_tensor_slice = oriInputTensorSlice;
      subgraphInfo.output_tensor_slice = outputTensorSlice;
      subgraphInfo.ori_output_tensor_slice = oriOutputTensorSlice;
      tsmp_ptr = make_shared<ffts::ThreadSliceMap>(subgraphInfo);
      op_desc_ptr->SetExtAttr(ffts::kAttrSgtStructInfo, tsmp_ptr);
      vector_node_ptr.emplace_back(node.get());
    }
  }

  Status ret = compile_tbe_op.SetSgtTensorSliceInfoToNodes(vector_node_ptr, 0);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_set_sgt_slice_task_to_te_fusion_suc) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusionV = SgtTeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);
  vector<ge::NodePtr> to_del_nodes;
  std::vector<ge::Node *> vector_node_ptr;
  ScopeNodeIdMap fusion_nodes_map;
  for (auto &node: graph->GetDirectNode()) {
    auto op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr->GetName() == "relu") {
      AttrUtils::SetInt(node->GetOpDesc(), kThreadScopeId, 1);
      ffts::ThreadSliceMap subgraphInfo;
      vector<vector<vector<ffts::DimRange>>> inputTensorSlice;
      vector<vector<vector<ffts::DimRange>>> oriInputTensorSlice;
      for (size_t i = 0; i < 2; i++) {
        vector<int64_t> vec1 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr1;
        for (size_t j = 0; j < vec1.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec1[j];
          dr.higher = vec1[j + 1];
          vdr1.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr1);
        vector<int64_t> vec2 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr2;
        for (size_t j = 0; j < vec2.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec2[j];
          dr.higher = vec2[j + 1];
          vdr2.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> oriThreadSlice;
        oriThreadSlice.push_back(vdr2);
        inputTensorSlice.push_back(threadSlice);
        oriInputTensorSlice.push_back(oriThreadSlice);
      }

      vector<vector<vector<ffts::DimRange>>> outputTensorSlice;
      vector<vector<vector<ffts::DimRange>>> oriOutputTensorSlice;
      for (size_t i = 0; i < 2; i++) {
        vector<int64_t> vec3 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr3;
        for (size_t j = 0; j < vec3.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec3[j];
          dr.higher = vec3[j + 1];
          vdr3.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr3);
        vector<int64_t> vec4 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr4;
        for (size_t j = 0; j < vec4.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec4[j];
          dr.higher = vec4[j + 1];
          vdr4.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> oriThreadSlice;
        oriThreadSlice.push_back(vdr4);
        outputTensorSlice.push_back(threadSlice);
        oriOutputTensorSlice.push_back(oriThreadSlice);
      }

      subgraphInfo.input_tensor_slice = inputTensorSlice;
      subgraphInfo.ori_input_tensor_slice = oriInputTensorSlice;
      subgraphInfo.output_tensor_slice = outputTensorSlice;
      subgraphInfo.ori_output_tensor_slice = oriOutputTensorSlice;
      subgraphInfo.thread_mode = 1;
      ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(subgraphInfo);
      node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
      vector_node_ptr.emplace_back(node.get());
    }
  }

  TbeOpStoreAdapter::CompileTaskPara task_para;
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  task_para.task_num = 2;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  task_para.task_scope_id_map.insert(make_pair(2, 1));
  vector<uint64_t> stim = {1, 2};
  task_para.scope_task_ids_map[1] = stim;
  te::FinComTask succ_tasks;
  succ_tasks.taskId = 1;
  succ_tasks.graphId = 996;
  task_para.succ_tasks[succ_tasks.taskId] = succ_tasks;
  Status ret = compile_tbe_op.SetSgtSliceTaskToTeFusion(task_para, to_del_nodes);

  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_set_sgt_slice_task_to_te_fusion_fail_1) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusionV = SgtTeFusionStub2;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  CreateGraph(graph);

  vector<ge::NodePtr> to_del_nodes;
  std::vector<ge::Node *> vector_node_ptr;
  ScopeNodeIdMap fusion_nodes_map;
  for (auto &node: graph->GetDirectNode()) {
    auto op_desc_ptr = node->GetOpDesc();

    if (op_desc_ptr->GetName() == "relu") {
      AttrUtils::SetInt(node->GetOpDesc(), kThreadScopeId, 1);
      ffts::ThreadSliceMap subgraphInfo;
      vector<vector<vector<ffts::DimRange>>> inputTensorSlice;
      vector<vector<vector<ffts::DimRange>>> oriInputTensorSlice;
      for (size_t i = 0; i < 2; i++) {
        vector<int64_t> vec1 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr1;
        for (size_t j = 0; j < vec1.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec1[j];
          dr.higher = vec1[j + 1];
          vdr1.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr1);

        vector<int64_t> vec2 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr2;
        for (size_t j = 0; j < vec2.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec2[j];
          dr.higher = vec2[j + 1];
          vdr2.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> oriThreadSlice;
        oriThreadSlice.push_back(vdr2);
        inputTensorSlice.push_back(threadSlice);
        oriInputTensorSlice.push_back(oriThreadSlice);
      }

      vector<vector<vector<ffts::DimRange>>> outputTensorSlice;
      vector<vector<vector<ffts::DimRange>>> oriOutputTensorSlice;
      for (size_t i = 0; i < 2; i++) {
        vector<int64_t> vec3 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr3;
        for (size_t j = 0; j < vec3.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec3[j];
          dr.higher = vec3[j + 1];
          vdr3.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> threadSlice;
        threadSlice.push_back(vdr3);

        vector<int64_t> vec4 = {0, 288, 0, 32, 0, 16, 0, 16};
        vector<ffts::DimRange> vdr4;
        for (size_t j = 0; j < vec4.size() - 1;) {
          ffts::DimRange dr;
          dr.lower = vec4[j];
          dr.higher = vec4[j + 1];
          vdr4.push_back(dr);
          j = j + 2;
        }
        vector<vector<ffts::DimRange>> oriThreadSlice;
        oriThreadSlice.push_back(vdr4);
        outputTensorSlice.push_back(threadSlice);
        oriOutputTensorSlice.push_back(oriThreadSlice);
      }

      subgraphInfo.input_tensor_slice = inputTensorSlice;
      subgraphInfo.ori_input_tensor_slice = oriInputTensorSlice;
      subgraphInfo.output_tensor_slice = outputTensorSlice;
      subgraphInfo.ori_output_tensor_slice = oriOutputTensorSlice;
      subgraphInfo.thread_mode = 1;
      ffts::ThreadSliceMapPtr tsmp_ptr = make_shared<ffts::ThreadSliceMap>(subgraphInfo);
      node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
      vector_node_ptr.emplace_back(node.get());
    }
  }

  TbeOpStoreAdapter::CompileTaskPara task_para;
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;

  task_para.task_num = 2;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  task_para.task_scope_id_map.insert(make_pair(2, 1));
  vector<uint64_t> stim = {1, 2};
  task_para.scope_task_ids_map[1] = stim;

  te::FinComTask succ_tasks;
  succ_tasks.taskId = 1;
  succ_tasks.graphId = 996;
  task_para.succ_tasks[succ_tasks.taskId] = succ_tasks;
  Status ret = compile_tbe_op.SetSgtSliceTaskToTeFusion(task_para, to_del_nodes);
  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_process_suc_sgt_slice_task) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  CreateGraph(graph);

  vector<ge::NodePtr> to_del_nodes;
  std::vector<ge::Node *> vector_node_ptr;
  ScopeNodeIdMap fusion_nodes_map;
  te::FinComTask succ_tasks;
  succ_tasks.taskId = 1;
  succ_tasks.graphId = 996;
  for (auto &node: graph->GetDirectNode()) {
    auto op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr->GetName() == "relu") {
      vector_node_ptr.emplace_back(node.get());
      succ_tasks.teNodeOpDesc = op_desc_ptr;
    }
  }

  TbeOpStoreAdapter::CompileTaskPara task_para;
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;

  task_para.task_num = 2;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  task_para.task_scope_id_map.insert(make_pair(2, 1));
  vector<uint64_t> stim = {1, 2};
  task_para.scope_task_ids_map[1] = stim;
  task_para.succ_tasks[succ_tasks.taskId] = succ_tasks;

  Status ret = compile_tbe_op.ProcessSuccSgtSliceTask(task_para);
  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, save_ms_tune_error_msg) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  Status ret = fe::SUCCESS;
  compile_tbe_op.SaveMsTuneErrorMsg(task_para);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_retry_compile_fail_op_fail_1) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));

  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  Status ret = compile_tbe_op.RetryCompileFailOp(task_para);
  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_process_fail_pre_compile) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);

  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));

  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  Status ret = compile_tbe_op.ProcessFailPreCompTask(task_para);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_process_fail_pre_compile_1) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);

  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  task_para.task_node_map.insert(make_pair(1, Node2.get()));

  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  Status ret = compile_tbe_op.ProcessFailPreCompTask(task_para);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_process_fail_compile) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));

  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;
  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;

  Status ret = compile_tbe_op.ProcessFailCompileTask(task_para, CompileStrategy::COMPILE_STRATEGY_OP_SPEC);
  //3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_get_sgt_slice_task_roll_back_node_fail_1) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  vector<ge::NodePtr> need_rollback_nodes;
  need_rollback_nodes.push_back(Node1);
  need_rollback_nodes.push_back(Node2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(2, 1));
  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  Status ret = compile_tbe_op.GetSgtSliceTaskRollbackNode(task_para, need_rollback_nodes);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_get_sgt_slice_task_roll_back_node_suc_1) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  vector<ge::NodePtr> need_rollback_nodes;
  need_rollback_nodes.push_back(Node1);
  need_rollback_nodes.push_back(Node2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;

  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;
  Status ret = compile_tbe_op.GetSgtSliceTaskRollbackNode(task_para, need_rollback_nodes);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_get_sgt_slice_task_roll_back_node_suc_2) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  (void) ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  vector<ge::NodePtr> need_rollback_nodes;
  need_rollback_nodes.push_back(Node1);
  need_rollback_nodes.push_back(Node2);

  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_scope_id_map.insert(make_pair(1, 1));
  te::FinComTask failed_tasks;
  failed_tasks.taskId = 1;
  failed_tasks.graphId = 996;
  task_para.failed_tasks[failed_tasks.taskId] = failed_tasks;
  vector<uint64_t> stim = {1, 2};
  task_para.scope_task_ids_map[1] = stim;
  ScopeNodeIdMap fusion_nodes_map;
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node2.get());
  fusion_nodes_map.insert(std::make_pair(1, vector_node_ptr));
  task_para.fusion_nodes_map = &fusion_nodes_map;

  Status ret = compile_tbe_op.GetSgtSliceTaskRollbackNode(task_para, need_rollback_nodes);
  EXPECT_EQ(fe::SUCCESS, ret);
}

Status SetTeTaskStub(TbeOpStoreAdapter *This, std::vector<ge::Node *> &node_vec,
                     TbeOpStoreAdapter::CompileTaskPara &task_para, uint64_t taskId,
                     std::vector<ge::NodePtr> &l1_to_del_nodes) {
  vector<uint64_t> taskid;
  for (auto iter: task_para.task_scope_id_map) {
    taskid.push_back(iter.first);
  }
  te::FinComTask fin_com_task;
  fin_com_task.taskId = taskid[0];
  fin_com_task.graphId = 996;
  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "jsonFilePath");

  task_para.failed_tasks[fin_com_task.taskId] = fin_com_task;
  return fe::SUCCESS;
}

TEST_F(STEST_FE_TBE_COMPILER, tbe_op_process_succ_pre_comp_task) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);

  // 1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  ge::AttrUtils::SetBool(weight_op_desc2, NEED_RE_PRECOMPILE, true);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  TbeOpInfoPtr tbe_op_info_ptr = make_shared<te::TbeOpInfo>("", "", "", "");
  tbe_op_info_ptr->SetPattern("w2Pattern");
  TbeOpStoreAdapter::CompileTaskPara task_para;
  task_para.task_num = 1;
  task_para.task_node_map.insert(make_pair(1, Node2.get()));
  task_para.task_tbe_info_map.insert(make_pair(1, tbe_op_info_ptr));
  te::FinComTask fin_com_task;
  fin_com_task.taskId = 1;
  fin_com_task.graphId = 996;
  task_para.succ_tasks[fin_com_task.taskId] = fin_com_task;
  OpKernelInfoPtr op_kernel_info = std::make_shared<OpKernelInfo>("test");
  task_para.task_kernel_info_map.insert(make_pair(1, op_kernel_info));

  Status ret = compile_tbe_op.ProcessSuccPreCompTask(task_para);
  // 3. result expected
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_op_compiler_failed_error_message_report) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = false;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  // 1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);

  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  // 3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_op_compiler_parallel_failed_error_message_report) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;
  ScopeNodeIdMap fusion_nodes_map;

  // 1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);


  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);

  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);

  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);

  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);

  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  fusion_nodes_map.insert(std::make_pair(0, vector_node_ptr));

  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  std::vector<ge::NodePtr> compile_failed_nodes;
  std::vector<ge::NodePtr> to_del_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes_map;
  compile_info.compile_ret_map = compile_ret_map;
  compile_info.buff_fus_to_del_nodes = to_del_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);

  // 3. result expected
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_compile_op) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  ScopeNodeIdMap fusion_nodes;
  fusion_nodes.insert(std::make_pair(0, vector_node_ptr));
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node2);
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes;
  Status ret = compile_tbe_op.CompileOp(compile_info);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_compile_op_multi_slice) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  std::vector<ge::Node *> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());
  ge::AttrUtils::SetInt(Node1->GetOpDesc(), kThreadMode, 1);

  ScopeNodeIdMap fusion_nodes;
  fusion_nodes.insert(std::make_pair(0, vector_node_ptr));
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node2);
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes;

  PlatformUtils::Instance().soc_version_ = "Ascend910B1";
  Status ret = compile_tbe_op.CompileOp(compile_info);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_set_sgt_json_path_fail) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  ge::OpDescPtr compile_op_desc = make_shared<ge::OpDesc>();
  string json_file_path;
  ge::AttrUtils::SetStr(compile_op_desc, "json_file_path", json_file_path);
  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  int scope_idx = 1;

  Status ret = compile_tbe_op.SetOpCompileResult(scope_idx, compile_op_desc, false, compile_ret_map);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_set_sgt_json_path_suc_1) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  ge::OpDescPtr compile_op_desc = make_shared<ge::OpDesc>();
  string json_file_path = "xxxx1";
  ge::AttrUtils::SetStr(compile_op_desc, "json_file_path", json_file_path);
  CompileResultMap compile_ret_map;
  CompileResultInfo com_ret_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {com_ret_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));
  int scope_idx = 1;

  Status ret = compile_tbe_op.SetOpCompileResult(scope_idx, compile_op_desc, false, compile_ret_map);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_tbe_set_sgt_json_path_suc_2) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub3;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  ge::OpDescPtr compile_op_desc = make_shared<ge::OpDesc>();
  string json_file_path = "xxxx1";
  ge::AttrUtils::SetStr(compile_op_desc, "json_file_path", json_file_path);
  CompileResultMap compile_ret_map;
  CompileResultInfo compile_info("xxxx1");
  std::vector<CompileResultInfo> compile_infos = {compile_info};
  compile_ret_map.emplace(make_pair(1, compile_infos));

  Status ret = compile_tbe_op.SetOpCompileResult(2, compile_op_desc, false, compile_ret_map);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_1) {
  OpDescPtr matmul_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  vector<int64_t> dim_data = {1, 3, 5, 5};
  GeShape shape_data(dim_data);
  GeTensorDesc data_desc(shape_data, FORMAT_NHWC, DT_FLOAT);
  matmul_desc->AddInputDesc("x1", data_desc);
  matmul_desc->AddInputDesc("x2", data_desc);
  matmul_desc->AddOutputDesc("y", data_desc);

  FEOpsStoreInfo tbe_opinfo{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_opinfo",
      "",
      false,
      false,
      false
  };
  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(tbe_opinfo);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
  OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
  shared_ptr<fe::SubOpsStore> sub_ops_store_ptr = make_shared<fe::SubOpsStore>(AI_CORE_NAME);

  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpType(
      tbe_opinfo.fe_ops_store_name, matmul_desc->GetType());
  op_kernel_info_ptr->op_flag_vec_[static_cast<size_t>(OP_KERNEL_FLAG::NeedCheckSupport)] = true;
  op_kernel_info_ptr->impl_type_ = EN_RESERVED;

  OpStoreAdapterPtr tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(AI_CORE_NAME);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::NodePtr test_node = graph->AddNode(matmul_desc);
  CheckSupportParam check_param;
  check_param.op_kernel_ptr = op_kernel_info_ptr;
  bool result = sub_ops_store_ptr->CheckOpSupported(test_node, false, check_param);

  EXPECT_EQ(result, false);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_2) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  string attr_name = "test";
  string attr_value = "1";
  AttrInfoPtr attrs_info = nullptr;
  te::TbeAttrValue tbe_attr_value;

  EXPECT_EQ(GetStrAttrValue(*op_desc, attr_name, tbe_attr_value, attrs_info), fe::FAILED);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_3) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  string attr_name = "test";
  int64_t attr_value = 1;
  AttrInfoPtr attrs_info = nullptr;
  te::TbeAttrValue tbe_attr_value;

  EXPECT_EQ(GetIntAttrValue(*op_desc, attr_name, tbe_attr_value, attrs_info), fe::FAILED);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_4) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  string attr_name = "test";
  float attr_value = 1.0;
  AttrInfoPtr attrs_info = nullptr;
  te::TbeAttrValue tbe_attr_value;

  EXPECT_EQ(GetFloatAttrValue(*op_desc, attr_name, tbe_attr_value, attrs_info), fe::FAILED);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_5) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  string attr_name = "test";
  bool attr_value = false;
  AttrInfoPtr attrs_info = nullptr;
  te::TbeAttrValue tbe_attr_value;

  EXPECT_EQ(GetBoolAttrValue(*op_desc, attr_name, tbe_attr_value, attrs_info), fe::FAILED);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_6) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  string attr_name = "test";
  vector<float> attr_value = {1.0};
  AttrInfoPtr attrs_info = nullptr;
  te::TbeAttrValue tbe_attr_value;

  EXPECT_EQ(GetListFloatAttrValue(*op_desc, attr_name, tbe_attr_value, attrs_info), fe::FAILED);
}

TEST_F(STEST_FE_TBE_COMPILER, converage_7) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("matmul", "MatMul");
  string attr_name = "test";
  vector<bool> attr_value = {false};
  AttrInfoPtr attrs_info = nullptr;
  te::TbeAttrValue tbe_attr_value;

  EXPECT_EQ(GetListBoolAttrValue(*op_desc, attr_name, tbe_attr_value, attrs_info), fe::FAILED);
}

std::map<uint64_t, bool> g_task_map;
uint64_t g_compile_count = 0;
te::OpBuildResCode CompileStub(std::vector<Node *> teGraphNode, OpDescPtr op_desc_ptr,
                               const std::vector<ge::NodePtr> &to_be_del, uint64_t taskid,
                               uint64_t graphId, const std::string op_compile_strategy) {
  bool is_l1_fusion = false;
  if (!teGraphNode.empty()) {
    is_l1_fusion = teGraphNode[0]->GetOpDesc()->HasAttr(L1_SCOPE_ID_ATTR);
  }
  g_task_map.emplace(taskid, is_l1_fusion);
  g_compile_count++;
  return te::OP_BUILD_SUCC;
}

bool WaitAllFinishedFailTaskStub(uint64_t tid, vector<te::FinComTask> &fin_task) {
  std::string json_path = GetCodeDir() + "/tests/engines/nn_engine/stub/te_op_info.json";
  for (auto &item : g_task_map) {
    te::FinComTask fin_com_task;
    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
    fin_com_task.taskId = item.first;
    fin_com_task.status = item.second ? fe::FAILED : fe::SUCCESS;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", json_path);
    fin_task.push_back(fin_com_task);
  }
  g_task_map.clear();
  return true;
}

bool WaitAllFinishedSuccTaskStub(uint64_t tid, vector<te::FinComTask> &fin_task) {
  std::string json_path = GetCodeDir() + "/tests/engines/nn_engine/stub/te_op_info.json";
  for (auto &item : g_task_map) {
    te::FinComTask fin_com_task;
    fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
    fin_com_task.taskId = item.first;
    fin_com_task.status = fe::SUCCESS;
    ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", json_path);
    fin_task.push_back(fin_com_task);
  }
  g_task_map.clear();
  return true;
}

TEST_F(STEST_FE_TBE_COMPILER, case_fusion_compile_check_success) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  std::vector<ge::Node*> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  ScopeNodeIdMap fusion_nodes;
  fusion_nodes.insert(std::make_pair(0, vector_node_ptr));
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node2);
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes;
  compile_info.is_fusion_check = true;
  Status ret = compile_tbe_op.CompileOp(compile_info);
  for (auto iter : compile_info.fusion_nodes_map) {
    for (auto node : iter.second) {
      auto ret = node->GetOpDesc()->HasAttr("_only_fusion_check");
      EXPECT_EQ(false, ret);
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(STEST_FE_TBE_COMPILER, case_fusion_compile_check_fail) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  compile_tbe_op.support_parallel_compile = true;
  compile_tbe_op.TeFusion = TeFusionStub;
  compile_tbe_op.WaitAllFinished = WaitAllFinishedStub1;
  compile_tbe_op.GetOpInfo = get_tbe_opinfo_stub;

  //1.create graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  vector<int64_t> dim_weight1 = {1, 3, 3, 3};
  GeShape shape_weight1(dim_weight1);
  GeTensorDesc weight_desc1(shape_weight1);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  OpDescPtr weight_op_desc2 = std::make_shared<OpDesc>("w2", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  weight_op_desc2->AddOutputDesc(weight_desc1);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  NodePtr Node2 = graph->AddNode(weight_op_desc2);
  ge::AttrUtils::SetStr(Node1->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  ge::AttrUtils::SetStr(Node2->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
  (void)ge::AttrUtils::SetInt(Node1->GetOpDesc(), SCOPE_ID_ATTR, 1);
  (void)ge::AttrUtils::SetInt(Node2->GetOpDesc(), SCOPE_ID_ATTR, 1);
  std::vector<ge::Node*> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1.get());
  vector_node_ptr.emplace_back(Node2.get());

  ScopeNodeIdMap fusion_nodes;
  fusion_nodes.insert(std::make_pair(0, vector_node_ptr));
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  buff_fus_compile_failed_nodes.push_back(Node2);
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.fusion_nodes_map = fusion_nodes;
  compile_info.is_fusion_check = true;
  auto ret = compile_tbe_op.CompileOp(compile_info);
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto iter : compile_info.fusion_nodes_map) {
    for (auto node : iter.second) {
      auto res = node->GetOpDesc()->HasAttr("_only_fusion_check");
      EXPECT_EQ(false, res);
      res = node->GetOpDesc()->HasAttr(SCOPE_ID_ATTR);
      EXPECT_EQ(false, res);
    }
  }
}

TEST_F(STEST_FE_TBE_COMPILER, test_set_opdesc_custom_op_fail) {
  TbeOpStoreAdapter compile_tbe_op(AI_CORE_NAME);
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Test", "Test");
  compile_tbe_op.SetOpDescCustomOp(op_desc);
}
}
