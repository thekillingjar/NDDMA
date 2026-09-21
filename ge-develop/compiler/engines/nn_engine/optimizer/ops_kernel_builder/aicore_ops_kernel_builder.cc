/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/aicore_ops_kernel_builder.h"

#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/op_tensor_utils.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_constants.h"
#include "common/scope_allocator.h"
#include "common/math/math_util.h"
#include "common/util/op_info_util.h"
#include "common/fe_gentask_utils.h"
#include "common/fe_op_info_common.h"
#include "param_calculate/tensorsize_calculator.h"
#include "ops_kernel_builder/task_builder/task_builder.h"
#include "ops_kernel_builder/task_builder/ffts_task_builder.h"
#include "ops_kernel_builder/task_builder/superkernel_task_builder.h"
#include "inc/ffts_param_calculator.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "register/ops_kernel_builder_registry.h"
#include "register/op_ext_gentask_registry.h"
#include "register/op_ext_calc_param_registry.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "framework/common/types.h"
#include "exe_graph/runtime/exe_res_generation_context.h"

namespace fe {
namespace {
const int64_t UINT64_BITS = 64;
const int64_t CUSTOM_TILING_OP_DUMP_SIZE = 20480;

std::string GetInputDependencyIndexs(const uint64_t dependency_num, vector<int64_t> &indexs) {
  std::string indexs_str = "";
  for (int i = 0; i < UINT64_BITS; i++) {
    if ((dependency_num & (1ULL << i)) != 0) {
      indexs.push_back(i);
      if (indexs_str != "") {
        indexs_str += ",";
      }
      indexs_str += std::to_string(i);
    }
  }
  return indexs_str;
}

Status SetTilingSinkCalcResources(const ge::Node &node, gert::ExeResGenerationContext* context) {
  const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
    static_cast<gert::OppImplVersionTag>(node.GetOpDesc()->GetOppImplVersion()));
  if (space_registry == nullptr) {
    FE_LOGE("Node[%s] type[%s] get space registry failed", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  auto funcs_ptr = space_registry->GetOpImpl(node.GetType().c_str());
  if (funcs_ptr == nullptr) {
    FE_LOGE("Node[%s] type[%s] Registry funcs_ptr is nullptr", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  if (funcs_ptr->tiling_dependency == 0) {
    FE_LOGE("Node[%s] type[%s] Registry tiling_dependency is 0", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  gert::StreamInfo stream_info;
  stream_info.name = "tiling";
  vector<int64_t> stream_depend_value_list(0);
  std::string indexs_str = GetInputDependencyIndexs(funcs_ptr->tiling_dependency, stream_depend_value_list);
  FE_LOGD("Node[%s] type[%s] tiling sink dependency indexs[%s]", node.GetNamePtr(), node.GetTypePtr(),
          indexs_str.c_str());
  stream_info.depend_value_input_indices = stream_depend_value_list;
  std::vector<gert::StreamInfo> stream_info_vec(0);
  stream_info_vec.push_back(stream_info);
  context->SetAttachedStreamInfos(stream_info_vec);

  gert::SyncResInfo sync_res_info;
  sync_res_info.type = gert::SyncResType::SYNC_RES_EVENT;
  sync_res_info.name = "tiling";
  std::vector<gert::SyncResInfo> sync_info_vec(0);
  sync_info_vec.push_back(sync_res_info);
  context->SetSyncResInfos(sync_info_vec);
  return SUCCESS;
}

Status CalculateTilingSinkWorkspace(ge::Node &node, gert::ExeResGenerationContext* context) {
  const ge::NodePtr node_ptr = node.shared_from_this();
  if (TilingForOneNode(node_ptr) != SUCCESS) {
    FE_LOGE("Node[%s] type[%s] tiling sink workspace failed.", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  ge::OpDescPtr op_desc = node.GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  auto workspace_bytes = op_desc->GetWorkspaceBytes();
  FE_LOGI("Node[%s] type[%s] tiling sink workspace size is [%ld].", node.GetNamePtr(), node.GetTypePtr(),
          workspace_bytes[0]);

  std::string prefix = "";
  if (IsCustomiseOp(*op_desc) || IsPrefixOpsPath(*op_desc, prefix)) {
    std::vector<uint32_t> mem_type;
    for (size_t i = 0; i < workspace_bytes.size(); ++i) {
      mem_type.emplace_back(static_cast<uint32_t>(ge::AicpuWorkSpaceType::INVALID_TYPE));
      FE_LOGI("Node[%s] type[%s] tiling sink op workspace[%lu]=%ld", node.GetNamePtr(), node.GetTypePtr(), i,
              workspace_bytes[i]);
    }
    workspace_bytes.emplace_back(CUSTOM_TILING_OP_DUMP_SIZE);
    mem_type.emplace_back(static_cast<uint32_t>(ge::AicpuWorkSpaceType::CUST_LOG));
    (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, mem_type);

    std::vector<int32_t> tvm_workspace_types;
    (void)ge::AttrUtils::GetListInt(op_desc, ge::TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);
    tvm_workspace_types.emplace_back(RT_MEMORY_HBM);
    (void)ge::AttrUtils::SetListInt(op_desc, ge::TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);
  }
  context->SetWorkspaceBytes(workspace_bytes);
  return SUCCESS;
}
} // namespace

REGISTER_OPS_KERNEL_BUILDER(AI_CORE_NAME, AICoreOpsKernelBuilder);
REGISTER_OPS_KERNEL_BUILDER(VECTOR_CORE_NAME, AICoreOpsKernelBuilder);

AICoreOpsKernelBuilder::AICoreOpsKernelBuilder() {}

AICoreOpsKernelBuilder::~AICoreOpsKernelBuilder() {}

Status AICoreOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  return SUCCESS;
}

Status AICoreOpsKernelBuilder::Finalize() { return SUCCESS; }

// ascendc dfx expand first workspace size
void ProcDfxBufferSize(const ge::OpDescPtr op_desc) {
  int64_t buffer_size = 0;
  if (!ge::AttrUtils::GetInt(op_desc, kOpDfxBufferSize, buffer_size)) {
    return;
  }
  auto new_workspaces = op_desc->GetWorkspaceBytes();
  if (new_workspaces.empty()) {
    FE_LOGW("Op[%s] need dfx buffer but not has workspace.", op_desc->GetNamePtr());
    return;
  }
  if (ge::CheckInt64AddOverflow(new_workspaces[0], buffer_size) != SUCCESS) {
    FE_LOGW("Op[%s] workspace size over flow int64.", op_desc->GetNamePtr());
    return;
  }
  FE_LOGW("Op[%s] workspace[0] size [%ld] expand with [%ld].", op_desc->GetNamePtr(), new_workspaces[0], buffer_size);
  new_workspaces[0] = new_workspaces[0] + buffer_size;
  op_desc->SetWorkspaceBytes(new_workspaces);
  return;
}

static Status CalcAutoThreadParam(const ge::NodePtr nodePtr) {
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = nodePtr->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  if (slice_info_ptr == nullptr) {
    FE_LOGI("[GenTask][CalcAutoThreadParam] There must have slice info in Node: %s", nodePtr->GetName().c_str());
    return SUCCESS;
  }

  if (slice_info_ptr->thread_mode == 0) {
    FE_LOGI("[GenTask][CalcAutoThreadParam] Node: %s This has to be auto thread mode", nodePtr->GetName().c_str());
    return SUCCESS;
  }

  if (slice_info_ptr->slice_instance_num == 0) {
    REPORT_FE_ERROR("[GenTask][CalcAutoThreadParam] Node: %s slice num is zero.",
                    nodePtr->GetName().c_str());
    return FAILED;
  }
  vector<vector<vector<int64_t>>> tensor_slice;
  vector<uint64_t> task_addr_offset;
  ffts::ParamCalculator::ConvertTensorSlice(slice_info_ptr->output_tensor_slice, tensor_slice);
  if (ffts::ParamCalculator::CalcAutoThreadOutput(nodePtr, tensor_slice, task_addr_offset,
                                                  slice_info_ptr->output_tensor_indexes) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CalcAutoThreadParam] Node: %s calculate thread output failed.",
                    nodePtr->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status AICoreOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  FE_LOGD("Begin to calculate running parameters of op [%s]", op_desc_ptr->GetName().c_str());

  if (TensorSizeCalculator::CalculateOpTensorSize(node.shared_from_this()) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CalcOpRunningParam][Node %s, type %s] Failed to calculate op tensor size.",
                    op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return FAILED;
  }

  if (node.GetOpDesc()->HasAttr(ge::ATTR_NAME_UNREGST_OPPATH)) {
    if (!ge::AttrUtils::SetInt(op_desc_ptr, FE_IMPLY_TYPE, EN_IMPL_HW_TBE)) {
      REPORT_FE_ERROR("[GenTask][CalcOpRunningParam][Node %s, type %s] Failed to set fe_impl_type!",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
  }
  if (op_desc_ptr->GetType() == OP_TYPE_ROIPOOLING || op_desc_ptr->GetType() == OP_TYPE_SSD_DETECTION_OUTPUT) {
    bool has_reuse_mem_attr = true;
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kLxNoReuseMemFlag, has_reuse_mem_attr);
    FE_LOGD("op:%s set no_reuse_mem_flag true", op_desc_ptr->GetName().c_str());
  }

  ge::NodePtr nodePtr = node.shared_from_this();
  FE_LOGD("[GenTask][CalcOpRunningParam] Node:%s start", nodePtr->GetName().c_str());
  ge::OpDescPtr opDescPtr = nodePtr->GetOpDesc();
  FE_CHECK_NOTNULL(opDescPtr);
  if (UnknownShapeUtils::IsUnknownShapeOp(*opDescPtr)) {
    FE_LOGI("[GenTask][CalcOpRunningParam] Node: %s is unknown shape, This has to be static shape",
            nodePtr->GetName().c_str());
    return SUCCESS;
  }
  if (CalcExtOpRunningParam(node) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CalcOpRunningParam] CalcExtOpRunningParam failed.");
    return FAILED;
  }
  ProcDfxBufferSize(opDescPtr);
  return CalcAutoThreadParam(nodePtr);
}

Status AICoreOpsKernelBuilder::CalcTilingSinkRunningParam(const bool is_tiling_sink, ge::Node &node, bool &proc_flag) {
  if (!is_tiling_sink) {
    FE_LOGD("Node[%s] type[%s] do not need calc tiling sink running param.", node.GetNamePtr(), node.GetTypePtr());
    return SUCCESS;
  }
  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(node);
  FE_CHECK_NOTNULL(res_ptr_holder);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  if (SetTilingSinkCalcResources(node, op_exe_res_ctx) != SUCCESS) {
    FE_LOGE("Node[%s] type[%s] failed to set tiling sink calc resources", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  if (CalculateTilingSinkWorkspace(node, op_exe_res_ctx) != SUCCESS) {
    FE_LOGE("Node[%s] type[%s] failed to calc tiling sink workspace", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  proc_flag = true;
  return SUCCESS;
}

Status AICoreOpsKernelBuilder::CalcExtOpRunningParam(ge::Node &node) {
  bool proc_flag = false;
  FE_TIMECOST_START(CalcTilingSinkRunningParam);
  if (CalcTilingSinkRunningParam(CheckTilingSink(node), node, proc_flag) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CalcExtOpRunningParam] CalcTilingSinkRunningParam failed.");
    return FAILED;
  }
  if (proc_flag) {
    FE_TIMECOST_END(CalcTilingSinkRunningParam, "CalcTilingSinkRunningParam");
    return SUCCESS;
  }
  const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(ge::OppImplVersion::kOpp));
  if (space_registry != nullptr) {
    auto funcs_ptr = space_registry->GetOpImpl(node.GetType().c_str());
    if (funcs_ptr != nullptr && funcs_ptr->calc_op_param != nullptr) {
      FE_TIMECOST_START(CalcNewExtOpRunningParam);
      FE_LOGD("Node[%s] type[%s] find new calc param func.", node.GetNamePtr(), node.GetTypePtr());
      gert::ExeResGenerationCtxBuilder exe_ctx_builder;
      auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(node);
      FE_CHECK_NOTNULL(res_ptr_holder);
      auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
      if (funcs_ptr->calc_op_param(op_exe_res_ctx) != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[GenTask][CalcExtOpRunningParam] Op[%s][%s] failed to calc exe running param.",
                        node.GetNamePtr(), node.GetTypePtr());
        return FAILED;
      }
      FE_TIMECOST_END(CalcNewExtOpRunningParam, "CalcNewExtOpRunningParam");
      return SUCCESS;
    }
  }
  auto func = OpExtCalcParamRegistry::GetInstance().FindRegisterFunc(node.GetType());
  if (func == nullptr) {
    FE_LOGD("Node[%s] type[%s] do not register self op running param.", node.GetNamePtr(), node.GetTypePtr());
    return SUCCESS;
  }
  FE_LOGD("Node[%s] type[%s] calc self op running param.", node.GetNamePtr(), node.GetTypePtr());
  auto ret = func(node);
  if (ret != ge::SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CalcExtOpRunningParam] Op[%s][%s] failed to calc self op running param.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  return SUCCESS;
}

Status GenerateExtTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &task_defs) {
  bool reg_flag = false;
  FE_TIMECOST_START(GenerateOpExtTask);
  if (GenerateOpExtTask(node, CheckTilingSink(node), task_defs, reg_flag) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateOpExtTask] Op[%s][%s] failed to gen extra task.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  if (reg_flag) {
    FE_TIMECOST_END(GenerateOpExtTask, "GenerateOpExtTask.");
    return SUCCESS;
  }
  auto func = OpExtGenTaskRegistry::GetInstance().FindRegisterFunc(node.GetType());
  if (func == nullptr) {
    return SUCCESS;
  }
  FE_LOGD("Node[%s] type[%s] generate extra task.", node.GetNamePtr(), node.GetTypePtr());
  auto ret = func(node, context, task_defs);
  if (ret != ge::SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateExtTask] Op[%s][%s] failed to gen extra task.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  return SUCCESS;
}

Status AICoreOpsKernelBuilder::GenerateTask(const ge::Node &node, ge::RunContext &context,
                                            std::vector<domi::TaskDef> &tasks) {
  FE_LOGD("Begin to generate task for node[%s, %s].", node.GetName().c_str(), node.GetType().c_str());
  const ge::ComputeGraphPtr owner_graph = node.GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();

  // ffts task
  bool ffts_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, kTypeFFTSPlus, ffts_flag);
  if (ffts_flag || op_desc_ptr->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    FftsTaskBuilder ffts_task_builder;
    auto res = ffts_task_builder.GenerateFFTSPlusCtx(node, context, AI_CORE_NAME);
    return res;
  }

  // superkernel task
  if (ScopeAllocator::HasSkpScopeAttr(op_desc_ptr)) {
    return SuperkernelTaskBuilder::GenerateKernelTask(node, context, tasks);
  }

  // ascendc superkernel task
  if (op_desc_ptr->GetType() == kSuperKernelType) {
    FE_TIMECOST_START(SuperKernelTimeCost);
    auto res = SuperkernelTaskBuilder::GenerateSuperKernelTask(node, context, tasks);
    FE_TIMECOST_END(SuperKernelTimeCost, "SuperKernelTimeCost");
    return res;
  }

  // memset and normal task
  if (TaskBuilder::GenerateKernelTask(node, context, tasks) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateTask] Op[%s][%s] failed gen task.", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  auto res = GenerateExtTask(node, context, tasks);
  return res;
}
}  // namespace fe