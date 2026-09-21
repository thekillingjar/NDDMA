/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/ffts_task_builder.h"
#include "args_format_constructor.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "common/op_tensor_utils.h"
#include "graph/def_types.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
using RunContextPtr = std::shared_ptr<ge::RunContext>;
namespace {
const vector<std::string> kMixPrefixs = { "_mix_aic", "_mix_aiv" };
const std::string kArgsFormat = "MIX_L2_ARGS_FORMAT";
inline void GetCoreTypeByMode(const ge::OpDescPtr &op_desc, std::string &core_type, bool auto_mode, bool is_dynamic)
{
  if (auto_mode && !is_dynamic) {
    vector<string> thread_core_type;
    (void)ge::AttrUtils::GetListStr(op_desc, ATTR_NAME_THREAD_CUBE_VECTOR_CORE_TYPE, thread_core_type);
    core_type = thread_core_type.empty() ? core_type : thread_core_type[0];
  } else {
    (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  }
  return;
}
Status GetCtxTypeByCoreType(const ge::OpDescPtr &op_desc, std::string &core_type, ffts::TaskBuilderType &ctx_type,
                            bool auto_mode, bool is_dynamic)
{
  if (core_type.empty()) {
    return FAILED;
  }
  bool core_type_aic_aiv = core_type == kCoreTypeAIC || core_type == kCoreTypeAIV;
  bool core_type_mix_aic_aiv = core_type == kCoreTypeMixAIC || core_type == kCoreTypeMixAIV;
  if (core_type_aic_aiv) {
    ctx_type = ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV;
    if (auto_mode) {
      ctx_type = is_dynamic ? ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV_DYNAMIC :
                 ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV_AUTO;
    }
  } else if (core_type_mix_aic_aiv) {
    ctx_type = ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV;
    if (auto_mode) {
      ctx_type = is_dynamic ? ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_DYNAMIC :
                 ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_AUTO;
    }
    if (op_desc->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
      ctx_type = ffts::TaskBuilderType::EN_TASK_TYPE_MIX_L2_AIC_AIV;
    }
  } else {
    return FAILED;
  }
  return SUCCESS;
}
}

Status FftsTaskBuilder::GenerateFFTSPlusCtx(const ge::Node &node, const ge::RunContext &context,
                                            const std::string &engine_name) {
  FE_CHECK_NOTNULL(context.dataMemBase);
  FE_CHECK_NOTNULL(context.weightMemBase);
  context_.dataMemSize = context.dataMemSize;
  context_.dataMemBase = context.dataMemBase;
  context_.weightMemSize = context.weightMemSize;
  context_.weightMemBase = context.weightMemBase;
  context_.weightBufferHost = context.weightsBuffer;
  auto op_desc = node.GetOpDesc();
  FE_LOGD("Start to generate FFTSPlus Ctx, node %s, dataMemSize %lu, weightMemSize %lu.",
          node.GetName().c_str(), context.dataMemSize, context.weightMemSize);
  FftsPlusCtxDefPtr ctx = nullptr;
  FE_MAKE_SHARED(ctx = std::make_shared<domi::FftsPlusCtxDef>(), return FAILED);
  FE_CHECK_NOTNULL(ctx);
  if (engine_name == kDsaCoreEngineName) {
    RunContextPtr contextptr = nullptr;
    FE_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(context), return FAILED);
    FE_CHECK_NOTNULL(contextptr);
    (void)op_desc->SetExtAttr(kRuntimeContentx, contextptr);
    (void)op_desc->SetExtAttr(kAttrDsaCtxDef, ctx);
    return SUCCESS;
  }
  if (op_desc->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    bool no_ir_flag = op_desc->GetIrInputs().empty() && op_desc->GetIrOutputs().empty();
    FE_LOGD("Set op[%s, %s] with no IR flag: %d.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), no_ir_flag);
    if (!ge::AttrUtils::HasAttr(op_desc, kAttrNameIsFusionOp) && !no_ir_flag) {
      ArgsFormatConstructor args_construct(op_desc, true);
      if (args_construct.ConstructNodeArgsDesc() != SUCCESS) {
        FE_LOGE("[FFTSTaskBuilder] [GenMixL2CtxDef] Construct args desc failed.");
        return FAILED;
      }
      size_t addr_size = 0;
      if (args_construct.GetArgsSize(addr_size) != SUCCESS || addr_size == 0) {
        FE_LOGE("[FFTSTaskBuilder] [ConstructNodeArgsDesc] Get args size failed.");
        return FAILED;
      }
      addr_size /= sizeof(uintptr_t);
      (void)ge::AttrUtils::SetInt(op_desc, kAttrArgsSizeByFormat, addr_size);
      auto args_format_str = args_construct.GetArgsFormatString();
      FE_LOGD("The size of Node[%s][%s] args_format[%s] is %zu bytes.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
              args_format_str.c_str(), addr_size);
      (void)ge::AttrUtils::SetStr(op_desc, kArgsFormat, args_format_str);
    }
  }
  ffts::TaskBuilderType ctx_type;
  Status status = GenCtxParamAndCtxType(node, ctx_type);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateFFTSPlusCtx] Failed to run GenCtxParamAndCtxType on Node %s with type %s.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return status;
  }
  status = (this->*(gen_ctx_func_map_[ctx_type]))(op_desc, ctx);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateFFTSPlusCtx][Node %s, type %s, ctx type %d] Failed to run GenCtxDef.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str(), static_cast<int32_t>(ctx_type));
    return status;
  }

  (void)ge::AttrUtils::SetInt(op_desc, kAttrAICoreCtxType, static_cast<int64_t>(ctx_type));
  (void)op_desc->SetExtAttr(kAttrAICoreCtxDef, ctx);
  return SUCCESS;
}
Status FftsTaskBuilder::GenCtxParamAndCtxType(const ge::Node &node, ffts::TaskBuilderType &ctx_type) {
  auto op_desc = node.GetOpDesc();
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  bool auto_mode = slice_info_ptr != nullptr && slice_info_ptr->thread_mode ==
                                     static_cast<uint32_t>(ffts::ThreadMode::AUTO_THREAD);
  bool is_unknown = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_OWNER_GRAPH_IS_UNKNOWN, is_unknown);
  bool is_dynamic = is_unknown || fe::UnknownShapeUtils::IsUnknownShapeOp(*op_desc);
  FE_LOGD("Node[%s] is_dynamic: %u, auto_mode: %u, is_unknown: %u.",
          node.GetName().c_str(), is_dynamic, auto_mode, is_unknown);
  if (is_dynamic) {
    FE_LOGD("Node [%s] has an unknown shape, no need to get arguments.", node.GetName().c_str());
  } else if (auto_mode) {
    FftsTaskBuilderAdapterPtr ffts_task_builder_adapter_ptr = nullptr;
    FE_MAKE_SHARED(ffts_task_builder_adapter_ptr = std::make_shared<FftsTaskBuilderAdapter>(node, context_),
                   return FAILED);
    FE_CHECK_NOTNULL(ffts_task_builder_adapter_ptr);
    Status status = ffts_task_builder_adapter_ptr->Init();
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[FFTSPlusTaskBuidler][GenContextArgs][Node %s] Ffts plus Init ffts task builder adapter failed.",
                      node.GetOpDesc()->GetName().c_str());
      return status;
    }
    (void)ffts_task_builder_adapter_ptr->GetThreadParamOffset(auto_thread_param_offset_);
    SetAutoThreadIOAddrForDataCtx(op_desc);
  } else {
    TaskBuilderAdapterPtr task_builder_adapter_ptr = nullptr;
    FE_MAKE_SHARED(task_builder_adapter_ptr = std::make_shared<TbeTaskBuilderAdapter>(node, context_), return FAILED);
    FE_CHECK_NOTNULL(task_builder_adapter_ptr);
    Status status = task_builder_adapter_ptr->Init();
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[FFTSPlusTaskBuidler][GenContextArgs][Node %s] Ffts plus init tbe task builder adapter failed.",
                      node.GetOpDesc()->GetName().c_str());
      return status;
    }
    (void)task_builder_adapter_ptr->GetTaskArgs(manual_thread_param_);
    SetManualThreadIOAddrForDataCtx(op_desc);
  }
  std::string core_type;
  GetCoreTypeByMode(op_desc, core_type, auto_mode, is_dynamic);
  int32_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  uint32_t thread_dim = (slice_info_ptr != nullptr) ? slice_info_ptr->slice_instance_num : 1;
  FE_LOGD("Gen ctx type, Node[%s %s]'s core type: %s, block dim: %d, thread dim: %u.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), core_type.c_str(), block_dim, thread_dim);
  return GetCtxTypeByCoreType(op_desc, core_type, ctx_type, auto_mode, is_dynamic);
}

void FftsTaskBuilder::SetAutoThreadIOAddrForDataCtx(const ge::OpDescPtr &op_desc) {
  vector<int64_t> input_addrs;
  for (auto ele : auto_thread_param_offset_.first_thread_input_addrs) {
    input_addrs.emplace_back(static_cast<int64_t>(reinterpret_cast<intptr_t>(ele)));
  }
  vector<int64_t> output_addrs;
  for (auto ele : auto_thread_param_offset_.first_thread_output_addrs) {
    output_addrs.emplace_back(static_cast<int64_t>(reinterpret_cast<intptr_t>(ele)));
  }

  (void)ge::AttrUtils::SetListInt(op_desc, "input_addrs", input_addrs);
  (void)ge::AttrUtils::SetListInt(op_desc, "output_addrs", output_addrs);
}

void FftsTaskBuilder::SetManualThreadIOAddrForDataCtx(const ge::OpDescPtr &op_desc) {
  vector<int64_t> input_addrs;
  for (auto ele : manual_thread_param_.input_addrs) {
    input_addrs.emplace_back(static_cast<int64_t>(reinterpret_cast<intptr_t>(ele)));
  }
  vector<int64_t> output_addrs;
  for (auto ele : manual_thread_param_.output_addrs) {
    output_addrs.emplace_back(static_cast<int64_t>(reinterpret_cast<intptr_t>(ele)));
  }

  (void)ge::AttrUtils::SetListInt(op_desc, "input_addrs", input_addrs);
  (void)ge::AttrUtils::SetListInt(op_desc, "output_addrs", output_addrs);
}

Status FftsTaskBuilder::GenManualAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ctx->mutable_aic_aiv_ctx();
  FE_CHECK_NOTNULL(aic_aiv_ctx_def);

  // cache managemet will do at GenerateDataTaskDef()
  aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  aic_aiv_ctx_def->set_atm(kManualMode);
  aic_aiv_ctx_def->set_thread_dim(1);

  int32_t block_dim = 0;
  (void) ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  uint32_t schedule_mode = 0;
  (void) ge::AttrUtils::GetInt(op_desc, kAttrScheduleMode, schedule_mode);
  aic_aiv_ctx_def->set_schem(schedule_mode);
  FE_LOGD("Set schedule mode[%u] on task of op[%s, %s].", schedule_mode, op_desc->GetNamePtr(), op_desc->GetTypePtr());

  for (auto input_addr: manual_thread_param_.input_addrs) {
    uint64_t input_addr_tmp = ge::PtrToValue(input_addr);
    aic_aiv_ctx_def->add_task_addr(input_addr_tmp);
    FE_LOGD("input_addr: %lu", input_addr_tmp);
  }

  for (auto output_addr: manual_thread_param_.output_addrs) {
    uint64_t output_addr_tmp = ge::PtrToValue(output_addr);
    aic_aiv_ctx_def->add_task_addr(output_addr_tmp);
    FE_LOGD("output_addr, %lu", output_addr_tmp);
  }

  for (auto workspace_addr: manual_thread_param_.workspace_addrs) {
    uint64_t workspace_addr_tmp = ge::PtrToValue(workspace_addr);
    aic_aiv_ctx_def->add_task_addr(workspace_addr_tmp);
    FE_LOGD("workspace_addr: %lu", workspace_addr_tmp);
  }

  if (OpTensorUtils::IsStaticReuseBinaryOp(op_desc)) {
    uint64_t tiling_addr = 0;
    aic_aiv_ctx_def->add_task_addr(tiling_addr);
  }

  string attr_kernel_name;
  (void) ge::AttrUtils::GetStr(op_desc, kKernelName, attr_kernel_name);
  aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);

  GenMemSetCtxDef(op_desc);
  FE_LOGD("aic_aiv_ctx_def FillContextData SUCCESS. Op:%s, optype:%s, block_dim:%u, size:%d, attr_kernel_name:%s",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), static_cast<uint32_t>(block_dim),
          aic_aiv_ctx_def->task_addr_size(), attr_kernel_name.c_str());
  return SUCCESS;
}

void FftsTaskBuilder::GenMemSetCtxDef(const ge::OpDescPtr &op_desc) const {
  ge::NodePtr memset_node = nullptr;
  memset_node = op_desc->TryGetExtAttr(ATTR_NAME_MEMSET_NODE, memset_node);
  if (memset_node == nullptr) {
    return;
  }

  std::vector<int64_t> output_index;
  (void)ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);

  std::vector<int64_t> workspace_index;
  (void)ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_WORKSPACE_INDEX, workspace_index);

  size_t index = 0;
  std::vector<int64_t> work_spaces = memset_node->GetOpDesc()->GetWorkspace();
  auto output_addr_size = manual_thread_param_.output_addrs.size();
  auto workspace_addr_size = manual_thread_param_.workspace_addrs.size();
  work_spaces.resize(output_addr_size + workspace_addr_size);
  for (size_t idx = 0; idx < output_addr_size; ++idx) {
    if (idx < output_index.size() && output_index[idx] == 1) {
      work_spaces[index] = static_cast<int64_t>(reinterpret_cast<intptr_t>(manual_thread_param_.output_addrs[idx]));
    } else {
      work_spaces[index] = 0;
    }
    ++index;
  }

  for (size_t idx = 0; idx < workspace_addr_size; ++idx) {
    if (idx < workspace_index.size() && workspace_index[idx] == 1) {
      work_spaces[index] = static_cast<int64_t>(reinterpret_cast<intptr_t>(manual_thread_param_.workspace_addrs[idx]));
    } else {
      work_spaces[index] = 0;
    }
    ++index;
  }

  if (OpTensorUtils::IsStaticReuseBinaryOp(memset_node->GetOpDesc())) {
    // add tiling data args, GenerateAtomicCtx will filter 0, so add 1
    work_spaces.emplace_back(1);
  }
  memset_node->GetOpDesc()->SetWorkspace(work_spaces);
  memset_node->GetOpDesc()->SetId(op_desc->GetId());
}

Status FftsTaskBuilder::GenAutoAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ctx->mutable_aic_aiv_ctx();
  FE_CHECK_NOTNULL(aic_aiv_ctx_def);

  if (GenCommonAutoAICAIVCtxDef(op_desc, aic_aiv_ctx_def) != SUCCESS) {
    FE_LOGE("Failed to generate common CTC for auto AIC/AIV task.");
    return FAILED;
  }

  string attr_key_kernel_name = kThreadKernelName;
  vector<string> thread_kernel_name;
  (void)ge::AttrUtils::GetListStr(op_desc, attr_key_kernel_name, thread_kernel_name);
  for (const auto &kernel_name : thread_kernel_name) {
    aic_aiv_ctx_def->add_kernel_name(kernel_name);
    FE_LOGD("auto kernel_name: %s.", kernel_name.c_str());
  }
  size_t args_num = auto_thread_param_offset_.first_thread_input_addrs.size() +
                    auto_thread_param_offset_.first_thread_output_addrs.size() +
                    auto_thread_param_offset_.thread_workspace_addrs[0].size();
  aic_aiv_ctx_def->set_task_param_ptr_offset(args_num * sizeof(uint64_t));
  FE_LOGD("aic_aiv_ctx_def FillContextData SUCCESS. Op:%s, optype:%s, def:%s.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), aic_aiv_ctx_def->DebugString().c_str());
  if (GenAutoAtomicCtxDef(op_desc) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FftsTaskBuilder::GenAutoAtomicCtxDef(const ge::OpDescPtr &op_desc) {
  ge::NodePtr atomic_node = nullptr;
  atomic_node = op_desc->TryGetExtAttr(ATTR_NAME_MEMSET_NODE, atomic_node);
  if (atomic_node == nullptr) {
    return SUCCESS;
  }
  FE_LOGD("Node[%s] Atomic[%s] generated context definition.", op_desc->GetName().c_str(), atomic_node->GetName().c_str());
  std::vector<int64_t> output_index;
  (void)ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);
  std::vector<int64_t> workspace_index;
  (void)ge::AttrUtils::GetListInt(op_desc, TBE_OP_ATOMIC_WORKSPACE_INDEX, workspace_index);
  size_t index = 0;
  std::vector<int64_t> work_spaces = atomic_node->GetOpDesc()->GetWorkspace();
  auto output_addr_size = auto_thread_param_offset_.first_thread_output_addrs.size();
  auto workspace_addr_size = auto_thread_param_offset_.thread_workspace_addrs[0].size();
  work_spaces.resize(output_addr_size + workspace_addr_size);
  for (size_t idx = 0; idx < output_addr_size; ++idx) {
    if ((idx < output_index.size()) && (output_index[idx] == 1)) {
      work_spaces[index] = static_cast<int64_t>(reinterpret_cast<intptr_t>(
          auto_thread_param_offset_.first_thread_output_addrs[idx]));
    } else {
      work_spaces[index] = 0;
    }
    ++index;
  }
  for (size_t idx = 0; idx < workspace_addr_size; ++idx) {
    if ((idx < workspace_index.size()) && (workspace_index[idx] == 1)) {
      work_spaces[index] = static_cast<int64_t>(reinterpret_cast<intptr_t>(
          auto_thread_param_offset_.thread_workspace_addrs[0][idx]));
    } else {
      work_spaces[index] = 0;
    }
    ++index;
  }
  atomic_node->GetOpDesc()->SetWorkspace(work_spaces);
  return SUCCESS;
}

Status FftsTaskBuilder::GenManualMixAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ctx->mutable_mix_aic_aiv_ctx();
  FE_CHECK_NOTNULL(mix_aic_aiv_ctx_def);
  uint32_t task_ratio;
  (void)ge::AttrUtils::GetInt(op_desc, kTaskRadio, task_ratio);

  mix_aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  mix_aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  mix_aic_aiv_ctx_def->set_ns(1);
  mix_aic_aiv_ctx_def->set_atm(kManualMode);
  mix_aic_aiv_ctx_def->set_thread_dim(1);

  mix_aic_aiv_ctx_def->set_tail_block_ratio_n(task_ratio);
  mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(task_ratio);

  int32_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  mix_aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  mix_aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  uint32_t schedule_mode = 0;
  (void) ge::AttrUtils::GetInt(op_desc, kAttrScheduleMode, schedule_mode);
  mix_aic_aiv_ctx_def->set_schem(schedule_mode);
  FE_LOGD("Set schedule mode[%u] on task of op[%s, %s].", schedule_mode, op_desc->GetNamePtr(), op_desc->GetTypePtr());

  // modeInArgsFirstField
  uint32_t mode = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kModeInArgsFirstField, mode);
  bool inter_core_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrIntercoreSync, inter_core_sync);
  // mode == 1 indicates we need reserve 8 Bytes for the args beginning
  if (mode == 1 || inter_core_sync) {
    uint64_t modeArgs = 0;
    mix_aic_aiv_ctx_def->add_task_addr(modeArgs);
  }
  for (auto input_addr : manual_thread_param_.input_addrs) {
    mix_aic_aiv_ctx_def->add_task_addr(ge::PtrToValue(input_addr));
  }

  for (auto output_addr : manual_thread_param_.output_addrs) {
    mix_aic_aiv_ctx_def->add_task_addr(ge::PtrToValue(output_addr));
  }

  for (auto workspace_addr : manual_thread_param_.workspace_addrs) {
    mix_aic_aiv_ctx_def->add_task_addr(ge::PtrToValue(workspace_addr));
  }

  if (OpTensorUtils::IsStaticReuseBinaryOp(op_desc)) {
    uint64_t tiling_addr = 0;
    mix_aic_aiv_ctx_def->add_task_addr(tiling_addr);
  }
  FillMixAICAIVKernelName(op_desc, mix_aic_aiv_ctx_def);
  GenMemSetCtxDef(op_desc);
  return SUCCESS;
}

void FftsTaskBuilder::FillMixAICAIVKernelName(const ge::OpDescPtr &op_desc,
                                              domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def) const {
  vector<std::string> mix_l2_prefix;
  if (!ge::AttrUtils::GetListStr(op_desc, kKernelNamesPrefix, mix_l2_prefix)) {
    mix_l2_prefix = kMixPrefixs;
  }
  for (auto &prefix : mix_l2_prefix) {
    string attr_key_kernel_name = prefix + kKernelName;
    string attr_kernel_name;
    (void)ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
    mix_aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
  }
}

Status FftsTaskBuilder::GenDynamicAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ctx->mutable_aic_aiv_ctx();
  FE_CHECK_NOTNULL(aic_aiv_ctx_def);
  aic_aiv_ctx_def->set_ns(1);
  aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  aic_aiv_ctx_def->set_aten(kAutoMode);
  aic_aiv_ctx_def->set_atm(kAutoMode);

  // generate _register_stub_func
  vector<string> unique_ids;
  string session_graph_id;
  if (ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
    unique_ids.push_back(session_graph_id + "_" + op_desc->GetName() + "_0");
  } else {
    unique_ids.push_back(op_desc->GetName() + "_0");
  }
  (void)ge::AttrUtils::SetListStr(op_desc, "_register_stub_func", unique_ids);
  FE_LOGD("FillContextData SUCCESS. Op:%s, type:%s.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return SUCCESS;
}

Status FftsTaskBuilder::GenAutoMixAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ctx->mutable_mix_aic_aiv_ctx();
  FE_CHECK_NOTNULL(mix_aic_aiv_ctx_def);

  if (GenCommonAutoAICAIVCtxDef(op_desc, mix_aic_aiv_ctx_def) != SUCCESS) {
    FE_LOGE("Failed to generate common CTC for auto AIC/AIV task.");
    return FAILED;
  }

  mix_aic_aiv_ctx_def->set_ns(1);

  vector<uint32_t> task_ratio_list;
  (void)ge::AttrUtils::GetListInt(op_desc, kThreadTaskRadio, task_ratio_list);
  if (task_ratio_list.size() > 1) {
    mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(task_ratio_list[0]);
    mix_aic_aiv_ctx_def->set_tail_block_ratio_n(task_ratio_list[1]);
  }

  // modeInArgsFirstField
  uint32_t mode = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kModeInArgsFirstField, mode);
  bool inter_core_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrIntercoreSync, inter_core_sync);
  // mode == 1 indicates we need reserve 8 Bytes for the args beginning
  if (mode == 1 || inter_core_sync) {
    uint64_t modeArgs = 0;
    mix_aic_aiv_ctx_def->add_task_addr(modeArgs);
  }
  vector<std::string> mix_l2_prefix;
  if (!ge::AttrUtils::GetListStr(op_desc, kKernelNamesPrefix, mix_l2_prefix)) {
    mix_l2_prefix = kMixPrefixs;
  }
  for (auto &prefix : mix_l2_prefix) {
    string attr_key_kernel_name = prefix + kThreadKernelName;
    string attr_kernel_name;
    (void)ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
    mix_aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
  }
  return SUCCESS;
}

Status FftsTaskBuilder::GenDynamicMixAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  FE_LOGD("Generated dynamic mix node [%s] context.", op_desc->GetName().c_str());
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ctx->mutable_mix_aic_aiv_ctx();
  FE_CHECK_NOTNULL(mix_aic_aiv_ctx_def);
  mix_aic_aiv_ctx_def->set_ns(1);
  mix_aic_aiv_ctx_def->set_aten(1);
  mix_aic_aiv_ctx_def->set_atm(1);
  uint32_t task_ratio = 0U;
  (void)ge::AttrUtils::GetInt(op_desc, kTaskRadio, task_ratio);
  mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(task_ratio);
  mix_aic_aiv_ctx_def->set_tail_block_ratio_n(task_ratio);
  return SUCCESS;
}

Status FftsTaskBuilder::GenMixL2CtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx) {
  FE_LOGD("Node [%s] fills MixL2 context data.", op_desc->GetName().c_str());
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ctx->mutable_mix_aic_aiv_ctx();
  FE_CHECK_NOTNULL(mix_aic_aiv_ctx_def);

  uint32_t task_ratio;
  (void)ge::AttrUtils::GetInt(op_desc, kTaskRadio, task_ratio);
  mix_aic_aiv_ctx_def->set_tail_block_ratio_n(task_ratio);
  mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(task_ratio);
  mix_aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  mix_aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  mix_aic_aiv_ctx_def->set_ns(1);
  mix_aic_aiv_ctx_def->set_atm(kManualMode);
  mix_aic_aiv_ctx_def->set_thread_dim(1);
  FillMixAICAIVKernelName(op_desc, mix_aic_aiv_ctx_def);
  int32_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  mix_aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  mix_aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  uint32_t schedule_mode = 0;
  (void) ge::AttrUtils::GetInt(op_desc, kAttrScheduleMode, schedule_mode);
  mix_aic_aiv_ctx_def->set_schem(schedule_mode);
  std::string args_format_str;
  (void)ge::AttrUtils::GetStr(op_desc, kArgsFormat, args_format_str);
  mix_aic_aiv_ctx_def->set_args_format(args_format_str);
  if (fe::UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    FE_LOGD("Node [%s] has an unknown shape", op_desc->GetName().c_str());
    return SUCCESS;
  }
  for (auto input_addr : manual_thread_param_.input_addrs) {
    mix_aic_aiv_ctx_def->add_task_addr(reinterpret_cast<uintptr_t>(input_addr));
  }
  for (auto output_addr : manual_thread_param_.output_addrs) {
    mix_aic_aiv_ctx_def->add_task_addr(reinterpret_cast<uintptr_t>(output_addr));
  }
  for (auto workspace_addr : manual_thread_param_.workspace_addrs) {
    mix_aic_aiv_ctx_def->add_task_addr(reinterpret_cast<uintptr_t>(workspace_addr));
  }
  if (OpTensorUtils::IsStaticReuseBinaryOp(op_desc)) {
    uint64_t tiling_addr = 0;
    mix_aic_aiv_ctx_def->add_task_addr(tiling_addr);
  }
  GenMemSetCtxDef(op_desc);
  return SUCCESS;
}

template <typename T>
Status FftsTaskBuilder::GenCommonAutoAICAIVCtxDef(const ge::OpDescPtr &op_desc, T *ctx) {
  // cache managemet will do at GenerateDataTaskDef()
  ctx->set_prefetch_once_bitmap(0);
  ctx->set_prefetch_enable_bitmap(0);

  // set ffts+ mode to auto, which value is 1;
  ctx->set_aten(1);
  ctx->set_atm(1);

  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  FE_CHECK_NOTNULL(slice_info_ptr);
  ctx->set_thread_dim(slice_info_ptr->slice_instance_num);

  vector<int32_t> block_dims;
  (void)ge::AttrUtils::GetListInt(op_desc, ge::TVM_ATTR_NAME_THREAD_BLOCKDIM, block_dims);
  if (block_dims.size() > 1) {
    ctx->set_non_tail_block_dim(static_cast<uint32_t>(block_dims[0]));
    ctx->set_tail_block_dim(static_cast<uint32_t>(block_dims[1]));
    FE_LOGD("block_dims[0]: %u, block_dims[1]: %u.", static_cast<uint32_t>(block_dims[0]),
            static_cast<uint32_t>(block_dims[1]));
  }

  uint32_t schedule_mode = 0;
  (void) ge::AttrUtils::GetInt(op_desc, kAttrScheduleMode, schedule_mode);
  ctx->set_schem(schedule_mode);
  FE_LOGD("Set schedule mode[%u] on task of op[%s, %s].", schedule_mode, op_desc->GetNamePtr(), op_desc->GetTypePtr());

  // generate _register_stub_func
  vector<string> unique_ids;
  string session_graph_id = "";
  if (ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
    unique_ids.push_back(session_graph_id + "_" + op_desc->GetName() + "_0");
    unique_ids.push_back(session_graph_id + "_" + op_desc->GetName() + "_1");
  } else {
    unique_ids.push_back(op_desc->GetName() + "_0");
    unique_ids.push_back(op_desc->GetName() + "_1");
  }
  (void)ge::AttrUtils::SetListStr(op_desc, "_register_stub_func", unique_ids);

  uint32_t input_output_num = auto_thread_param_offset_.first_thread_input_addrs.size() +
                              auto_thread_param_offset_.first_thread_output_addrs.size();
  ctx->set_input_output_count(input_output_num);
  for (auto input_addr : auto_thread_param_offset_.first_thread_input_addrs) {
    ctx->add_task_addr(ge::PtrToValue(input_addr));
  }
  for (auto output_addr : auto_thread_param_offset_.first_thread_output_addrs) {
    ctx->add_task_addr(ge::PtrToValue(output_addr));
  }
  if (auto_thread_param_offset_.thread_workspace_addrs.size() > 1) {
    for (auto workspace_addr : auto_thread_param_offset_.thread_workspace_addrs[0]) {
      ctx->add_task_addr(ge::PtrToValue(workspace_addr));
    }
  }
  if (OpTensorUtils::IsStaticReuseBinaryOp(op_desc)) {
    uint64_t tiling_addr = 0;
    ctx->add_task_addr(tiling_addr);
    ctx->add_task_addr(tiling_addr);
  }
  for (auto addr_offset : auto_thread_param_offset_.thread_addr_offset) {
    ctx->add_task_addr_offset(static_cast<uint64_t>(addr_offset));
  }
  return SUCCESS;
}
}
