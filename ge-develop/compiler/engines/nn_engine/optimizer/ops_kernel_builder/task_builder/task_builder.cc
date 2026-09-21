/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/task_builder.h"

#include <securec.h>
#include <string>
#include "args_format_constructor.h"
#include "common/fe_utils.h"
#include "common/fe_error_code.h"
#include "common/op_tensor_utils.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_constants.h"
#include "common/fe_inner_attr_define.h"
#include "common/platform_utils.h"
#include "framework/common/taskdown_common.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "ops_kernel_builder/task_builder/cmo_task_builder.h"
#include "adapter/common/task_builder_adapter_factory.h"
#include "graph/utils/node_utils.h"
#include "rt_error_codes.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "runtime/stream.h"

namespace fe {
namespace {
const uint32_t kKernelTypeTe = 2;
const std::string kAttrSlowContextSwitch = "_slow_context_switch";
const std::string kAttrSwitchBufferType = "_switch_buffer_type";
const std::string kSwitchBufferTypeUB = "UF";
struct MixTaskPara {
  int64_t main_blk_dim;
  int64_t main_offset;
  int64_t sub_blk_dim;
  int64_t sub_offset;
  uint32_t kernel_type;
};

enum class CoreNumType {
  ALL_CORE,
  AI_CORE,
  VECTOR_CORE,
  TYPE_NUM
};

MixTaskPara CalcMixTaskParaByType(string &core_type, int64_t block_dim, int64_t ai_core_num,
                                  int64_t vec_core_num, int64_t all_core_num, int64_t ratio) {
  MixTaskPara para;
  FE_LOGD("Before calculating core type[%s] blk[%ld] ai_core[%ld] vec_core[%ld] all_core[%ld].", core_type.c_str(), block_dim,
          ai_core_num, vec_core_num, all_core_num);
  if (core_type == kCoreTypeMixVectorCore || ratio == 1) {
    para.main_offset = 0;
    // need round up
    para.main_blk_dim = (block_dim * ai_core_num + all_core_num - 1) / all_core_num;
    para.sub_offset = para.main_blk_dim;
    para.sub_blk_dim = block_dim - para.main_blk_dim;
    para.kernel_type = static_cast<uint32_t>(ge::ccKernelType::MIX_VECTOR_CORE);
  } else {
    para.main_offset = 0;
    para.sub_offset = 0;
    para.main_blk_dim = block_dim;
    para.sub_blk_dim = vec_core_num;
    para.kernel_type = static_cast<uint32_t>(ge::ccKernelType::MIX_AICORE);
  }
  FE_LOGD("After calculating sub_offset[%ld], main_blk_dim[%ld], sub_blk_dim[%ld], kernel_type[%u].", para.sub_offset,
          para.main_blk_dim, para.sub_blk_dim, para.kernel_type);
  return para;
}

Status UpdateMixAiCoreTask(MixTaskPara &para, domi::TaskDef& main_task, domi::TaskDef& sub_task) {
  domi::KernelContext *main_kernel_context = nullptr;
  domi::KernelContext *sub_kernel_context = nullptr;
  if (main_task.type() == RT_MODEL_TASK_KERNEL) {
    sub_task.set_type(RT_MODEL_TASK_VECTOR_KERNEL);
    domi::KernelDef *main_kernel_def = main_task.mutable_kernel();
    FE_CHECK_NOTNULL(main_kernel_def);
    main_kernel_context = main_kernel_def->mutable_context();
    domi::KernelDef *sub_kernel_def = sub_task.mutable_kernel();
    FE_CHECK_NOTNULL(sub_kernel_def);
    sub_kernel_context = sub_kernel_def->mutable_context();
    main_kernel_def->set_block_dim_offset(para.main_offset);
    sub_kernel_def->set_block_dim_offset(para.sub_offset);
    main_kernel_def->set_block_dim(para.main_blk_dim);
    sub_kernel_def->set_block_dim(para.sub_blk_dim);
  } else {
    domi::KernelDefWithHandle *main_kernel_def = main_task.mutable_kernel_with_handle();
    FE_CHECK_NOTNULL(main_kernel_def);
    main_kernel_context = main_kernel_def->mutable_context();
    sub_task.set_type(RT_MODEL_TASK_VECTOR_ALL_KERNEL);
    domi::KernelDefWithHandle *sub_kernel_def = sub_task.mutable_kernel_with_handle();
    FE_CHECK_NOTNULL(sub_kernel_def);
    sub_kernel_context = sub_kernel_def->mutable_context();
    main_kernel_def->set_block_dim_offset(para.main_offset);
    sub_kernel_def->set_block_dim_offset(para.sub_offset);
    main_kernel_def->set_block_dim(para.main_blk_dim);
    sub_kernel_def->set_block_dim(para.sub_blk_dim);
  }
  FE_CHECK_NOTNULL(main_kernel_context);
  FE_CHECK_NOTNULL(sub_kernel_context);
  main_kernel_context->set_kernel_type(para.kernel_type);
  sub_kernel_context->set_kernel_type(para.kernel_type);
  FE_LOGD("Main kernel type is %u.", para.kernel_type);
  return SUCCESS;
}

Status FillMixExtraTask(const ge::Node &node, string &core_type, MixTaskPara &para, domi::TaskDef& main_task,
                        std::vector<domi::TaskDef> &task_defs) {
  (void) core_type;
  int64_t main_stream_id = node.GetOpDesc()->GetStreamId();
  int64_t sub_stream_id = node.GetOpDesc()->GetAttachedStreamId();
  FE_LOGD("Mix node[%s] stream id:%ld sub stream id:%ld.", node.GetNamePtr(), main_stream_id, sub_stream_id);
  std::vector<int32_t> notify_id_v;
  (void)ge::AttrUtils::GetListInt(node.GetOpDesc(), ge::RECV_ATTR_NOTIFY_ID, notify_id_v);
  if (notify_id_v.size() != kMixNotifyIdNum) {
    FE_LOGW("Op[%s][%s] Get notify id size: %zu is not 2, switching to normal mode.",
            node.GetNamePtr(), node.GetTypePtr(), notify_id_v.size());
    return SUCCESS;
  }
  domi::TaskDef sub_task = main_task;
  sub_task.set_stream_id(sub_stream_id);
  if (UpdateMixAiCoreTask(para, main_task, sub_task) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][FillMixExtraTask] Op[%s][%s] Update task definition failed.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  domi::TaskDef sub_wait;
  sub_wait.set_notify_id(notify_id_v[0]);
  sub_wait.set_stream_id(sub_stream_id);
  sub_wait.set_type(RT_MODEL_TASK_NOTIFY_WAIT);

  domi::TaskDef main_record;
  main_record.set_notify_id(notify_id_v[0]);
  main_record.set_stream_id(main_stream_id);
  main_record.set_type(RT_MODEL_TASK_NOTIFY_RECORD);

  domi::TaskDef sub_record;
  sub_record.set_notify_id(notify_id_v[1]);
  sub_record.set_stream_id(sub_stream_id);
  sub_record.set_type(RT_MODEL_TASK_NOTIFY_RECORD);

  domi::TaskDef main_wait;
  main_wait.set_notify_id(notify_id_v[1]);
  main_wait.set_stream_id(main_stream_id);
  main_wait.set_type(RT_MODEL_TASK_NOTIFY_WAIT);

  task_defs.insert(task_defs.begin() + task_defs.size() - 1, main_record);
  task_defs.emplace_back(sub_wait);
  task_defs.emplace_back(sub_task);
  task_defs.emplace_back(sub_record);
  task_defs.emplace_back(main_wait);
  return SUCCESS;
}

Status GenerateStaMixTask(const ge::Node &node, string &core_type, domi::TaskDef& main_task,
                          std::vector<domi::TaskDef> &task_defs, std::vector<uint32_t> &core_num_v) {
  int64_t block_dim = 1;
  int64_t ratio = 0;
  // This include static reuse binary sense
  (void)ge::AttrUtils::GetInt(node.GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  (void)ge::AttrUtils::GetInt(node.GetOpDesc(), kTaskRadio, ratio);
  PlatFormInfos platform_infos;
  OptionalInfos opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos,
                                                                       opti_compilation_info) != SUCCESS) {
    FE_LOGE("Failed to get platform.");
    return FAILED;
  }
  uint32_t all_core_num = core_num_v[static_cast<size_t>(CoreNumType::ALL_CORE)];
  uint32_t ai_core_num = core_num_v[static_cast<size_t>(CoreNumType::AI_CORE)];
  uint32_t vec_core_num = core_num_v[static_cast<size_t>(CoreNumType::VECTOR_CORE)];
  if (block_dim <= ai_core_num && ratio == 0) {
    FE_LOGD("Mix block dim[%ld] less than ai core num[%u], go normal mode.", block_dim, ai_core_num);
    return SUCCESS;
  }
  auto mix_para = CalcMixTaskParaByType(core_type, block_dim, ai_core_num, vec_core_num, all_core_num, ratio);
  return FillMixExtraTask(node, core_type, mix_para, main_task, task_defs);
}

Status DynMixSetCoreNum(const ge::Node &node, std::vector<uint32_t> &core_num_v) {
  (void)ge::AttrUtils::SetListInt(node.GetOpDesc(), ATTR_NAME_MIX_CORE_NUM_VEC, core_num_v);
  return SUCCESS;
}

std::vector<uint32_t> GetPlatformCoreNum(string &mix_core_type) {
  std::vector<uint32_t> core_num_v;
  PlatFormInfos platform_infos;
  OptionalInfos opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos,
                                                                       opti_compilation_info) != SUCCESS) {
    FE_LOGE("Failed to get platform.");
    return core_num_v;
  }
  platform_infos.SetCoreNumByCoreType(mix_core_type);
  uint32_t all_core_num = platform_infos.GetCoreNum();
  platform_infos.SetCoreNumByCoreType(kCoreTypeAIC);
  uint32_t ai_core_num = platform_infos.GetCoreNum();
  platform_infos.SetCoreNumByCoreType(kCoreTypeAIV);
  uint32_t vec_core_num = platform_infos.GetCoreNum();
  if (ai_core_num == 0 || all_core_num == 0 || vec_core_num == 0) {
    FE_LOGE("AICore/ALL/Vector num: %u/%u/%u is invalid.", ai_core_num, all_core_num, vec_core_num);
    return core_num_v;
  }
  core_num_v.emplace_back(all_core_num);
  core_num_v.emplace_back(ai_core_num);
  core_num_v.emplace_back(vec_core_num);
  return core_num_v;
}

/*
 *  for 51 enable vector core
 *  (51 do not have cmo task)
 * */
Status GenerateMixTask(const ge::Node &node, std::vector<domi::TaskDef> &task_defs) {
  string core_type;
  (void)ge::AttrUtils::GetStr(node.GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  if (core_type != kCoreTypeMixVectorCore && core_type != kCoreTypeMixAICore) {
    return SUCCESS;
  }
  if (ge::AttrUtils::HasAttr(node.GetOpDesc(), ATTR_NAME_DISABLE_MIX_VECTOR_CORE)) {
    return SUCCESS;
  }
  auto &ai_task = task_defs[task_defs.size() - 1];
  if (ai_task.type() != RT_MODEL_TASK_KERNEL && ai_task.type() != RT_MODEL_TASK_ALL_KERNEL) {
    REPORT_FE_ERROR("[GenTask][GenerateMixTask] Op[%s][%s] did not find ai core task.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  static std::vector<uint32_t> core_num_v = GetPlatformCoreNum(core_type);
  if (core_num_v.empty()) {
    REPORT_FE_ERROR("[GenTask][GenerateMixTask] Failed to get core number.");
    return FAILED;
  }
  auto own_graph = node.GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(own_graph);
  bool is_in_dyn_graph = (own_graph->GetParentNode() != nullptr) && own_graph->GetGraphUnknownFlag();
  FE_LOGD("Node[%s] is in %s graph flag.", node.GetNamePtr(), is_in_dyn_graph ? "dynamic" : "static");
  if (is_in_dyn_graph) {
    return DynMixSetCoreNum(node, core_num_v);
  } else {
    return GenerateStaMixTask(node, core_type, ai_task, task_defs, core_num_v);
  }
}
} // namespace

Status TaskBuilder::GenerateKernelTask(const ge::Node &node, const ge::RunContext &context,
                                       std::vector<domi::TaskDef> &task_defs) {
  FE_LOGD("TaskBuilder::GenerateKernelTask begin, node name:%s, type:%s.", node.GetName().c_str(),
          node.GetType().c_str());
  FE_TIMECOST_START(GenerateTask);
  TaskBuilderContext task_context;
  if (InitTaskContext(context, task_context) != SUCCESS) {
    FE_LOGE("Failed to init task context.");
    return FAILED;
  }

  Status status = GenerateKernelTask(node, task_context, task_defs);
  FE_TIMECOST_END_LOGI(GenerateTask, "TaskBuilder::GenerateTask");
  return status;
}

Status TaskBuilder::GenerateKernelTask(const ge::Node &node, TaskBuilderContext &task_context,
                                       std::vector<domi::TaskDef> &task_defs) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  int64_t stream_id = op_desc_ptr->GetStreamId();
  StartKernelFusion(op_desc_ptr, stream_id, task_defs);

  GenerateCMOTask(node, true, task_context, task_defs);
  if (GenTaskForOneNode(node, task_context, task_defs, stream_id) != SUCCESS) {
    return FAILED;
  }
  if (GenerateMixTask(node, task_defs) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateMixTask] Op[%s][%s] failed to generate mix task.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  GenerateCMOTask(node, false, task_context, task_defs);
  EndKernelFusion(op_desc_ptr, stream_id, task_defs);
  FE_LOGD("TaskBuilder::GenerateTask end, node name:%s, type:%s, stream_id:%ld.", node.GetNamePtr(), node.GetTypePtr(),
          stream_id);
  return SUCCESS;
}

Status TaskBuilder::InitTaskContext(const ge::RunContext &context, TaskBuilderContext &task_context) {
  FE_CHECK_NOTNULL(context.dataMemBase);
  FE_CHECK_NOTNULL(context.weightMemBase);
  task_context.dataMemSize = context.dataMemSize;
  task_context.dataMemBase = context.dataMemBase;
  task_context.weightMemSize = context.weightMemSize;
  task_context.weightMemBase = context.weightMemBase;
  task_context.weightBufferHost = context.weightsBuffer;
  task_context.mem_type_to_data_mem_base = context.mem_type_data_mem_base;
  task_context.mem_type_to_data_mem_size = context.mem_type_data_mem_size;
  if (PlatformUtils::Instance().IsSpecifiedMemBase()) {
    task_context.dataMemBase = nullptr;
    task_context.weightMemBase = nullptr;
    FE_LOGD("Data mem base and weight mem base have been set to null for current soc is nano.");
  }
  return SUCCESS;
}

void TaskBuilder::StartKernelFusion(const ge::OpDescPtr &op_desc_ptr, const int32_t &stream_id,
                                    std::vector<domi::TaskDef> &task_defs) {
  bool is_first_node = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_FIRST_NODE, is_first_node);
  if (!is_first_node) {
    return;
  }

  FE_LOGD("Start kernel fusion from node %s, type %s.", op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  domi::TaskDef task_def = {};
  task_def.set_type(RT_MODEL_TASK_FUSION_START);
  task_def.set_stream_id(stream_id);
  task_defs.push_back(task_def);
}

void TaskBuilder::EndKernelFusion(const ge::OpDescPtr &op_desc_ptr, const int32_t &stream_id,
                                  std::vector<domi::TaskDef> &task_defs) {
  bool is_last_node = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_LAST_NODE, is_last_node);
  if (!is_last_node) {
    return;
  }

  FE_LOGD("Finish kernel fusion of node %s, type %s.", op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  domi::TaskDef task_def = {};
  task_def.set_type(RT_MODEL_TASK_FUSION_END);
  task_def.set_stream_id(stream_id);
  task_defs.push_back(task_def);
}

Status TaskBuilder::GenTaskForOneNode(const ge::Node &node, TaskBuilderContext &task_context,
                                      std::vector<domi::TaskDef> &task_defs, int64_t stream_id) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  // generate task for memset node, cmo and memset will not appear at same time
  ge::NodePtr memset_node = op_desc_ptr->TryGetExtAttr<ge::NodePtr>(ATTR_NAME_MEMSET_NODE, nullptr);
  if (memset_node != nullptr) {
    int64_t op_desc_id = op_desc_ptr->GetId();
    FE_LOGD("Op:%s op id is %ld", op_desc_ptr->GetName().c_str(), op_desc_id);
    memset_node->GetOpDesc()->SetId(op_desc_id);
    domi::TaskDef mem_set_task_def = {};
    mem_set_task_def.set_stream_id(stream_id);
    if (DoGenerateTask(*memset_node, task_context, mem_set_task_def) != SUCCESS) {
      REPORT_FE_ERROR("[GenTask][MemsetGen][Node %s, type %s] generate task failed!",
                      memset_node->GetName().c_str(), memset_node->GetType().c_str());
      FE_LOGE("op:%s is memset op, generate task failed.", memset_node->GetName().c_str());
      return FAILED;
    }
    task_defs.push_back(mem_set_task_def);
    FE_LOGD("Generate task for memset node[%s, %s] of op[%s, %s].",
            memset_node->GetName().c_str(), memset_node->GetType().c_str(),
            node.GetName().c_str(), node.GetType().c_str());
  }

  // generate task for current node
  domi::TaskDef task_def = {};
  task_def.set_stream_id(stream_id);
  if (DoGenerateTask(node, task_context, task_def) != SUCCESS) {
    return FAILED;
  }
  task_defs.push_back(task_def);
  SetSlowContextSwitchAttr(op_desc_ptr);
  FE_LOGD("Generate task for node [%s, %s].", node.GetName().c_str(), node.GetType().c_str());
  return SUCCESS;
}

Status TaskBuilder::DoGenerateTask(const ge::Node &node, TaskBuilderContext &task_context, domi::TaskDef &task_def) {
  std::string args_format_str;
  if (GenerateOpArgsFormat(node.GetOpDesc(), args_format_str) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][InitKernelTask] Node [%s][%s]: Generating args format failed.",
                    node.GetNamePtr(), node.GetTypePtr());
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  // Create TaskBuilderAdapter
  TaskBuilderAdapterPtr task_adapter = CreateTaskAdapter(node, task_context);
  if (task_adapter == nullptr) {
    REPORT_FE_ERROR("[GenTask][DoGenerateTask][Node %s type %s] Failed to create adapter.",
                    node.GetName().c_str(), node.GetType().c_str());
    return TASK_BUILDER_CREATE_ADAPTER_FAILED;
  }

  Status status = task_adapter->Run(task_def);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][DoGenerateTask][Node %s type %s] Failed to run task adapter.",
                    node.GetName().c_str(), node.GetType().c_str());
    return status;
  }

  status = FillTaskDefAfterGenTask(node.GetOpDesc(), task_def, args_format_str);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][DoGenerateTask][Node %s type %s] Failed to fill up task def.",
                    node.GetName().c_str(), node.GetType().c_str());
    return status;
  }
  return SUCCESS;
}

Status TaskBuilder::GenerateOpArgsFormat(const ge::OpDescPtr &op_desc, std::string &args_format_str) {
  if (op_desc->GetType() == "MemSet") {
    FE_LOGD("Memset[%s] can not construct args format.", op_desc->GetNamePtr());
    return SUCCESS;
  }
  ArgsFormatConstructor args_construct(op_desc, false);
  if (args_construct.ConstructNodeArgsDesc() != SUCCESS) {
    FE_LOGE("[GenTask][DoGenerateTask][Node %s type %s] Construct args desc failed.",
            op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return FAILED;
  }
  args_format_str = args_construct.GetArgsFormatString();
  FE_LOGD("Node[%s][%s] generate args format %s.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), args_format_str.c_str());
  return SUCCESS;
}

Status TaskBuilder::FillTaskDefAfterGenTask(const ge::OpDescPtr &op_desc, domi::TaskDef &task_def,
                                            std::string &args_format) {
  std::string attr_val_kernel_name;
  (void)ge::AttrUtils::GetStr(op_desc, kKernelName, attr_val_kernel_name);
  uint32_t schedule_mode = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kAttrScheduleMode, schedule_mode);
  FE_LOGD("Set schedule mode[%u] on task of op[%s, %s].", schedule_mode, op_desc->GetNamePtr(), op_desc->GetTypePtr());
  domi::KernelContext *kernel_context = nullptr;
  if (task_def.type() == RT_MODEL_TASK_KERNEL) {
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    FE_CHECK_NOTNULL(kernel_def);
    kernel_def->set_kernel_name(attr_val_kernel_name);
    FE_LOGD("Set kernel_name[%s] for the kernel_def of node[%s, %s].",
            attr_val_kernel_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());

    kernel_def->set_schedule_mode(schedule_mode);
    kernel_context = kernel_def->mutable_context();
  }
  if (task_def.type() == RT_MODEL_TASK_ALL_KERNEL) {
    domi::KernelDefWithHandle *kernel_def_with_handle = task_def.mutable_kernel_with_handle();
    FE_CHECK_NOTNULL(kernel_def_with_handle);
    std::string first_kernel_name;
    if (ge::AttrUtils::GetStr(op_desc, ATTR_NAME_KERNEL_LIST_FIRST_NAME, first_kernel_name))  {
      kernel_def_with_handle->set_node_info(first_kernel_name);
      kernel_def_with_handle->set_original_kernel_key(attr_val_kernel_name);
      FE_LOGD("Setting node_info[%s] and original_kernel_key[%s] for the kernel_def_with_handle of node[%s, %s].",
              first_kernel_name.c_str(), attr_val_kernel_name.c_str(),
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    } else {
      REPORT_FE_ERROR("[GenTask][GenAllKernelTask] No kernel list name in op desc.");
      return ACL_ERROR_RT_PARAM_INVALID;
    }

    kernel_def_with_handle->set_schedule_mode(schedule_mode);
    kernel_context = kernel_def_with_handle->mutable_context();
  }
  FE_CHECK_NOTNULL(kernel_context);
  uint32_t op_index = static_cast<uint32_t>(op_desc->GetId());
  kernel_context->set_args_format(args_format);
  kernel_context->add_origin_op_index(op_index);
  kernel_context->set_op_index(op_index);
  kernel_context->set_kernel_type(kKernelTypeTe);
  kernel_context->set_is_flowtable(false);
  return SUCCESS;
}

TaskBuilderAdapterPtr TaskBuilder::CreateTaskAdapter(const ge::Node &node, TaskBuilderContext &task_context) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  int64_t imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_type)) {
    REPORT_FE_ERROR("[GenTask][CreateTaskAdapter][Node %s type %s] Get op imply_type failed, imply_type[%ld].",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str(), imply_type);
    return nullptr;
  }
  OpImplType op_impl_type = GetMainImplType<OpImplType>(imply_type);

  TaskBuilderAdapterPtr task_adapter =
          TaskBuilderAdapterFactory::Instance().CreateTaskBuilderAdapter(op_impl_type, node, task_context);
  if (task_adapter == nullptr) {
    FE_LOGE("Failed to create task adapter.");
    return task_adapter;
  }

  FE_LOGD("Create TaskBuilderAdapter success. OpName:%s, OpType:%s.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (task_adapter->Init() != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CreateTaskAdapter][Node %s, type %s] Failed to init TaskBuilderAdapter.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }

  return task_adapter;
}

void TaskBuilder::SetSlowContextSwitchAttr(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return;
  }
  if (!PlatformUtils::Instance().IsSupportContextSwitch()) {
    return;
  }
  (void)ge::AttrUtils::SetBool(op_desc, kAttrSlowContextSwitch, true);
  FE_LOGD("Set slow context switch for op[%s, %s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());

  std::vector<int64_t> output_mem_types;
  if (ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_mem_types)) {
    auto iter_l1 = std::find(output_mem_types.begin(), output_mem_types.end(), static_cast<int64_t>(RT_MEMORY_L1));
    auto iter_l2 = std::find(output_mem_types.begin(), output_mem_types.end(), static_cast<int64_t>(RT_MEMORY_L2));
    if (iter_l1 != output_mem_types.end() || iter_l2 != output_mem_types.end()) {
      (void)ge::AttrUtils::SetStr(op_desc, kAttrSwitchBufferType, kSwitchBufferTypeUB);
      FE_LOGD("Set switch buffer type[%s] for op[%s, %s].", kSwitchBufferTypeUB.c_str(),
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    }
  }
}

void TaskBuilder::GenerateCMOTask(const ge::Node &node, const bool pre_cmo_task,
                                  TaskBuilderContext &task_context, std::vector<domi::TaskDef> &task_defs) {
  CMOTaskBuilderPtr cmo_task_builder_ptr = nullptr;
  FE_MAKE_SHARED(cmo_task_builder_ptr = std::make_shared<CMOTaskBuilder>(), return);
  FE_CHECK(cmo_task_builder_ptr == nullptr, FE_LOGD("Cmo task builder ptr is null."), return);
  (void)cmo_task_builder_ptr->GenerateCMOTask(node, task_defs, task_context, pre_cmo_task);
}
}  // namespace fe
