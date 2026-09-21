/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <regex>
#include "common/fe_gentask_utils.h"
#include "common/platform_utils.h"
#include "runtime/rt_model.h"
#include "graph/utils/args_format_desc_utils.h"
#include "platform/platform_info.h"
#include "framework/common/taskdown_common.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"
#include "common/configuration.h"
#include "common/aicore_util_constants.h"

namespace fe {
namespace
{
const std::string TILING_SINK_OP = "_tiling_sink_op";
constexpr const char* kShortSocVersionAscend310P = "Ascend310P";
const int64_t LOC_EVENT = 2;

const std::unordered_set<int64_t> BUILT_IN_IMPLY_TYPE{
  EN_IMPL_HW_CONSTANT_CCE, EN_IMPL_HW_GENERAL_CCE, EN_IMPL_HW_TIK, EN_IMPL_HW_TBE,
  EN_IMPL_RL, EN_IMPL_PLUGIN_TBE, EN_IMPL_VECTOR_CORE_HW_TBE
};

const std::set<rtModelTaskType_t> op_task_list = {RT_MODEL_TASK_VECTOR_ALL_KERNEL, RT_MODEL_TASK_FFTS_PLUS_TASK,
                                                  RT_MODEL_TASK_ALL_KERNEL, RT_MODEL_TASK_KERNEL};

Status GetExecuteMode(const ge::Node &node, gert::ExecuteMode &exe_mode) {
  const  auto own_graph = node.GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(own_graph);
  if (own_graph->GetGraphUnknownFlag()) {
    exe_mode = gert::ExecuteMode::kDynamicExecute;
  } else {
    exe_mode = gert::ExecuteMode::kStaticOffloadExecute;
  }
  FE_LOGD("Node[%s, %s] GetExecuteMode success, exe_mode=%d", node.GetNamePtr(), node.GetTypePtr(), exe_mode);
  return SUCCESS;
}

bool GetOppSoFilePath(const ge::OpDescPtr op_desc, const std::string &opp_path, std::string &so_path) {
  (void)op_desc;
  const std::string fuzzy_str = ".*";
  std::string so_name = "";
  if (PlatformUtils::Instance().GetShortSocVersion() == kShortSocVersionAscend310P) {
    so_name = "Ascend310P-v" + fuzzy_str + "-libopmaster.so";
  } else {
    so_name = "Ascend-v" + fuzzy_str + "-libopmaster.so";
  }
  auto fuzzyMatch = [] (const std::string& str, const std::string& re_str) -> bool {
    std::regex re(re_str);
    bool res = false;
    try {
      res = std::regex_match(str, re);
    } catch (std::regex_error& e) {
      FE_LOGD("Regex match error %s", e.what());
    }
    return res;
  };
  DIR *dir = nullptr;
  struct dirent *dirp = nullptr;
  std::string dir_path = opp_path + "/built-in/op_impl/ai_core/tbe/op_master_device/lib/";
  dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    FE_LOGD("Unable to access directory [%s].", dir_path.c_str());
    return false;
  };

  std::string file_name = "";
  while ((dirp = readdir(dir)) != nullptr) {
    file_name.assign(dirp->d_name);
    if (fuzzyMatch(file_name, so_name)) {
      so_name = file_name;
      break;
    }
  }
  FE_LOGD("Finished finding files in directory %s", dir_path.c_str());
  closedir(dir);
  if (file_name != so_name) {
    FE_LOGD("Failed to get file name from directory [%s], file name is [%s], so name is [%s].",
            dir_path.c_str(), file_name.c_str(), so_name.c_str());
    return false;
  }
  so_path = dir_path + so_name;
  return true;
}

Status GetSoPath(const ge::Node &node, std::string &so_path) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  std::string opp_path;
  ge::PluginManager::GetOppPath(opp_path);
  if (IsCustomiseOp(*op_desc)) {
    std::string custom_op_file_path;
    (void)ge::AttrUtils::GetStr(op_desc, CUSTOM_OP_FILE_PATH, custom_op_file_path);
    so_path = custom_op_file_path + "/op_impl/ai_core/tbe/op_master_device/lib/libcust_opmaster.so";
  } else {
    std::string ops_path_name_prefix = "";
    if (!IsPrefixOpsPath(*op_desc, ops_path_name_prefix)) {
      bool opp_kernel_ret = false;
      if (op_desc->GetOppImplVersion() == ge::OppImplVersion::kOppKernel) {
        std::string opp_kernel_path = "";
        if (fe::Configuration::Instance(AI_CORE_NAME).GetOppLatestPath(opp_kernel_path) != SUCCESS) {
          FE_LOGD("Node[%s, %s]Failed to get opp latest path", op_desc->GetNamePtr(), op_desc->GetTypePtr());
        }
        opp_kernel_ret = GetOppSoFilePath(op_desc, opp_kernel_path, so_path);
        FE_LOGD("The ops impl version is oppkernel.");
      }
      if (opp_kernel_ret) {
        FE_LOGD("The versions of CANN and OPPKernel are different.");
      } else {
        if (!GetOppSoFilePath(op_desc, opp_path, so_path)) {
          FE_LOGE("Node[%s, %s]'s So path:%s does not exist", op_desc->GetNamePtr(),
                  op_desc->GetTypePtr(), so_path.c_str());
          return FAILED;
        }
      }
    } else {
      std::string so_name = "libtiling_device_" + ops_path_name_prefix + ".so";
      so_path = opp_path + "/built-in/op_impl/ai_core/tbe/op_tiling_device/lib/" + so_name;
    }
  }
  std::ifstream fin(so_path);
  if (!fin) {
    FE_LOGE("Node[%s, %s]'s So path:%s does not exist", op_desc->GetNamePtr(), op_desc->GetTypePtr(), so_path.c_str());
    return FAILED;
  } else {
    if (fin.is_open()) {
      fin.close();
    }
  }
  FE_LOGD("Node[%s, %s] Got so path successfully [%s]", op_desc->GetNamePtr(), op_desc->GetTypePtr(), so_path.c_str());
  return SUCCESS;
}

bool FeedParamDefInfo(const ge::Node &node, ParamDef &param) {
  FE_CHECK(strcpy_s(param.kernelName, sizeof(param.kernelName), "RunAicpuRpcSrvLaunch") != 0,
           FE_LOGI("Node[%s, %s]: unable to copy kernel name", node.GetNamePtr(), node.GetTypePtr()),
           return false);
  std::string so_path;
  FE_CHECK(GetSoPath(node, so_path) != SUCCESS,
          FE_LOGI("Node [%s, %s] unable to get the So path successfully", node.GetNamePtr(), node.GetTypePtr()),
          return false);
  FE_CHECK(strcpy_s(param.soName, sizeof(param.soName), so_path.c_str()) != 0,
          FE_LOGI("Node [%s, %s] unable to copy the so path", node.GetNamePtr(), node.GetTypePtr()),
          return false);
  param.is_custom = IsCustomiseOp(*(node.GetOpDesc()));
  std::string ops_path_name_prefix;
  param.is_prefix_ops_path = IsPrefixOpsPath(*(node.GetOpDesc()), ops_path_name_prefix);
  FE_LOGI("Node[%s, %s]Feed param def info success, is_custom[%d], is_prefix_ops_path[%d], prefix[%s]",
          node.GetNamePtr(), node.GetTypePtr(), param.is_custom, param.is_prefix_ops_path,
          ops_path_name_prefix.c_str());
  return true;
}

Status CreateTilingTask(const gert::ExeResGenerationContext* context, const ParamDef &param,
                          domi::TaskDef &aicpu_task) {
  const vector<gert::StreamInfo> stream_v = context->GetAttachedStreamInfos();
  FE_CHECK(stream_v.size() != 1, FE_LOGE("Node[%s, %s] stream_v size is not equal to 1", context->GetNodeName(),
           context->GetNodeType()), return FAILED);
  const int64_t stream_id = stream_v[0].stream_id;
  aicpu_task.set_type(RT_MODEL_TASK_PREPROCESS_KERNEL);
  aicpu_task.set_stream_id(stream_id);

  std::string task_args;
  task_args.append(reinterpret_cast<const char *>(&param), sizeof(param));
  auto kernel_def = aicpu_task.mutable_kernel();
  kernel_def->set_args(task_args.data(), task_args.size());
  kernel_def->set_args_size(task_args.size());
  kernel_def->set_so_name(param.soName);
  kernel_def->set_kernel_name(param.kernelName);

  auto mutable_context = kernel_def->mutable_context();
  auto kernel_type = (param.is_custom || param.is_prefix_ops_path) ?
                      ge::ccKernelType::CUST_AI_CPU : ge::ccKernelType::AI_CPU;
  mutable_context->set_kernel_type(static_cast<uint32_t>(kernel_type));
  mutable_context->set_op_index(context->GetOpId());

  ge::ArgsFormatDescUtils args_desc_util;
  vector<ge::ArgDesc> arg_descs;
  args_desc_util.AppendTilingContext(arg_descs, ge::TilingContextSubType::TILING_CONTEXT);
  args_desc_util.Append(arg_descs, ge::AddrType::OP_TYPE);
  args_desc_util.Append(arg_descs, ge::AddrType::PLACEHOLDER);
  if (param.is_custom || param.is_prefix_ops_path) {
    auto workspace_bytes = context->GetWorkspaceBytes();
    auto workspace_index = workspace_bytes.empty() ? 0 : workspace_bytes.size() - 1;
    args_desc_util.Append(arg_descs, ge::AddrType::WORKSPACE, workspace_index);
    FE_LOGI("Node[%s, %s] tiling sink op append WORKSPACE[%lu]", context->GetNodeName(), context->GetNodeType(),
            workspace_index);
  }
  const std::string args_format = args_desc_util.ToString(arg_descs);
  FE_LOGD("Aicpu tiling task args format is %s.", args_format.c_str());
  mutable_context->set_args_format(args_format);
  return SUCCESS;
}

Status CreateRefreshTask(const gert::ExeResGenerationContext* context, domi::TaskDef &task) {
  const vector<gert::StreamInfo> stream_v = context->GetAttachedStreamInfos();
  FE_CHECK(stream_v.size() != 1, FE_LOGE("Node[%s, %s] stream_v size is not equal to 1", context->GetNodeName(),
           context->GetNodeType()), return FAILED);
  const int64_t stream_id = stream_v[0].stream_id;
  task.set_type(RT_MODEL_TASK_UPDATE);
  task.set_stream_id(stream_id);
  task.mutable_update_pc_task()->set_op_index(context->GetOpId());
  task.mutable_update_pc_task()->set_stream_id(stream_id);

  ge::ArgsFormatDescUtils args_desc_util;
  vector<ge::ArgDesc> arg_descs;
  args_desc_util.AppendTilingContext(arg_descs, ge::TilingContextSubType::TILING_KEY);
  args_desc_util.AppendTilingContext(arg_descs, ge::TilingContextSubType::BLOCK_DIM);
  task.mutable_update_pc_task()->set_args_format(args_desc_util.ToString(arg_descs));
  return SUCCESS;
}

Status CreateRecordTask(const gert::ExeResGenerationContext* context, domi::TaskDef &task) {
  std::vector<gert::SyncResInfo> sync_res_info_v = context->GetSyncResInfos();
  FE_CHECK(sync_res_info_v.size() != 1, FE_LOGE("Node[%s, %s]: sync_res_info_v size is not 1", context->GetNodeName(),
           context->GetNodeType()), return FAILED);
  int32_t event_id = sync_res_info_v[0].sync_res_id;
  task.mutable_event_ex()->set_op_index(context->GetOpId());
  task.set_event_id(event_id);
  FE_LOGI("Node[%s, %s] event_id is %d", context->GetNodeName(), context->GetNodeType(), event_id);
  task.set_type(RT_MODEL_TASK_EVENT_RECORD);

  const vector<gert::StreamInfo> stream_v = context->GetAttachedStreamInfos();
  FE_CHECK(stream_v.size() != 1, FE_LOGE("Node[%s, %s] stream_v size is not equal to 1", context->GetNodeName(),
           context->GetNodeType()), return FAILED);
  const int64_t stream_id = stream_v[0].stream_id;
  task.set_stream_id(stream_id);
  return SUCCESS;
}

Status CreateWaitTask(const gert::ExeResGenerationContext* context, domi::TaskDef &task) {
  std::vector<gert::SyncResInfo> sync_res_info_v = context->GetSyncResInfos();
  FE_CHECK(sync_res_info_v.size() != 1, FE_LOGE("Node[%s, %s] sync_res_info_v size is not 1", context->GetNodeName(),
           context->GetNodeType()), return FAILED);
  int32_t event_id = sync_res_info_v[0].sync_res_id;
  task.mutable_event_ex()->set_op_index(context->GetOpId());
  task.set_event_id(event_id);
  FE_LOGI("Node[%s, %s] event_id is %d", context->GetNodeName(), context->GetNodeType(), event_id);
  task.set_type(RT_MODEL_TASK_EVENT_WAIT);

  const int64_t stream_id = context->GetStreamId();
  task.set_stream_id(stream_id);
  return SUCCESS;
}

Status CreateNopTask(const gert::ExeResGenerationContext* context, domi::TaskDef &task) {
  task.set_type(RT_MODEL_TASK_NOP);
  const int64_t stream_id = context->GetStreamId();
  task.set_stream_id(stream_id);
  return SUCCESS;
}

Status GenerateTaskForTilingSink(const ge::Node &node, const gert::ExeResGenerationContext* context,
                                 std::vector<domi::TaskDef> &task_defs) {
  ParamDef param;
  if (!FeedParamDefInfo(node, param)) {
    return FAILED;
  }
  return GenerateTaskForSinkOp(context, param, task_defs);
}

Status ProcessFftsPlusTask(const gert::ExeResGenerationContext* context, domi::TaskDef &task) {
  auto op_name = context->GetNodeName();
  auto op_type = context->GetNodeType();
  auto mixl2_task_def = task.mutable_ffts_plus_task();
  // get mix task idx
  const int32_t ctx_num = mixl2_task_def->ffts_plus_ctx_size();
  int32_t mix_ctx_idx = 0;
  for (; mix_ctx_idx < ctx_num; ++mix_ctx_idx) {
    if (mixl2_task_def->ffts_plus_ctx(mix_ctx_idx).op_type() != domi::FftsPlusCtxDef::ATOMIC) {
      break;
    }
  }
  FE_CHECK(mix_ctx_idx == ctx_num,
           FE_LOGE("Node[%s, %s]No valid mixl2 context in %d context(s)", op_name, op_type, ctx_num),
           return FAILED);
  // get insert pos
  auto mixl2_task_ctx = mixl2_task_def->mutable_ffts_plus_ctx(mix_ctx_idx)->mutable_mix_aic_aiv_ctx();
  std::vector<ge::ArgDesc> arg_descs;
  const std::string args_format = mixl2_task_ctx->args_format();
  FE_CHECK(ge::ArgsFormatDescUtils::Parse(args_format, arg_descs) != SUCCESS || arg_descs.empty(),
           FE_LOGE("Node[%s, %s] failed to parse, args_format[%s]", op_name, op_type, args_format.c_str()),
           return FAILED);
  int64_t insert_idx = static_cast<int64_t>(arg_descs.size()) - 1;
  for (; insert_idx >= 0; --insert_idx) {
    if (arg_descs[insert_idx].addr_type == ge::AddrType::WORKSPACE) {
      if ((static_cast<size_t>(insert_idx) != arg_descs.size() - 1) &&
          (arg_descs[insert_idx + 1].addr_type == ge::AddrType::TILING_FFTS ||
          arg_descs[insert_idx].addr_type == ge::AddrType::TILING)) {
        arg_descs[insert_idx + 1].addr_type = ge::AddrType::TILING_CONTEXT;
        arg_descs[insert_idx + 1].ir_idx = static_cast<int32_t>(ge::TilingContextSubType::TILING_DATA);
      } else {
        // 插入一个新的
        FE_LOGD("Node [%s, %s] inserted new after workspace", op_name, op_type);
        arg_descs.insert(arg_descs.begin() + insert_idx + 1,
          {ge::AddrType::TILING_CONTEXT, static_cast<int32_t>(ge::TilingContextSubType::TILING_DATA), false, {0}});
        // 刷新args size
        mixl2_task_ctx->add_task_addr(0x0);
        mixl2_task_def->set_addr_size(mixl2_task_def->addr_size() + 1);
      }
      break;
    }
    if (insert_idx == 0) {
      FE_LOGE("Node [%s, %s] Origin format is %s, failed to find workspace", op_name, op_type, args_format.c_str());
      return FAILED;
    }
  }
  FE_LOGD("Node [%s, %s]: Original format is %s, insertion position is %ld", op_name, op_type, args_format.c_str(),
          insert_idx);
  const std::string new_args_format = ge::ArgsFormatDescUtils::Serialize(arg_descs);
  mixl2_task_ctx->set_args_format(new_args_format);
  return SUCCESS;
}

Status ProcessMixAicoreTask(const gert::ExeResGenerationContext* context, domi::TaskDef &task) {
  auto op_name = context->GetNodeName();
  auto op_type = context->GetNodeType();
  auto task_def = task.mutable_kernel_with_handle();
  auto task_ctx = task_def->mutable_context();
  const std::string args_format = task_ctx->args_format();
  std::vector<ge::ArgDesc> arg_descs;
  FE_CHECK(ge::ArgsFormatDescUtils::Parse(args_format, arg_descs) != SUCCESS || arg_descs.empty(),
           FE_LOGE("Node [%s, %s] failed to parse, args_format [%s].", op_name, op_type, args_format.c_str()),
           return FAILED);
  int64_t insert_idx = static_cast<int64_t>(arg_descs.size()) - 1;
  for (; insert_idx >= 0; --insert_idx) {
    if (arg_descs[insert_idx].addr_type == ge::AddrType::WORKSPACE) {
      if ((static_cast<size_t>(insert_idx) != arg_descs.size() - 1) &&
          (arg_descs[insert_idx + 1].addr_type == ge::AddrType::TILING_FFTS ||
          arg_descs[insert_idx].addr_type == ge::AddrType::TILING)) {
        arg_descs[insert_idx + 1].addr_type = ge::AddrType::TILING_CONTEXT;
        arg_descs[insert_idx + 1].ir_idx = static_cast<int32_t>(ge::TilingContextSubType::TILING_DATA);
      } else {
        // 插入一个新的
        FE_LOGD("Node [%s, %s] inserted into new workspace.", op_name, op_type);
        arg_descs.insert(arg_descs.begin() + insert_idx + 1,
          {ge::AddrType::TILING_CONTEXT, static_cast<int32_t>(ge::TilingContextSubType::TILING_DATA), false, {0}});
      }
      break;
    }
    if (insert_idx == 0) {
      FE_LOGE("Node [%s, %s] Origin format is %s, failed to find workspace", op_name, op_type, args_format.c_str());
      return FAILED;
    }
  }
  FE_LOGD("Node[%s, %s] Origin format is %s, insert position is %ld", op_name, op_type, args_format.c_str(), insert_idx);
  // set args into task
  const std::string new_args_format = ge::ArgsFormatDescUtils::Serialize(arg_descs);
  task_ctx->set_args_format(new_args_format);
  return SUCCESS;
}

Status PreProcessTasks(const gert::ExeResGenerationContext* context, std::vector<domi::TaskDef> &tasks, size_t &i) {
  auto op_name = context->GetNodeName();
  auto op_type = context->GetNodeType();
  for (; i < tasks.size(); ++i) {
    if (tasks[i].type() == RT_MODEL_TASK_FFTS_PLUS_TASK) {
      if (ProcessFftsPlusTask(context, tasks[i]) == FAILED) return FAILED;
      break; // 找到当前aicoretask，直接break
    }
    if (tasks[i].type() == RT_MODEL_TASK_ALL_KERNEL || tasks[i].type() == RT_MODEL_TASK_VECTOR_ALL_KERNEL) {
      if (ProcessMixAicoreTask(context, tasks[i]) == FAILED) return FAILED;
      break; // 找到当前aicoretask，直接break
    }
  }
  if (i == tasks.size()) {
    FE_LOGE("Node [%s, %s] failed to find a valid task", op_name, op_type);
    return FAILED;
  }
  return SUCCESS;
}
} // namespace

bool CheckTilingSink(const ge::Node &node) {
  gert::ExecuteMode exe_mode;
  auto ret = GetExecuteMode(node, exe_mode);
  if (ret != SUCCESS || exe_mode != gert::ExecuteMode::kStaticOffloadExecute) {
    FE_LOGD("Node [%s, %s] exe_mode is not static, skipping tiling sink.", node.GetNamePtr(), node.GetTypePtr());
    return false;
  }
  auto op_desc = node.GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  bool is_support_tiling_sink = false;
  (void)ge::AttrUtils::GetBool(op_desc, TILING_SINK_OP, is_support_tiling_sink);
  if (!is_support_tiling_sink) {
    FE_LOGD("Node[%s, %s]: TILING_SINK_OP is false, skipping tiling sink.", node.GetNamePtr(), node.GetTypePtr());
    return false;
  }
  std::string compile_info_json;
  (void)ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  if (compile_info_json.empty()) {
    FE_LOGD("Node [%s, %s]: COMPILE_INFO_JSON is empty, skipping tiling sink.", node.GetNamePtr(), node.GetTypePtr());
    return false;
  }
  return true;
}

ge::Status CreateTilingTaskSuperKernel(const gert::ExeResGenerationContext* context,
    domi::TaskDef &aicpu_task, const ParamDef &param) {
  FE_CHECK(CreateTilingTask(context, param, aicpu_task) != SUCCESS,
           FE_LOGE("Creat tiling task failed."), return FAILED);
  std::vector<gert::SyncResInfo> sync_res_info_v = context->GetSyncResInfos();
  FE_CHECK(sync_res_info_v.size() != 1,
           FE_LOGE("stream_v size is not equal to 1"), return FAILED);
  int32_t event_id = sync_res_info_v[0].sync_res_id;
  ge::ArgsFormatDescUtils args_des_util;
  vector<ge::ArgDesc> arg_descs;
  args_des_util.AppendTilingContext(arg_descs, ge::TilingContextSubType::TILING_CONTEXT);
  args_des_util.Append(arg_descs, ge::AddrType::OP_TYPE);
  args_des_util.Append(arg_descs, ge::AddrType::EVENT_ADDR, event_id);
  const std::string &args_format = args_des_util.ToString(arg_descs);
  auto kernel_def = aicpu_task.mutable_kernel();
  auto mutable_context = kernel_def->mutable_context();
  mutable_context->set_args_format(args_format);
  FE_LOGD("format is %s", args_format.c_str());
  return SUCCESS;
}
 
ge::Status GenerateTaskSuperKernel(const gert::ExeResGenerationContext* context,
                                   std::vector<domi::TaskDef> &tasks, const ParamDef &param) {
  FE_LOGD("Start to generate super kernel task for FIA.");
  const std::vector<gert::SyncResInfo> &sync_res_info_v = context->GetSyncResInfos();
  FE_CHECK(sync_res_info_v.size() != 1,
           FE_LOGE("stream_v size is not equal to 1"), return FAILED);
  int32_t eventId = sync_res_info_v[0].sync_res_id;
  int64_t index = -1L;
  // find aicore task
  for (int64_t i = static_cast<int64_t>(tasks.size()) - 1; i >= 0; i--) {
    if ((tasks[i].type() == RT_MODEL_TASK_KERNEL) || (tasks[i].type() == RT_MODEL_TASK_ALL_KERNEL)) {
      index = i;
      break;
    }
  }
  FE_CHECK(index < 0,
           FE_LOGE("FIA failed to find aicore task."),
           return FAILED);
  FE_LOGD("FIA aicore index: %ld.", index);
  // get aicore context
  domi::KernelContext *kernel_context;
  if (tasks[index].type() == RT_MODEL_TASK_KERNEL) {
    auto kernel_def = tasks[index].mutable_kernel();
    FE_CHECK(kernel_def == nullptr,
             FE_LOGE("kernel_def for aicore task is nullptr."),
             return FAILED);
    kernel_context = kernel_def->mutable_context();
  } else if (tasks[index].type() == RT_MODEL_TASK_ALL_KERNEL) {
    auto kernelWithHandle = tasks[index].mutable_kernel_with_handle();
    FE_CHECK(kernelWithHandle == nullptr,
             FE_LOGE("The kernel_def for the aicore task is nullptr."),
             return FAILED);
    kernel_context = kernelWithHandle->mutable_context();
  } else {
    FE_LOGE("Invalid task type [%d].", tasks[index].type());
    return FAILED;
  }
  FE_CHECK(kernel_context == nullptr,
           FE_LOGE("Kernel context for aicore task is nullptr."),
           return FAILED);
  const std::string argsFormat = kernel_context->args_format();
  std::vector<ge::ArgDesc> argDescs;
  FE_CHECK(ge::ArgsFormatDescUtils::Parse(argsFormat, argDescs) != SUCCESS || argDescs.empty(),
           FE_LOGE("Failed to parse, argsFormat:[%s]", argsFormat.c_str()),
           return FAILED);
  int64_t argIndex = static_cast<int64_t>(argDescs.size()) - 1;
  for (; argIndex >= 0; argIndex--) {
    if (argDescs[argIndex].addr_type == ge::AddrType::WORKSPACE) {
      if ((static_cast<size_t>(argIndex) != argDescs.size() - 1) &&
          (argDescs[argIndex + 1].addr_type == ge::AddrType::TILING_FFTS ||
          argDescs[argIndex + 1].addr_type == ge::AddrType::TILING)) {
        argDescs[argIndex + 1].addr_type = ge::AddrType::TILING_CONTEXT;
        argDescs[argIndex + 1].ir_idx = static_cast<int32_t>(ge::TilingContextSubType::TILING_DATA);
      } else {
        FE_LOGD("insert new after workspace");
        argDescs.insert(argDescs.begin() + argIndex + 1,
          {ge::AddrType::TILING_CONTEXT, static_cast<int32_t>(ge::TilingContextSubType::TILING_DATA), false, {0}});
      }
      // order:tiling_data / tiling_key / block_dim / event_addr
      argDescs.insert(argDescs.begin() + argIndex + LOC_EVENT,
        {ge::AddrType::EVENT_ADDR, eventId, false, {0}});
      argDescs.insert(argDescs.begin() + argIndex + LOC_EVENT,
        {ge::AddrType::TILING_CONTEXT, static_cast<int32_t>(ge::TilingContextSubType::BLOCK_DIM), false, {0}});
      argDescs.insert(argDescs.begin() + argIndex + LOC_EVENT,
        {ge::AddrType::TILING_CONTEXT, static_cast<int32_t>(ge::TilingContextSubType::TILING_KEY), false, {0}});
      break;
    }
    if (argIndex == 0) {
      FE_LOGE("Origin format is %s, find workspace fail",
              argsFormat.c_str());
      return FAILED;
    }
  }
  FE_LOGD("Origin format is %s, insertion position is %ld.", argsFormat.c_str(), argIndex);
  // set args into task
  const std::string newArgsFormat = ge::ArgsFormatDescUtils::Serialize(argDescs);
  kernel_context->set_args_format(newArgsFormat);
  FE_LOGD("new format is %s, insertion position is %ld.", newArgsFormat.c_str(), argIndex);
 
  domi::TaskDef tilingTask;
  FE_CHECK(CreateTilingTaskSuperKernel(context, tilingTask, param) != SUCCESS,
            FE_LOGE("create tiling task failed"),
            return FAILED);
  tasks.insert(tasks.begin() + index, tilingTask);
  return SUCCESS;
}

Status GenerateTaskForSinkOp(const gert::ExeResGenerationContext* context, const ParamDef &param,
                             std::vector<domi::TaskDef> &tasks) {
  auto op_name = context->GetNodeName();
  auto op_type = context->GetNodeType();
  bool is_spk_scene = false;
  int64_t spk_cnt = -1;
  is_spk_scene = context->GetIntAttrVal("_ascendc_sp_cnt", spk_cnt);
  FE_LOGI("Node [%s, %s] starts to generate tasks for the tiling sink, sk_flag [%d].", op_name, op_type, is_spk_scene);
  if (is_spk_scene) {
    GenerateTaskSuperKernel(context, tasks, param);
    return SUCCESS;
  }
  size_t i = 0UL;
  if (PreProcessTasks(context, tasks, i) == FAILED) return FAILED;

  domi::TaskDef tiling_task;
  domi::TaskDef refresh_task;
  domi::TaskDef record_task;
  domi::TaskDef wait_task;
  domi::TaskDef nop_task;
  FE_CHECK(CreateTilingTask(context, param, tiling_task) != SUCCESS,
           FE_LOGE("Node [%s, %s] failed to create tiling task.", op_name, op_type), return FAILED);
  FE_CHECK(CreateRefreshTask(context, refresh_task) != SUCCESS,
           FE_LOGE("Node [%s, %s] creat refresh task failed.", op_name, op_type), return FAILED);
  FE_CHECK(CreateRecordTask(context, record_task) != SUCCESS,
           FE_LOGE("Node [%s, %s] failed to create record task.", op_name, op_type), return FAILED);
  FE_CHECK(CreateWaitTask(context, wait_task) != SUCCESS,
           FE_LOGE("Node [%s, %s] Create wait task failed.", op_name, op_type), return FAILED);
  CreateNopTask(context, nop_task);
  // tiling -> wait -> nops -> aicore -> refresh -> record
  tasks.insert(tasks.begin() + i + 1, record_task);
  tasks.insert(tasks.begin() + i + 1, refresh_task);

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != SUCCESS) {
    FE_LOGE("Failed to retrieve platform information due to missing SoC version.");
    return FAILED;
  }

  std::string nop_num_str;
  platform_infos.GetPlatformResWithLock("SoCInfo", "prefetch_num", nop_num_str);
  uint32_t nop_num = atoi(nop_num_str.c_str());
  // 910b应该为8，310p应该为5
  FE_LOGD("Node [%s, %s] finished generating tasks for the tiling sink, with prefetch_num=%u.", op_name, op_type, nop_num);

  // 硬件预取指令与notify有时间差，根据GE经验插入空task
  for(uint32_t j = 0; j < nop_num; ++j) {
    tasks.insert(tasks.begin() + i, nop_task);
  }
  tasks.insert(tasks.begin() + i, wait_task);
  tasks.insert(tasks.begin() + i, tiling_task);
  return SUCCESS;
}

bool IsCustomiseOp(const ge::OpDesc &op_desc) {
  int64_t tmp_implt_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, tmp_implt_type)) {
    FE_LOGW("Node [%s, %s] failed to get implementation type", op_desc.GetNamePtr(), op_desc.GetTypePtr());
    return false;
  }
  int64_t impl_type = GetMainImplType<int64_t>(tmp_implt_type);
  if (BUILT_IN_IMPLY_TYPE.count(impl_type) != 0) {
    return false;
  }
  return true;
}

bool IsPrefixOpsPath(const ge::OpDesc &op_desc, std::string &ops_path_name_prefix) {
  ops_path_name_prefix = "";
  (void)ge::AttrUtils::GetStr(op_desc, OPS_PATH_NAME_PREFIX, ops_path_name_prefix);
  bool ret = ops_path_name_prefix != "";
  if (ret) {
    size_t pos = ops_path_name_prefix.find("_");
    if (pos != string::npos) {
      ops_path_name_prefix = ops_path_name_prefix.substr(static_cast<size_t>(pos + 1));
      FE_LOGD("Node[%s, %s] after substr, ops_path_name_prefix:[%s]",
      op_desc.GetNamePtr(), op_desc.GetTypePtr(), ops_path_name_prefix.c_str());
    } else {
      FE_LOGW("Node[%s, %s] ops_path_name_prefix:[%s] is not in the expected format",
      ops_path_name_prefix.c_str(), op_desc.GetNamePtr(), op_desc.GetTypePtr());
    }
  }
  return ret;
}

Status GenerateOpExtTask(const ge::Node &node, const bool is_tiling_sink, std::vector<domi::TaskDef> &task_defs,
                         bool &reg_flag) {
  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(const_cast<ge::Node&>(node));
  FE_CHECK_NOTNULL(res_ptr_holder);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  if (is_tiling_sink) {
    FE_LOGI("Node[%s, %s] Start GenerateTaskForTilingSink", node.GetNamePtr(), node.GetTypePtr());
    return GenerateTaskForTilingSink(node, op_exe_res_ctx, task_defs);
  }
  const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(ge::OppImplVersion::kOpp));
  if (space_registry == nullptr) {
    FE_LOGW("Node [%s, %s]: Failed to get default space registry", node.GetNamePtr(), node.GetTypePtr());
    return SUCCESS;
  }
  auto funcs_ptr = space_registry->GetOpImpl(node.GetType().c_str());
  if (funcs_ptr == nullptr || funcs_ptr->gen_task == nullptr) {
    reg_flag = false;
    return SUCCESS;
  }
  reg_flag = true;
  FE_LOGD("Node[%s][%s] find new gen task func with ori task size:%zu, reg_flag:%d",
          node.GetNamePtr(), node.GetTypePtr(), task_defs.size(), reg_flag);
  std::vector<std::vector<uint8_t>> serialize_tasks;
  std::vector<int> op_task_defs;
  int index = 0;
  for (const auto &task : task_defs) {
    rtModelTaskType_t task_type = static_cast<rtModelTaskType_t>(task.type());
    if (op_task_list.count(task_type) == 0U) {
      index++;
      continue;
    }
    op_task_defs.push_back(index++);
    size_t pb_size = task.ByteSizeLong();
    FE_CHECK(pb_size == 0U, FE_LOGE("pb_size is 0."), return FAILED);
    std::vector<uint8_t> pb_buff(pb_size, 0);
    FE_CHECK(task.SerializeToArray(pb_buff.data(), pb_size) != true, FE_LOGE("Failed to serialize PB."), return FAILED);
    serialize_tasks.emplace_back(std::move(pb_buff));
  }
  if (funcs_ptr->gen_task(op_exe_res_ctx, serialize_tasks) != SUCCESS) {
    FE_LOGE("Node [%s][%s] failed to generate extra task.", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  FE_LOGI("Node[%s][%s] after gen task size is:%zu.", node.GetNamePtr(), node.GetTypePtr(), serialize_tasks.size());
  FE_CHECK(serialize_tasks.size() == 0U, FE_LOGE("Serialize pb size is 0."), return FAILED);
  auto task_defs_tmp = task_defs;
  task_defs.clear();
  if (task_defs_tmp.size() > 0 && op_task_defs.size() > 0) {
    for (int i = 0; i < op_task_defs[0]; i++) {
      FE_LOGI("Add the head tasks");
      task_defs.emplace_back(std::move(task_defs_tmp[i]));
    }
  }

  for (const auto &serialize_task : serialize_tasks) {
    domi::TaskDef task;
    FE_CHECK(task.ParseFromArray(serialize_task.data(), serialize_task.size()) != true,
             FE_LOGE("Parse pb failed."), return FAILED);
    FE_LOGD("After gen task emplace type:%u.", task.type());
    task_defs.emplace_back(std::move(task));
  }

  if (task_defs_tmp.size() > 0 && op_task_defs.size() > 0) {
    size_t last_index = op_task_defs[op_task_defs.size() - 1] + 1;
    for (size_t i = last_index; i < task_defs_tmp.size(); i++) {
      FE_LOGI("Add the tail tasks");
      task_defs.emplace_back(std::move(task_defs_tmp[i]));
    }
  }
  return SUCCESS;
}
} // namespace fe
