/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_kernel_builder.h"

#include <mutex>
#include "util/constant.h"
#include "util/log.h"
#include "util/tf_util.h"
#include "error_code/error_code.h"
#include "ir2tf/ir2tf_parser_factory.h"
#include "runtime/kernel.h"
#include "base/err_msg.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "rt_error_codes.h"
#include "aicpu_task_struct.h"
#include"util/util.h"
#include "common/sgt_slice_type.h"
#include "tf_kernel_info/tf_kernel_info.h"

using domi::tensorflow::NodeDef;

namespace {
const uint64_t kNum = 2;
static domi::TaskDef g_task_def;
static int64_t g_op_index;
static std::string g_task_info;
static std::shared_ptr<STR_FWK_OP_KERNEL> g_str_fwkop_kernel_ptr;
constexpr uint32_t kKernelTypeFwk = 1;
constexpr int32_t OP_TYPE_FOUR = 4;
constexpr uint32_t kBasicTfOpSqeNumber = 1;    /* sqe number of basic operation */
constexpr uint32_t kAsyncTfOpSqeNumber = 3;    /* sqe number of async operation */
const std::string kEngineNameAicputfFfts = "ffts_plus_aicpu_tf";
}

namespace aicpu {
KernelBuilderPtr TfKernelBuilder::instance_ = nullptr;

inline KernelBuilderPtr TfKernelBuilder::Instance() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    instance_.reset(new (std::nothrow) TfKernelBuilder);
  });
  return instance_;
}

ge::Status TfKernelBuilder::Initialize() {
  return KernelBuilder::Initialize();
}

/**
 * blocks' size are needed:
 *  1.STR_FWK_OP_KERNEL, this is the task's API struct;
 *  2.InputOuputBuf, defined by protobuf, the struct is KernelRunParam, inside the
 *    struct, input and output buffer's pointer are defined;
 *  3.NodeDefBuf, defined by protobuf, definition is from tensorflow, data is from
 *    GE's graph, need to do the ir transfer;
 *  4.FuncDef, defined by protobuf, for the fused graph.
 */
ge::Status TfKernelBuilder::CalcOpRunningParam(const ge::Node &node) const {
  AICPUE_LOGI("TFKernel's op[%s] run CalcOpRunningParam", node.GetType().c_str());
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, INPUT_PARAM_NULL)

  // check whether the WorkspaceBytes is set
  int64_t workspace_size = 0;
  std::vector<int64_t> workspace_bytes = op_desc_ptr->GetWorkspaceBytes();
  if ((workspace_bytes.empty()) || (workspace_bytes[0] <= 0)) {
    // calc and set WorkspaceBytes
    AICPU_CHECK_RES_WITH_LOG(CalcWorkspaceSize(node, workspace_size),
        "Call TfKernelBuilder::CalcWorkspaceSize function failed, op[%s].",
        node.GetName().c_str())
    op_desc_ptr->SetWorkspaceBytes({workspace_size});
  } else {
    workspace_size = workspace_bytes[0];
    AICPUE_LOGI("Op type[%s] Workspace size already exist, workspace_size is [%ld]",
                node.GetType().c_str(), workspace_size);
  }

  AICPU_CHECK_RES_WITH_LOG(SetOutPutsSize(op_desc_ptr),
      "Call SetOutPutsSize function failed, op[%s].",
      node.GetName().c_str())
  // Set workspace memory reuse flag
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetListBool(op_desc_ptr, kWorkspaceReuseFlag, {false}),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetListBool Failed to set attr[%s], op[%s].",
          kWorkspaceReuseFlag.c_str(), node.GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)

  AICPU_CHECK_RES(SetAttrResource(node.GetName(), op_desc_ptr));

  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameThreadScopeId)) {
    AICPU_CHECK_RES(CalFftsMaxThread(op_desc_ptr));
  }

  if (node.GetName().find("OptionalFromValue") != node.GetName().npos) {
    bool no_reuse_mem_flag = true;
    (void)ge::AttrUtils::SetBool(op_desc_ptr, "no_reuse_mem_flag", no_reuse_mem_flag);
    AICPUE_LOGI("op:%s name:%s set no_reuse_mem_flag true.", op_desc_ptr->GetName().c_str(), node.GetName().c_str());
  }

  AICPUE_LOGI("Op type[%s] Calc the Op running param successfully, workspace_size is [%ld]",
              node.GetType().c_str(), workspace_size);
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::BuildFFtsKernelRunParam(const ge::OpDesc &op_desc,
                                                    FWKAdapter::KernelRunParam &kernel_run_param,
                                                    const uint32_t &index,
                                                    const FftsPlusInfo &ffts_info) const {
  // Construct input's content
  std::set<std::string> ref_input_set;
  std::string op_type = op_desc.GetType();
  std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfParserFactory::Instance().CreateIRParser(op_type);
  AICPU_CHECK_NOTNULL_ERRCODE(parser, ErrorCode::INPUT_PARAM_NULL);
  parser->GetRefInputSet(op_type, ref_input_set);

  size_t input_size = op_desc.GetInputsSize();
  aicpu::State state;
  for (size_t i = 0; i < input_size; i++) {
    FWKAdapter::TensorDataInfo *input_tensor = kernel_run_param.add_input();
    ge::GeTensorDesc ge_tensor_desc = op_desc.GetInputDesc(i);
    std::string input_name = op_desc.GetInputNameByIndex(i);
    bool is_ref = false;
    auto iter = ref_input_set.find(input_name);
    if (iter != ref_input_set.end()) {
      is_ref = true;
    }
    AICPUE_LOGD("Op type[%s], input name[%s], is ref[%d]",
                op_type.c_str(), input_name.c_str(), is_ref);
    state = SetTensorDataInfo(ge_tensor_desc, input_tensor,
                              ffts_info.thread_input_shape[index][i], is_ref);
    if (state.state != ge::SUCCESS) {
      state.msg = Stringcat(i, "th input's ", state.msg,
          ", op[", op_desc.GetName(), "].");
      AICPU_REPORT_INNER_ERR_MSG("%s", state.msg.c_str());
      return state.state;
    }
  }

  // Construct output's content
  size_t output_size = op_desc.GetOutputsSize();
  for (size_t i = 0; i < output_size; i++) {
    FWKAdapter::TensorDataInfo *output_tensor = kernel_run_param.add_output();
    ge::GeTensorDesc ge_tensor_desc = op_desc.GetOutputDesc(i);
    state = SetTensorDataInfo(ge_tensor_desc, output_tensor,
                              ffts_info.thread_output_shape[index][i], false);
    if (state.state != ge::SUCCESS) {
      state.msg = Stringcat(i, "th output's ",
          state.msg, ", op[", op_desc.GetName(), "].");
      AICPU_REPORT_INNER_ERR_MSG("%s", state.msg.c_str());
      return state.state;
    }
  }
  return ge::SUCCESS;
}

aicpu::State TfKernelBuilder::SetTensorDataInfo(const ge::GeTensorDesc &ge_tensor_desc,
                                                FWKAdapter::TensorDataInfo *tensor_data_info,
                                                const std::vector<int64_t> &tensor_shape,
                                                bool is_ref,
                                                bool skip_dim_check) const {
  ge::DataType data_type = ge_tensor_desc.GetDataType();
  uint32_t tf_data_type = static_cast<uint32_t>(ConvertGeDataType2TfDataType(data_type, is_ref));
  tensor_data_info->set_dtype(tf_data_type);

  tensor_data_info->set_data_addr(ULLONG_MAX);
  uint32_t dim_shape = static_cast<uint32_t>(tensor_shape.size());
  if (!skip_dim_check) {
    for (uint32_t i = 0; i < dim_shape; i++) {
      int64_t dim = tensor_shape[i];
      bool is_invalid_dim = ((dim < 0) &&
                             (dim != ge::UNKNOWN_DIM) &&
                             (dim != ge::UNKNOWN_DIM_NUM));
      if (is_invalid_dim) {
        std::string err_msg =  Stringcat("dim[", i,
            "] is invalid, shape is [", dim, "].");
        aicpu::State state(GE_SHAPE_SIZE_INVAILD, err_msg);
        return state;
      }
      tensor_data_info->add_dim(dim);
    }
  } else {
    AICPUE_LOGI("Skip_dim_check for unknown shape");
  }
  return aicpu::State(ge::SUCCESS);
}

ge::Status TfKernelBuilder::UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) {
  std::shared_ptr<ge::OpDesc> op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)

  for (size_t index = 0; index < tasks.size(); index++) {
    tasks[index].set_stream_id(static_cast<uint32_t>(op_desc_ptr->GetStreamId()));
  }

  AICPUE_LOGI("TFKernel's op[%s], op type[%s] run UpdateTask, stream_id=%ld, task def size:%zu.",
    node.GetName().c_str(), node.GetType().c_str(), op_desc_ptr->GetStreamId(), tasks.size());
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::GenerateTask(const ge::Node &node, const ge::RunContext &run_context,
                                         std::vector<domi::TaskDef> &tasks) {
  AICPUE_LOGI("TFKernel's op[%s], op type[%s] run GenerateTask. ", node.GetName().c_str(), node.GetType().c_str());
  // Check the input data
  std::shared_ptr<ge::OpDesc> op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameThreadScopeId)) {
    AICPUE_LOGI("TFKernel's op[%s], op type[%s] run GenerateFftsPlusTask. ",
                node.GetName().c_str(), node.GetType().c_str());
    return GenerateFftsPlusTask(node, run_context);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  g_op_index = op_desc_ptr->GetId();

  AICPU_CHECK_RES_WITH_LOG(BuildAndLaunchKernel(node, run_context),
      "Call TfKernelBuilder::BuildAndLaunchKernel function failed, op[%s].",
      node.GetName().c_str())
  tasks.emplace_back(g_task_def);

  bool is_no_tiling = IsNotiling(op_desc_ptr);
  int32_t shape_type = 0;
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    CHECK_RES_BOOL(ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, shape_type),
        INVOKE_GRAPH_ITF_FAILED,
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
            ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(), node.GetName().c_str()))
    // unknow shape
    if (shape_type == ge::DEPEND_COMPUTE && !is_no_tiling) {
      // unknow type 4
      STR_FWK_OP_KERNEL task = {};
      std::string mem_copy_task_info;
      uint64_t data_info_size = static_cast<uint64_t>(op_desc_ptr->GetOutputsSize()) * kNum;
      GenMemCopyTask(data_info_size, task, mem_copy_task_info);

      domi::TaskDef task_def;
      task_def.set_type(RT_MODEL_TASK_KERNEL_EX);
      domi::KernelExDef *kernel_def_ex = task_def.mutable_kernel_ex();
      AICPU_CHECK_NOTNULL_ERRCODE(kernel_def_ex, ErrorCode::INPUT_PARAM_NULL);
      kernel_def_ex->set_args(reinterpret_cast<void *>(&task), sizeof(STR_FWK_OP_KERNEL));
      kernel_def_ex->set_args_size(sizeof(STR_FWK_OP_KERNEL));
      kernel_def_ex->set_task_info(mem_copy_task_info);
      kernel_def_ex->set_task_info_size(mem_copy_task_info.size());
      kernel_def_ex->set_op_index(g_op_index);
      tasks.emplace_back(task_def);
    }
  }
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::BuildFftsInfo(const ge::OpDescPtr &op_desc_ptr, FftsPlusInfo &ffts_info) const {
  ffts::ThreadSliceMapPtr slice_info = nullptr;
  slice_info = op_desc_ptr->TryGetExtAttr(kAttrNameSgtStruct, slice_info);
  if (slice_info == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("The Node[%s] has no _sgt_struct_info", op_desc_ptr->GetName().c_str());
    return INVOKE_GRAPH_ITF_FAILED;
  }
  ffts_info.valid = true;
  bool is_unknown_shape = false;
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    is_unknown_shape = true;
  }
  ffts_info.is_unknown_shape = is_unknown_shape;
  ffts_info.thread_mode = slice_info->thread_mode;
  ffts_info.auto_static_split = (!is_unknown_shape) && (ffts_info.thread_mode);
  ffts_info.thread_id = (ffts_info.thread_mode) ? 0u : slice_info->thread_id;
  ffts_info.slice_instance_num = (ffts_info.auto_static_split) ? slice_info->slice_instance_num : kAicpuManualSliceNum;
  if (ffts_info.auto_static_split) {
    auto thread_input_range = slice_info->input_tensor_slice;
    GetThreadTensorShape(thread_input_range, ffts_info.thread_input_shape);
    auto thread_output_range = slice_info->output_tensor_slice;
    GetThreadTensorShape(thread_output_range, ffts_info.thread_output_shape);
  } else {
    std::vector<std::vector<int64_t>> input_shape;
    std::vector<std::vector<int64_t>> output_shape;
    GetInOutPutsShape(op_desc_ptr, input_shape, output_shape);
    ffts_info.thread_input_shape.push_back(input_shape);
    ffts_info.thread_output_shape.push_back(output_shape);
  }
  AICPUE_LOGD("ffts_info [%s][%s]slicenum[%u], thread_mode[%u]", op_desc_ptr->GetName().c_str(),
              op_desc_ptr->GetType().c_str(), ffts_info.slice_instance_num, ffts_info.thread_mode);
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::GenerateFftsPlusTask(const ge::Node &node, const ge::RunContext &run_context) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    (void)ge::AttrUtils::SetStr(op_desc_ptr, "_ge_attr_lowering_func", kEngineNameAicputfFfts);
  }
  FftsPlusInfo ffts_info;
  ge::Status state = BuildFftsInfo(op_desc_ptr, ffts_info);
  if (state != ge::SUCCESS) {
    return state;
  }

  FftsPlusCtxDefPtr ffts_plus_ctx_def = nullptr;
  try {
      ffts_plus_ctx_def = std::make_shared<domi::FftsPlusCtxDef>();
  } catch (...) {
    AICPU_REPORT_INNER_ERR_MSG("op[%s] Create FftsPlusCtxDefPtr fail",
                             node.GetName().c_str());
    return MEMORY_ALLOC_FAILED;
  }
  ffts_plus_ctx_def->set_context_type(kCtxTypeAicpu);
  domi::FftsPlusAicpuCtxDef *aicpu_ctx = ffts_plus_ctx_def->mutable_aicpu_ctx();
  AICPU_CHECK_NOTNULL(aicpu_ctx);
  aicpu_ctx->set_topic_type(static_cast<uint32_t>(GetOpNodeTopicType(op_desc_ptr)));
  aicpu_ctx->set_bm(static_cast<uint32_t>(kDefaultNum));
  aicpu_ctx->set_group_id(static_cast<uint32_t>(kDefaultNum));
  aicpu_ctx->set_topic_id(static_cast<uint32_t>(kEventHwTsKernelMsg));
  aicpu_ctx->set_sub_topic_id(static_cast<uint32_t>(kEventFftsPlusMsg));
  aicpu_ctx->set_kernel_type(kKernelTypeFwk);
  aicpu_ctx->set_atm(static_cast<uint32_t>(ffts_info.thread_mode));
  if (ffts_info.is_unknown_shape) {
    aicpu_ctx->set_thread_dim(0);
  } else {
    aicpu_ctx->set_thread_dim(ffts_info.slice_instance_num);
  }
  aicpu_ctx->set_thread_id(ffts_info.thread_id);
  aicpu_ctx->set_non_tail_block_dim(kAicpuBlockDim);
  aicpu_ctx->set_tail_block_dim(kAicpuBlockDim);

  domi::aicpuKernelDef *aicpu_kernel_def = aicpu_ctx->mutable_kernel();
  aicpu_kernel_def->set_so_name(kTfKernelSo);
  aicpu_kernel_def->set_kernel_name(kTfFunctionName);
  STR_FWK_OP_KERNEL str_fwkop_kernel = {};
  std::string task_info;
  std::vector<char> task_ext_info;
  std::string temp_args;
  std::string ext_info_args;
  for (uint32_t index = 0; index < ffts_info.slice_instance_num; index++) {
    ffts_info.slice_instance_index = index;
    task_info.clear();
    task_ext_info.clear();
    AICPU_CHECK_RES_WITH_LOG(
        MakeTaskExtInfo(node, task_ext_info, ffts_info),
        "Call TfKernelBuilder::MakeTaskExtInfo function failed, op[%s].",
        node.GetName().c_str())
    ffts_info.ext_len = task_ext_info.size();
    AICPU_CHECK_RES_WITH_LOG(BuildAicpuFftsPlusKernel(node, str_fwkop_kernel, run_context, task_info, index, ffts_info),
                             "Call TfKernelBuilder::BuildAndLaunchKernel function failed, op[%s].",
                             node.GetName().c_str())

    ext_info_args.append(task_ext_info.data(), task_ext_info.size());
    if (index == 0) {
      uint32_t param_offset =
          static_cast<uint32_t>(task_info.length()) +
          static_cast<uint32_t>(sizeof(STR_FWK_OP_KERNEL));
      aicpu_ctx->set_task_param_offset(param_offset);
      FWKAdapter::FWKOperateParam *str_tf_kernel =
          &(str_fwkop_kernel.fwkKernelBase.fwk_kernel);
      str_tf_kernel->extInfoAddr = ffts_info.ext_len;
    }
    temp_args.append(reinterpret_cast<const char *>(&str_fwkop_kernel),
                     sizeof(STR_FWK_OP_KERNEL));
    temp_args.append(task_info);
  }
  aicpu_kernel_def->set_kernel_ext_info(ext_info_args.data(), ext_info_args.size());
  aicpu_kernel_def->set_kernel_ext_info_size(ext_info_args.size());
  aicpu_kernel_def->set_args(temp_args.c_str(), temp_args.length());
  aicpu_kernel_def->set_args_size(temp_args.length());

  ffts_info.slice_instance_index = 0;
  GetFftsPlusInOutAddrOffset(op_desc_ptr, ffts_info);
  AICPU_CHECK_FALSE_EXEC(
      ge::AttrUtils::SetListInt(op_desc_ptr, "tf_ffts_input_offset",
                                ffts_info.input_addr_offset),
      AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetInt failed to set "
                              "attr[%s],op[%s]",
                              ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(),
                              op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPU_CHECK_FALSE_EXEC(
      ge::AttrUtils::SetListInt(op_desc_ptr, "tf_ffts_output_offset",
                                ffts_info.output_addr_offset),
      AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetInt failed to set "
                              "attr[%s],op[%s]",
                              ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(),
                              op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  (void)op_desc_ptr->SetExtAttr(kAttrNameFftsPlusCtxDef, ffts_plus_ctx_def);
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::BuildAicpuFftsPlusKernel(const ge::Node &node,
                                                     STR_FWK_OP_KERNEL &str_fwkop_kernel,
                                                     const ge::RunContext &run_context,
                                                     std::string &task_info,
                                                     const uint32_t &index,
                                                     const FftsPlusInfo &ffts_info) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  // Type 0 represent tensorflow
  str_fwkop_kernel.fwkKernelType = FMK_KERNEL_TYPE_TF;
  FWKAdapter::FWKOperateParam *str_tf_kernel = &(str_fwkop_kernel.fwkKernelBase.fwk_kernel);
  str_tf_kernel->opType = FWKAdapter::FWK_ADPT_KERNEL_RUN;
  str_tf_kernel->sessionID = run_context.sessionId;
  str_tf_kernel->stepIDAddr = 0ul;
  str_tf_kernel->kernelID = 0ul;
  str_tf_kernel->extInfoLen = ffts_info.ext_len;
  str_tf_kernel->extInfoAddr = 0ul;

  // Step2 : Build the StrFWKKernel
  FWKAdapter::KernelRunParam kernel_run_param;
  AICPU_CHECK_RES(BuildFFtsKernelRunParam(*op_desc_ptr, kernel_run_param, index, ffts_info))
  str_tf_kernel->inputOutputLen = static_cast<uint64_t>(kernel_run_param.ByteSizeLong());
  str_tf_kernel->inputOutputBuf = 0ul;
  AICPUE_LOGI("The kernel_run_param_size size is [%lu], op type[%s]",
              str_tf_kernel->inputOutputLen, node.GetType().c_str());
  // Serialize the kernel_run_param
  ge::Buffer kernel_run_param_buffer(kernel_run_param.ByteSizeLong());
  auto state = SerializeKernelRunParamToBuffer(kernel_run_param, node.GetName(), kernel_run_param_buffer);
  if (state != ge::SUCCESS) {
    return state;
  }
  task_info.append(reinterpret_cast<const char *>(kernel_run_param_buffer.GetData()),
                   kernel_run_param_buffer.GetSize());
  // Step3~4 : build the tf's node_def and funcDef
  ge::GeAttrValue::BYTES node_def_bytes;
  ge::GeAttrValue::BYTES func_def_lib_bytes;
  int64_t node_def_size = 0l;
  int64_t func_def_lib_size = 0l;
  AICPU_CHECK_RES_WITH_LOG(
      ParseNodeDefAndFuncDef(node, node_def_bytes, func_def_lib_bytes,
                             node_def_size, func_def_lib_size),
      "Call TfKernelBuilder::ParseNodeDefAndFuncDef function failed, op[%s].",
      node.GetName().c_str())
  str_tf_kernel->funDefLibLen = 0ul; // initial value
  str_tf_kernel->nodeDefLen = static_cast<uint64_t>(node_def_size);
  str_tf_kernel->nodeDefBuf = str_tf_kernel->inputOutputBuf + str_tf_kernel->inputOutputLen;
  if (node_def_bytes.GetData() == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG(
        "Append node def to task_info failed data, node def is null, op[%s].",
        node.GetName().c_str());
    return INPUT_PARAM_NULL;
  }
  task_info.append(reinterpret_cast<const char *>(node_def_bytes.GetData()), node_def_size);

  // Serialize the funcDef
  if (func_def_lib_size > 0 && func_def_lib_bytes.GetData() != nullptr) {
    str_tf_kernel->funDefLibLen = static_cast<uint64_t>(func_def_lib_size);
    str_tf_kernel->funDefLibBuf = str_tf_kernel->nodeDefBuf + str_tf_kernel->nodeDefLen;
    const char *func_def_lib_data = reinterpret_cast<const char *>(func_def_lib_bytes.GetData());
    if (func_def_lib_data == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("Append function def to task_info failed data,"
          " function def is null, op[%s].", node.GetName().c_str());
      return INPUT_PARAM_NULL;
    }
    task_info.append(func_def_lib_data, func_def_lib_size);
  }

  // Init inputOutputAddr and workspaceBaseAddr, GE will refresh this value
  str_tf_kernel->inputOutputAddr = 0ul;
  str_tf_kernel->workspaceBaseAddr = static_cast<uint64_t>(sizeof(STR_FWK_OP_KERNEL));

  // Update the FmkOp info
  AICPU_CHECK_RES(UpdateFmkOpInfo(op_desc_ptr))

  if (CheckUint64AddOverflow(str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen)) {
    AICPU_REPORT_INNER_ERR_MSG("Overflow occurred when calculate total bytes of input/output info[%lu] and"
                           "node def[%lu]. Calculate workspace total bytes failed, op[%s]",
                            str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen,
                            node.GetName().c_str());
    return ErrorCode::DATA_OVERFLOW;
  }
  if (CheckUint64AddOverflow(str_tf_kernel->inputOutputLen + str_tf_kernel->nodeDefLen,
                             str_tf_kernel->funDefLibLen)) {
    AICPU_REPORT_INNER_ERR_MSG("Overflow occurred when calculate total bytes of input/output info[%lu], node"
                            "def[%lu] and function def[%lu]. Calculate workspace total bytes failed, op[%s]",
                            str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen,
                            str_tf_kernel->funDefLibLen, node.GetName().c_str());
    return ErrorCode::DATA_OVERFLOW;
  }

  if (!ffts_info.is_unknown_shape) {
    uint64_t workspace_bytes_size = 0ul;
    AICPU_CHECK_RES_WITH_LOG(
        GetWorkspaceInfo(op_desc_ptr, run_context.dataMemBase,
                         workspace_bytes_size),
        "Call KernelBuilder::GetWorkspaceInfo function failed, op[%s].",
        node.GetName().c_str())

    uint64_t min_memory = str_tf_kernel->inputOutputLen +
                          str_tf_kernel->nodeDefLen + str_tf_kernel->funDefLibLen;
    if (workspace_bytes_size < min_memory) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Workspace memory not enough, given[%lu] bytes, expected[%lu] bytes, op[%s]",
          workspace_bytes_size, min_memory, node.GetType().c_str());
      return GE_MEM_NOT_ENOUGH;
    }
  }
  AICPUE_LOGI("BuildAicpuFftsPlusKernel success, kernel id[%lu], op[%s], op type[%s]",
              str_tf_kernel->kernelID, node.GetName().c_str(), node.GetType().c_str());
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::ConstructTfKernel(const ge::Node &node, const ge::RunContext &run_context,
  FWKAdapter::FWKOperateParam *str_tf_kernel) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)

  str_tf_kernel->opType = FWKAdapter::FWK_ADPT_KERNEL_RUN;
  str_tf_kernel->sessionID = run_context.sessionId;
  str_tf_kernel->stepIDAddr = 0;
  str_tf_kernel->kernelID = GenerateUniqueId();
  str_tf_kernel->extInfoLen = 0;
  str_tf_kernel->extInfoAddr = 0;
  AICPUE_LOGI("Op type[%s] The kernel id is [%lu], session id is [%lu].",
              node.GetType().c_str(), str_tf_kernel->kernelID, str_tf_kernel->sessionID);

  // Step2 : Build the StrFWKKernel
  FWKAdapter::KernelRunParam kernel_run_param;
  AICPU_CHECK_RES(BuildKernelRunParam(*op_desc_ptr, kernel_run_param))
  // Serialize the kernel_run_param
  str_tf_kernel->inputOutputLen = static_cast<int64_t>(kernel_run_param.ByteSizeLong());
  str_tf_kernel->inputOutputBuf = 0;
  AICPUE_LOGI("The param_size size [%lu], op type[%s]", str_tf_kernel->inputOutputLen, node.GetType().c_str());
  ge::Buffer kernel_run_param_buffer(kernel_run_param.ByteSizeLong());
  auto state = SerializeKernelRunParamToBuffer(kernel_run_param, node.GetName(), kernel_run_param_buffer);
  if (state != ge::SUCCESS) {
    return state;
  }
  g_task_info.append(reinterpret_cast<const char *>(kernel_run_param_buffer.GetData()), kernel_run_param_buffer.GetSize());

  // Step3~4 : build the tf's node_def and funcDef
  ge::Buffer node_def_bytes;
  ge::Buffer func_def_lib_bytes;
  int64_t node_def_size = 0;
  int64_t func_def_lib_size = 0;
  AICPU_CHECK_RES_WITH_LOG(
      ParseNodeDefAndFuncDef(node, node_def_bytes, func_def_lib_bytes, node_def_size, func_def_lib_size),
      "Call TfKernelBuilder::ParseNodeDefAndFuncDef function failed, op[%s].", node.GetName().c_str())
  str_tf_kernel->funDefLibLen = 0; // initial value
  str_tf_kernel->nodeDefLen = node_def_size;
  str_tf_kernel->nodeDefBuf = str_tf_kernel->inputOutputBuf + str_tf_kernel->inputOutputLen;
  if (node_def_bytes.GetData() == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Append node def to g_task_info failed op[%s].", node.GetName().c_str());
    return INPUT_PARAM_NULL;
  }
  g_task_info.append(reinterpret_cast<const char *>(node_def_bytes.GetData()), node_def_size);

  // Serialize the funcDef
  if (func_def_lib_size > 0 && func_def_lib_bytes.GetData() != nullptr) {
    str_tf_kernel->funDefLibLen = func_def_lib_size;
    str_tf_kernel->funDefLibBuf = str_tf_kernel->nodeDefBuf + str_tf_kernel->nodeDefLen;
    const char *func_def_lib_data = reinterpret_cast<const char *>(func_def_lib_bytes.GetData());
    if (func_def_lib_data == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("Append function def to g_task_info failed op[%s],", node.GetName().c_str());
      return INPUT_PARAM_NULL;
    }
    g_task_info.append(func_def_lib_data, static_cast<size_t>(func_def_lib_size));
  }

  // Init inputOutputAddr and workspaceBaseAddr, GE will refresh this value
  str_tf_kernel->inputOutputAddr = 0;
  str_tf_kernel->workspaceBaseAddr = 0;

  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::ConstructExtendInfoAndTaskdef(const ge::Node &node, const ge::RunContext &run_context,
  FWKAdapter::FWKOperateParam *str_tf_kernel) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)

  bool is_unknown_shape = false;
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    // kAttrNameUnknownShape attr exist, means unknow shape
    is_unknown_shape = true;
  }
  if (!is_unknown_shape) {
    uint64_t workspace_bytes_size = 0;
    AICPU_CHECK_RES_WITH_LOG(GetWorkspaceInfo(op_desc_ptr, run_context.dataMemBase, workspace_bytes_size),
                             "Call KernelBuilder::GetWorkspaceInfo function failed, op[%s].", node.GetName().c_str())
    uint64_t min_size = str_tf_kernel->inputOutputLen + str_tf_kernel->nodeDefLen + str_tf_kernel->funDefLibLen;
    if (workspace_bytes_size < min_size) {
      AICPU_REPORT_INNER_ERR_MSG("Workspace memory not enough, given[%lu] bytes, expected[%lu] bytes, op[%s]",
          workspace_bytes_size, min_size, node.GetType().c_str());
      return GE_MEM_NOT_ENOUGH;
    }
  }

  // make and set extend info
  std::vector<char> task_ext_info;
  domi::KernelExDef *kernel_def_ex = g_task_def.mutable_kernel_ex();
  AICPU_CHECK_NOTNULL(kernel_def_ex);
  FftsPlusInfo ffts_info;
  AICPU_CHECK_RES_WITH_LOG(MakeTaskExtInfo(node, task_ext_info, ffts_info),
                           "Call TfKernelBuilder::MakeTaskExtInfo function failed, op[%s].", node.GetName().c_str())
  if (task_ext_info.size() == 0) {
    str_tf_kernel->extInfoLen = 0;
    kernel_def_ex->clear_kernel_ext_info();
    kernel_def_ex->set_kernel_ext_info_size(0);
  } else {
    str_tf_kernel->extInfoLen = task_ext_info.size();
    kernel_def_ex->set_kernel_ext_info(reinterpret_cast<void *>(task_ext_info.data()), task_ext_info.size());
    kernel_def_ex->set_kernel_ext_info_size(task_ext_info.size());
  }
  AICPUE_LOGI("Node info: unknown shape is [%d], extend info length[%lu], op[%s], op type[%s]", is_unknown_shape,
              str_tf_kernel->extInfoLen, op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());

  g_task_def.set_stream_id(op_desc_ptr->GetStreamId());
  g_task_def.set_type(RT_MODEL_TASK_KERNEL_EX);

  g_task_def.set_sqe_num(kBasicTfOpSqeNumber);
  bool is_blocking_aicpu_op = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op);
  if (is_blocking_aicpu_op) {
    g_task_def.set_sqe_num(kAsyncTfOpSqeNumber);
  }

  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::BuildAndLaunchKernel(const ge::Node &node, const ge::RunContext &run_context) const {
  // Step1 : define the task api struct.
  AICPU_MAKE_SHARED(g_str_fwkop_kernel_ptr = std::make_shared<STR_FWK_OP_KERNEL>(),
      AICPU_REPORT_INNER_ERR_MSG("Create STR_FWK_OP_KERNEL object failed, op[%s]", node.GetName().c_str());
      return ErrorCode::MEMORY_ALLOC_FAILED)

  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)

  // Type 0 represent tensorflow
  g_str_fwkop_kernel_ptr->fwkKernelType = FMK_KERNEL_TYPE_TF;
  FWKAdapter::FWKOperateParam *str_tf_kernel = &(g_str_fwkop_kernel_ptr->fwkKernelBase.fwk_kernel);

  AICPU_CHECK_RES_WITH_LOG(ConstructTfKernel(node, run_context, str_tf_kernel),
                           "construct tf kernel fail, op[%s].", node.GetName().c_str());
  // Update the FmkOp info
  AICPU_CHECK_RES(UpdateImplyTypeInfo(op_desc_ptr))
  CHECK_UINT64_ADD_OVERFLOW(str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen, ErrorCode::DATA_OVERFLOW,
      "Overflow occurred when calculate total bytes of input/output info[%lu] and node def[%lu] op[%s].",
      str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen, node.GetName().c_str())
  CHECK_UINT64_ADD_OVERFLOW(str_tf_kernel->inputOutputLen + str_tf_kernel->nodeDefLen,
      str_tf_kernel->funDefLibLen, ErrorCode::DATA_OVERFLOW,
      "Overflow occurred when calculate total bytes of input/output [%lu] node def[%lu] function def[%lu] op[%s].",
      str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen, str_tf_kernel->funDefLibLen, node.GetName().c_str())

  AICPU_CHECK_RES_WITH_LOG(ConstructExtendInfoAndTaskdef(node, run_context, str_tf_kernel),
                           "construct tf task def fail, op[%s].", node.GetName().c_str());

  domi::KernelExDef *kernel_def_ex = g_task_def.mutable_kernel_ex();
  AICPU_CHECK_NOTNULL(kernel_def_ex);
  kernel_def_ex->set_args(reinterpret_cast<void *>(g_str_fwkop_kernel_ptr.get()), sizeof(STR_FWK_OP_KERNEL));
  kernel_def_ex->set_args_size(sizeof(STR_FWK_OP_KERNEL));
  kernel_def_ex->set_task_info(g_task_info);
  kernel_def_ex->set_task_info_size(g_task_info.size());
  kernel_def_ex->set_op_index(g_op_index);
  kernel_def_ex->set_flags(0);
  g_task_info.clear();

  AICPUE_LOGI("BuildAndLaunchKernel success, kernel id[%lu], op[%s], op type[%s].",
              str_tf_kernel->kernelID, node.GetName().c_str(), node.GetType().c_str());
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::UpdateFmkOpInfo(std::shared_ptr<ge::OpDesc> &op_desc_ptr) const {
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  std::string original_type = op_desc_ptr->GetType();
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetStr(op_desc_ptr, kOriginalType, original_type),
      AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetStr failed to set attr[%s], op[%s].",
          kOriginalType.c_str(), op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  ge::OpDescUtilsEx::SetType(op_desc_ptr, kFrameworkOp);

  // value 3 represent the framework tensorflow
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc_ptr, kFrameworkType, 3),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetInt failed to set attr[%s], op[%s].",
          kFrameworkType.c_str(), op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE,
                                               static_cast<int64_t>(domi::ImplyType::AI_CPU)),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetInt failed to set attr[%s], op[%s].",
          ge::ATTR_NAME_IMPLY_TYPE.c_str(), op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::UpdateImplyTypeInfo(std::shared_ptr<ge::OpDesc> &op_desc_ptr) const {
  AICPU_CHECK_FALSE_EXEC(
      ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU)),
      AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetInt failed to set attr[%s], op[%s].",
                              ge::ATTR_NAME_IMPLY_TYPE.c_str(), op_desc_ptr->GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  return ge::SUCCESS;
}

// Make task extend info for node
ge::Status TfKernelBuilder::MakeTaskExtInfo(const ge::Node &node,
                                            std::vector<char> &task_ext_info,
                                            const FftsPlusInfo &ffts_info) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)
  // op name extend info
  // WARNING: OP NAME MUST BE THE FIRST EXTEND INFO FOR RUNTIME!!!
  ge::Status status = MakeExtInfoForOpName(op_desc_ptr, task_ext_info);
  if (status != ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call MakeExtInfoForOpName failed, op[%s].",
        op_desc_ptr->GetName().c_str());
    return status;
  }

  int32_t unknow_shape_type = 0;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknow_shape_type)) {
    AICPUE_LOG_RUN_INFO("ge::AttrUtils::not get attr,set attr[%s],op[%s].",
        ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(), op_desc_ptr->GetName().c_str());

    KernelInfoPtr kernel_info_ptr = TfKernelInfo::Instance();
    AICPU_CHECK_NOTNULL(kernel_info_ptr);
    OpFullInfo op_full_info;
    const string *op_type = ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_CHECK_NOTNULL(op_type);
    status = kernel_info_ptr->GetOpInfo(*op_type, op_full_info);
    AICPU_CHECK_FALSE_EXEC(status == ge::SUCCESS,
        AICPU_REPORT_INNER_ERR_MSG("op type[%s] not support,op[%s].",
            op_type->c_str(), op_desc_ptr->GetName().c_str());
        return status;)
    int32_t shape_type = 0;
    if (!IsUnknowShape(op_desc_ptr) && (op_full_info.shapeType == OP_TYPE_FOUR)) {
      shape_type = 1;
    } else {
      shape_type = op_full_info.shapeType;
    }
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, shape_type),
        AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetInt failed to set attr[%s],op[%s],op type[%s].",
          ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(), op_desc_ptr->GetName().c_str(),
          op_type->c_str());
        return ErrorCode::ADD_ATTR_FAILED)
  }

  // common base extend info
  status = MakeBaseExtInfo(op_desc_ptr, task_ext_info, ffts_info);
  if (status != ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call MakeBaseExtInfo failed, op[%s], op type[%s].",
        op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return status;
  }

  // make no tiling extend info
  AICPU_CHECK_RES_WITH_LOG(MakeNoTilingExtInfo(op_desc_ptr, task_ext_info),
                           "Call MakeNoTilingExtInfo funtion failed, op[%s].",
                           op_desc_ptr->GetName().c_str())

  int32_t update_addr_flag = 0;
  if (NeedUpdateAddr(node, update_addr_flag)) {
    AICPUE_LOGI("Set update inputs or outputs address extend info, update_addr_flag[%d], op[%s], op type[%s].",
                update_addr_flag, node.GetName().c_str(), op_desc_ptr->GetType().c_str());
    // add update inputs/outputs address extend info
    uint64_t cur_ext_info_len = task_ext_info.size();
    // value length: sizeof(int32_t)
    uint64_t update_addr_ext_info_len = FWKAdapter::kExtInfoHeadSize + sizeof(int32_t);
    task_ext_info.resize(cur_ext_info_len + update_addr_ext_info_len, 0);
    char *ext_info_buf = task_ext_info.data() + cur_ext_info_len;
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(ext_info_buf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_UPDATE_ADDR;
    extInfo->infoLen = sizeof(int32_t);
    // set value
    ext_info_buf += FWKAdapter::kExtInfoHeadSize;
    *reinterpret_cast<int32_t *>(ext_info_buf) = update_addr_flag;
  }
  return ge::SUCCESS;
}

// Check the node whether need update inputs/outputs address
bool TfKernelBuilder::NeedUpdateAddr(const ge::Node &node, int32_t &update_addr_flag) const {
  if (IsKnownNodeDynamic(node)) {
    // known node in dynamic shape graph, need update inputs/outputs address
    update_addr_flag = FWKAdapter::FWK_ADPT_UPDATE_INPUT_OUTPUT;
    AICPUE_LOGI("The known shape node in dynamic shape graph, op[%s], op type[%s].",
                node.GetName().c_str(), node.GetType().c_str());
    return true;
  }

  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, false)
  // mini inference(zero copy): check the node whether is first or last node
  if (ge::AttrUtils::HasAttr(op_desc_ptr, ge::ATTR_NAME_NODE_CONNECT_INPUT) &&
      ge::AttrUtils::HasAttr(op_desc_ptr, ge::ATTR_NAME_NODE_CONNECT_OUTPUT)) {
    // both first and last node, update inputs/outputs address
    update_addr_flag = FWKAdapter::FWK_ADPT_UPDATE_INPUT_OUTPUT;
    return true;
  }
  if (ge::AttrUtils::HasAttr(op_desc_ptr, ge::ATTR_NAME_NODE_CONNECT_INPUT)) {
    // first node, update inputs address
    update_addr_flag = FWKAdapter::FWK_ADPT_UPDATE_INPUT;
    return true;
  } else if (ge::AttrUtils::HasAttr(op_desc_ptr, ge::ATTR_NAME_NODE_CONNECT_OUTPUT)) {
    // last node, update outputs address
    update_addr_flag = FWKAdapter::FWK_ADPT_UPDATE_OUTPUT;
    return true;
  } else {
    return false;
  }
}

bool TfKernelBuilder::IsKnownNodeDynamic(const ge::Node &node) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, false)
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    // unknown shape node
    return false;
  }

  auto owner_graph = node.GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    AICPUE_LOGW("Get null owner compute graph, op[%s], op type[%s].",
                node.GetName().c_str(), node.GetType().c_str());
    return false;
  }
  auto rootGraph = ge::GraphUtils::FindRootGraph(owner_graph);
  if (rootGraph == nullptr) {
    AICPUE_LOGW("Get null root graph, op[%s], op type[%s].",
                node.GetName().c_str(), node.GetType().c_str());
    return false;
  }
  bool is_dynamic = false;
  (void)ge::AttrUtils::GetBool(rootGraph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic);
  return is_dynamic;
}

ge::Status TfKernelBuilder::GenTaskImply(const ge::NodePtr &node,
                                         FWKAdapter::FWKOperateParam *str_tf_kernel,
                                         std::string &task_info,
                                         bool skip_dim_check) const {
  // Set default value, these fields may not be used
  str_tf_kernel->sessionID = 0;
  str_tf_kernel->kernelID = 0;

  // for single op run
  str_tf_kernel->opType = FWKAdapter::FWK_ADPT_SINGLE_OP_RUN;

  // Build the KernelRunParam
  FWKAdapter::KernelRunParam kernel_run_param;
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr);
  AICPU_CHECK_RES_WITH_LOG(BuildKernelRunParam(*op_desc_ptr, kernel_run_param, skip_dim_check),
                           "Call TfKernelBuilder::BuildKernelRunParam function failed, op[%s]", node->GetName().c_str())
  str_tf_kernel->inputOutputLen = static_cast<int64_t>(kernel_run_param.ByteSizeLong());
  str_tf_kernel->inputOutputBuf = 0;
  AICPUE_LOGI("The kernel_run_param_size size is [%lu], op type[%s].",
              str_tf_kernel->inputOutputLen, node->GetType().c_str());
  // Serialize the kernel_run_param
  ge::Buffer kernel_run_param_buffer(kernel_run_param.ByteSizeLong());
  auto state = SerializeKernelRunParamToBuffer(kernel_run_param, node->GetName(), kernel_run_param_buffer);
  if (state != ge::SUCCESS) {
    return state;
  }
  task_info.append(reinterpret_cast<const char *>(kernel_run_param_buffer.GetData()),
                   kernel_run_param_buffer.GetSize());
  // Build the tf's nodeDef and funcDef
  ge::Buffer node_def_bytes;
  ge::Buffer func_def_lib_bytes;
  int64_t node_def_size = 0;
  int64_t func_def_lib_size = 0;
  // Serialize the nodeDef
  AICPU_CHECK_RES_WITH_LOG(
      ParseNodeDefAndFuncDef(*node, node_def_bytes, func_def_lib_bytes, node_def_size, func_def_lib_size),
      "Call TfKernelBuilder::ParseNodeDefAndFuncDef function failed, op[%s].", node->GetName().c_str())
  str_tf_kernel->funDefLibLen = 0; // initial value
  str_tf_kernel->nodeDefLen = static_cast<uint64_t>(node_def_size);
  str_tf_kernel->nodeDefBuf = str_tf_kernel->inputOutputBuf + str_tf_kernel->inputOutputLen;
  if (node_def_bytes.GetData() == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Append node def to task_info failed, node def is null, op[%s].", node->GetName().c_str());
    return INPUT_PARAM_NULL;
  }

  task_info.append(reinterpret_cast<const char *>(node_def_bytes.GetData()), node_def_size);
  // Serialize the funcDef
  if (func_def_lib_size > 0 && func_def_lib_bytes.GetData() != nullptr) {
    str_tf_kernel->funDefLibLen = func_def_lib_size;
    str_tf_kernel->funDefLibBuf = str_tf_kernel->nodeDefBuf + str_tf_kernel->nodeDefLen;
    const char *func_def_lib_data = reinterpret_cast<const char *>(func_def_lib_bytes.GetData());
    if (func_def_lib_data == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Append function def to task_info failed, function def if null, op[%s]",
          node->GetType().c_str());
      return INPUT_PARAM_NULL;
    }
    task_info.append(func_def_lib_data, func_def_lib_size);
  }

  // Update the FmkOp info
  AICPU_CHECK_RES(UpdateFmkOpInfo(op_desc_ptr))
  CHECK_UINT64_ADD_OVERFLOW(str_tf_kernel->inputOutputLen,
      str_tf_kernel->nodeDefLen,
      ErrorCode::DATA_OVERFLOW,
      "Overflow occurred when calculate total bytes of input/output info[%lu] and"
      " node def[%lu]. Calculate workspace total bytes failed, op[%s]",
      str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen,
      node->GetName().c_str())
  CHECK_UINT64_ADD_OVERFLOW(str_tf_kernel->inputOutputLen + str_tf_kernel->nodeDefLen,
      str_tf_kernel->funDefLibLen,
      ErrorCode::DATA_OVERFLOW,
      "Overflow occurred when calculate total bytes of input/output info[%lu], node "
      "def[%lu] and function def[%lu]. Calculate workspace total bytes failed, op[%s]",
      str_tf_kernel->inputOutputLen, str_tf_kernel->nodeDefLen,
      str_tf_kernel->funDefLibLen, node->GetName().c_str())

  // disable ext info
  str_tf_kernel->extInfoLen = 0;
  str_tf_kernel->extInfoAddr = 0;
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::GenSingleOpRunTask(const ge::NodePtr &node, STR_FWK_OP_KERNEL &task,
                                               std::string &task_info) {
  AICPU_CHECK_NOTNULL(node);
  AICPUE_LOGI("Op[%s], op type[%s] start GenSingleOpRunTask.", node->GetName().c_str(), node->GetType().c_str());
  task.fwkKernelType = FMK_KERNEL_TYPE_TF;
  FWKAdapter::FWKOperateParam *str_tf_kernel = &(task.fwkKernelBase.fwk_kernel);
  // Build the str_tf_kernel
  AICPU_CHECK_RES_WITH_LOG(GenTaskImply(node, str_tf_kernel, task_info, true),
      "Call TfKernelBuilder::GenTaskImply function failed, op[%s].",
      node->GetName().c_str())
  return ge::SUCCESS;
}

ge::Status TfKernelBuilder::GenMemCopyTask(uint64_t count,
                                           STR_FWK_OP_KERNEL &task,
                                           std::string &task_info) {
  task.fwkKernelType = FMK_KERNEL_TYPE_TF;
  FWKAdapter::FWKOperateParam *str_tf_kernel = &(task.fwkKernelBase.fwk_kernel);
  // Build Ge node
  static int copy_count = 0;
  std::string node_type("MemCopy");
  // memCopy has four inputs and zero outputs
  int in_count = 4;
  int out_count = 0;
  ge::Format format = ge::FORMAT_NCHW;
  // DT_UINT64 is the data type of the element in the struct DataPtrInfo
  ge::DataType data_type = ge::DT_UINT64;
  std::vector<int64_t> shape = {};
  shape.push_back(count);
  std::string node_name(node_type + "_" + std::to_string(copy_count));
  ge::NodePtr node = aicpu::GenGeNode(node_name, node_type, in_count, out_count, format, data_type, shape);
  AICPU_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc);
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc, "num", count),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetInt failed to set attr[num], op[%s].",
          node_name.c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPUE_LOGI("Op[%s], op type[%s] start GenMemCopyTask", node->GetName().c_str(), node->GetType().c_str());
  // Build the str_tf_kernel
  AICPU_CHECK_RES_WITH_LOG(GenTaskImply(node, str_tf_kernel, task_info),
                           "Call TfKernelBuilder::GenTaskImply function failed, op[%s].", node->GetName().c_str())
  return ge::SUCCESS;
}

void TfKernelBuilder::GetInOutPutsDataType(const ge::OpDescPtr &op_desc_ptr,
                                           std::vector<uint32_t> &inputs_type,
                                           std::vector<uint32_t> &outputs_type) const {
  std::set<std::string>  refinput_set;
  std::string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    // using frameworkop type can not get the is_ref flag, so use original one.
    (void)ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType, op_type);
  }
  auto parser = Ir2tfParserFactory::Instance().CreateIRParser(op_type);
  parser->GetRefInputSet(op_type, refinput_set);
  size_t input_size = op_desc_ptr->GetInputsSize();
  for (size_t index = 0; index < input_size; index++) {
    ge::GeTensorDesc tensor_desc = op_desc_ptr->GetInputDesc(index);
    std::string input_name = op_desc_ptr->GetInputNameByIndex(index);
    bool is_ref = false;
    auto iter = refinput_set.find(input_name);
    if (iter != refinput_set.end()) {
      is_ref = true;
    }
    ge::DataType data_type = tensor_desc.GetDataType();
    inputs_type.push_back(static_cast<uint32_t>(ConvertGeDataType2TfDataType(data_type, is_ref)));
  }
  size_t outSize = op_desc_ptr->GetOutputsSize();
  for (size_t index = 0; index < outSize; index++) {
    ge::GeTensorDesc tensor_desc = op_desc_ptr->GetOutputDesc(index);
    ge::DataType data_type = tensor_desc.GetDataType();
    outputs_type.push_back(static_cast<uint32_t>(ConvertGeDataType2TfDataType(data_type, false)));
  }
}
FACTORY_KERNEL_BUILDER_CLASS_KEY(TfKernelBuilder, "TFBuilder")
} // namespace aicpu
