/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "args_format_constructor.h"
#include "framework/common/types.h"
#include "common/op_tensor_utils.h"
#include "common/platform_utils.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_constants.h"
#include "common/fe_op_info_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
void ArgsFormatConstructor::AddDynamicDesc(const std::pair<size_t, size_t> &range, size_t ir_index, bool is_input) {
  dyn_io_v_.clear();
  for (size_t dy_idx = 0; dy_idx < range.second; ++dy_idx) {
    if (!is_dy_folded_) {
      if (is_input) {
        format_desc_.Append(ge::AddrType::INPUT, ir_index);
      } else {
        format_desc_.Append(ge::AddrType::OUTPUT, ir_index);
      }
      return;
    }
    if (dy_idx == 0) {
      if (is_input) {
        format_desc_.Append(ge::AddrType::INPUT_DESC, ir_index, true);
      } else {
        format_desc_.Append(ge::AddrType::OUTPUT_DESC, ir_index, true);
      }
    }
    dyn_io_v_.emplace_back(static_cast<int64_t>(range.first + dy_idx));
  }
  dyn_io_vv_.emplace_back(dyn_io_v_);
}
bool GetInIrIndexByName(const std::vector<std::pair<std::string, ge::IrInputType>> &ir_inputs, const std::string &name,
                   size_t &ir_index) {
  size_t ir_size = ir_inputs.size();
  for (size_t i = 0UL; i < ir_size; ++i) {
    if (ir_inputs[i].first == name) {
      ir_index = i;
      return true;
    }
  }
  return false;
}

bool GetOutIrIndexByName(const std::vector<std::pair<std::string, ge::IrOutputType>> &ir_outputs,
                         const std::string &name, size_t &ir_index) {
  size_t ir_size = ir_outputs.size();
  for (size_t i = 0UL; i < ir_size; ++i) {
    if (ir_outputs[i].first == name) {
      ir_index = i;
      return true;
    }
  }
  return false;
}

bool FindInputInGraph(size_t idx, const std::vector<InputOrOutputInfoPtr>& input_infos,
                      std::vector<std::string> &input_name_list) {
  auto &input_name = input_infos[idx]->GetName();
  auto paramType = input_infos[idx]->GetParamType();
  if (paramType == REQUIRED || paramType == DYNAMIC) {
    return true;
  }
  auto input_size = input_name_list.size();
  for (size_t j = 0; j < input_size; ++j) {
    FE_LOGD("Input name [%s].", input_name_list[j].c_str());
    if (input_name == input_name_list[j]) {
      return true;
    }
  }
  FE_LOGD("Input name [%s] not find, with input size %zu.", input_name.c_str(), input_size);
  return false;
}

bool ArgsFormatConstructor::FindOptInsertPos(size_t ir_idx, const std::vector<InputOrOutputInfoPtr>& input_infos,
    std::vector<std::string> &input_name_list, size_t &insert_pos) const {
  insert_pos = 0;
  if (ir_idx == 0) {
    return true;
  }
  auto &pre_input_name = input_infos[ir_idx - 1]->GetName();
  for (size_t j = input_name_list.size(); j > 0; --j) {
    if (pre_input_name == input_name_list[j - 1]) {
      insert_pos = j;
      break;
    }
  }
  if (insert_pos == 0) {
    REPORT_FE_ERROR("[ArgsFormatConstructor] Op[name=%s,type=%s]Not find pre input name [%s].",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), pre_input_name.c_str());
    return false;
  }
  return true;
}

bool ArgsFormatConstructor::InsertMissOptInput(std::vector<uint32_t> &input_type_list,
    std::vector<int32_t> &input_graph_idx, std::vector<std::string> &input_name_list, size_t exp_num) const {
  int64_t imply_type = -1;
  (void)ge::AttrUtils::GetInt(op_desc_, FE_IMPLY_TYPE, imply_type);
  OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(op_desc_->GetOpEngineName())
      .GetOpKernelInfoByOpType(static_cast<OpImplType>(imply_type), op_desc_->GetType());
  if (op_kernel_info_ptr == nullptr) {
    REPORT_FE_ERROR("[ArgsFormatConstructor] Op[name=%s,type=%s] Failed to get kernel info.",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return false;
  }
  const auto &input_infos = op_kernel_info_ptr->GetAllInputInfo();
  std::vector<uint32_t> insert_pos_vec;
  for (size_t i = 0; i < input_infos.size(); ++i) {
    if (FindInputInGraph(i, input_infos, input_name_list)) {
      continue;
    }
    size_t insert_pos = 0;
    if (!FindOptInsertPos(i, input_infos, input_name_list, insert_pos)) {
      return false;
    }
    auto &input_name = input_infos[i]->GetName();
    FE_LOGD("Insert miss optional input [%s] at pos[%zu].", input_name.c_str(), insert_pos);
    insert_pos_vec.emplace_back(insert_pos);
    input_type_list.insert(input_type_list.begin() + insert_pos, static_cast<uint32_t>(OPTIONAL));
    input_name_list.insert(input_name_list.begin() + insert_pos, input_name);
    input_graph_idx.insert(input_graph_idx.begin() + insert_pos, -1);
  }
  for (size_t i = 0; i < input_name_list.size(); ++i) {
    FE_LOGD("After reconstructing the input name [%s].", input_name_list[i].c_str());
  }
  if (input_type_list.size() != exp_num) {
    REPORT_FE_ERROR("[ArgsFormatConstructor] Op[name=%s,type=%s]In size[%zu] not equal[%zu].",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), input_type_list.size(), exp_num);
    return false;
  }
  (void)ge::AttrUtils::SetListInt(op_desc_, kInputParaTypeList, input_type_list);
  (void)ge::AttrUtils::SetListStr(op_desc_, kInputNameList, input_name_list);
  (void)ge::AttrUtils::SetListInt(op_desc_, kInputInsertOptPosList, insert_pos_vec);
  return true;
}

bool ArgsFormatConstructor::GetOpInputInfo(std::vector<uint32_t> &input_type_list,
    std::vector<int32_t> &input_graph_idx, std::vector<std::string> &input_name_list,
    std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  (void)ge::AttrUtils::GetListInt(op_desc_, kInputParaTypeList, input_type_list);
  (void)ge::AttrUtils::GetListStr(op_desc_, kInputNameList, input_name_list);
  // (ir index, (input index on graph, range))
  if(ge::OpDescUtils::GetIrInputInstanceDescRange(op_desc_, ir_input_2_range) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Get ir input range failed.");
    return false;
  }
  size_t input_size = input_name_list.size();
  if (input_type_list.size() != input_size) {
    FE_LOGW("Input name size[%zu] not equal with type size[%zu].", input_size, input_type_list.size());
    return false;
  }
  for (size_t i = 0; i < input_size; ++i) {
    input_graph_idx.emplace_back(i);
  }
  size_t all_num = 0;
  size_t exp_num = 0;
  for (const auto &range : ir_input_2_range) {
    all_num += range.second.second;
    if (range.second.second > 1) {
      exp_num += (range.second.second - 1);
    }
  }
  if (all_num > input_size) {
    FE_LOGW("Input name size[%zu] less size by ir[%zu].", input_size, all_num);
    return false;
  }
  FE_LOGD("Op[%s] dynamic input expand num is %zu.", op_desc_->GetNamePtr(), exp_num);
  (void)ge::AttrUtils::SetInt(op_desc_, kDyInputsAddNum, exp_num);

  size_t ops_in_size = 0;
  (void)ge::AttrUtils::GetInt(op_desc_, kOpKernelAllInputSize, ops_in_size);
  exp_num += ops_in_size;
  FE_LOGD("Op expect input num[%zu] with real[%zu].", exp_num, input_size);
  if (!is_input_gen_place_ || (exp_num <= input_size)) {
    return true;
  }
  return InsertMissOptInput(input_type_list, input_graph_idx, input_name_list, exp_num);
}

Status ArgsFormatConstructor::ConstructInArgsDescByOps(const std::vector<std::pair<std::string, ge::IrInputType>>
                                                      &ir_inputs) {
  std::vector<uint32_t> input_type_list;
  std::vector<int32_t> input_graph_idx;
  std::vector<std::string> input_name_list;
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  if (!GetOpInputInfo(input_type_list, input_graph_idx, input_name_list, ir_input_2_range)) {
    return FAILED;
  }
  size_t in_num = input_type_list.size();
  size_t dy_ir_idx = 0xFFFF;
  size_t ir_index = 0;
  for (size_t in_idx = 0; in_idx < in_num; ++in_idx) {
    auto input_type = input_type_list[in_idx];
    auto &input_name = input_name_list[in_idx];
    if (!GetInIrIndexByName(ir_inputs, input_name, ir_index)) {
      FE_LOGW("Op[%s] Input name[%s] not found in ir.", op_desc_->GetNamePtr(), input_name.c_str());
      return FAILED;
    }
    FE_CHECK(ir_index >= ir_input_2_range.size(), FE_LOGW("Index [%zu] is out of range.", ir_index), return FAILED);
    auto &range = ir_input_2_range[ir_index];
    FE_LOGD("Input[%zu]/IR_idx[%zu] with name[%s]/type[%u] and range[%zu/%zu].", in_idx, ir_index,
            input_name.c_str(), input_type, range.first, range.second);
    if (input_type == static_cast<uint32_t>(OpParamType::REQUIRED)) {
      if (range.second == 0) {
        FE_LOGW("The required input size for Op [%s, %s] is 0.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
        return FAILED;
      }
      format_desc_.Append(ge::AddrType::INPUT, ir_index);
    } else if (input_type == static_cast<uint32_t>(OpParamType::OPTIONAL)) {
      auto graph_idx = input_graph_idx[in_idx];
      bool is_used = (graph_idx != -1 && op_desc_->GetInputDesc(graph_idx).IsValid() == ge::GRAPH_SUCCESS);
      FE_LOGD("Ir Input[%u] with graph id [%d] has used flag as %d.", in_idx, graph_idx, is_used);
      if (is_used || is_input_gen_place_) {
        format_desc_.Append(ge::AddrType::INPUT, ir_index);
      }
    } else if (input_type == static_cast<uint32_t>(OpParamType::DYNAMIC)) {
      if (dy_ir_idx == ir_index) {
        continue;
      }
      dy_ir_idx = ir_index;
      if (range.second == 0 || (range.second > in_num)) {
        // if dynamic is 0 need gen place holder, here need append INPUT desc
        FE_LOGW("Op[%s] Dy Input[%zu] size[%zu] invalid.", op_desc_->GetNamePtr(), ir_index, range.second);
        return FAILED;
      }
      AddDynamicDesc(range, ir_index, true);
    } else {
      return FAILED;
    }
  }
  if (!dyn_io_vv_.empty()) {
    (void)ge::AttrUtils::SetListListInt(op_desc_, kDyInputsIndexes, dyn_io_vv_);
  }
  return SUCCESS;
}

void ArgsFormatConstructor::ConstructOptOutputArgs(size_t ir_index) {
  auto output_desc_ptr = op_desc_->MutableOutputDesc(ir_index);
  if (output_desc_ptr == nullptr) {
    return;
  }
  int32_t calc_type = 0;
  (void)ge::AttrUtils::GetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  if (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY)) {
    FE_LOGD("Op[%s:%s] opt output[%zu] mem type empty, is_output_gen_place_[%d]", op_desc_->GetNamePtr(),
            op_desc_->GetTypePtr(), ir_index, is_output_gen_place_);
    if (is_output_gen_place_) {
      format_desc_.Append(ge::AddrType::PLACEHOLDER);
    }
    return;
  }
  format_desc_.Append(ge::AddrType::OUTPUT, ir_index);
  return;
}

bool ArgsFormatConstructor::GetOpOutputInfo(std::vector<uint32_t> &output_type_list,
    std::vector<std::string> &output_name_list, std::map<size_t, std::pair<size_t, size_t>> &ir_out_2_range) {
  (void)ge::AttrUtils::GetListInt(op_desc_, kOutputParaTypeList, output_type_list);
  (void)ge::AttrUtils::GetListStr(op_desc_, kOutputNameList, output_name_list);
  if (ge::OpDescUtils::GetIrOutputDescRange(op_desc_, ir_out_2_range) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Get ir input range failed.");
    return false;
  }
  if (output_name_list.size() != output_type_list.size()) {
    FE_LOGW("Output name size[%zu] not equal with type size[%zu].", output_name_list.size(), output_type_list.size());
    return false;
  }
  size_t all_num = 0;
  for (const auto &range : ir_out_2_range) {
    all_num += range.second.second;
  }
  if (all_num > output_name_list.size()) {
    FE_LOGW("Output name size [%zu] is smaller than IR size [%zu].", output_name_list.size(), all_num);
    return false;
  }
  return true;
}

Status ArgsFormatConstructor::ConstructOutArgsDescByOps(const std::vector<std::pair<std::string, ge::IrOutputType>>
                                                       &ir_outputs) {
  std::vector<uint32_t> output_type_list;
  std::vector<std::string> output_name_list;
  // (ir index, (input index on graph, range))
  std::map<size_t, std::pair<size_t, size_t>> ir_out_2_range;
  if (!GetOpOutputInfo(output_type_list, output_name_list, ir_out_2_range)) {
    return FAILED;
  }
  size_t out_num = output_type_list.size();
  size_t dy_ir_idx = 0xFFFF;
  for (size_t out_idx = 0; out_idx < out_num; ++out_idx) {
    auto output_type = output_type_list[out_idx];
    auto &output_name = output_name_list[out_idx];
    size_t ir_index = 0;
    if (!GetOutIrIndexByName(ir_outputs, output_name, ir_index)) {
      FE_LOGW("Op[%s,%s] output[%s] not in ir.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), output_name.c_str());
      return FAILED;
    }
    FE_CHECK(ir_index >= ir_out_2_range.size(), FE_LOGW("Index [%zu] is out of range.", ir_index), return FAILED);
    auto &range = ir_out_2_range[ir_index];
    FE_LOGD("Output[%zu]/IR_idx[%zu] with name[%s]/type[%u] and range[%zu/%zu].", out_idx, ir_index,
            output_name.c_str(), output_type, range.first, range.second);
    if (output_type == static_cast<uint32_t>(OpParamType::REQUIRED)) {
      if (range.second == 0) {
        FE_LOGW("Op[%s,%s] Output[%zu][%s] not found in ir.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), ir_index,
                output_name.c_str());
        return FAILED;
      }
      format_desc_.Append(ge::AddrType::OUTPUT, ir_index);
    } else if (output_type == static_cast<uint32_t>(OpParamType::DYNAMIC)) {
      if (dy_ir_idx == ir_index) {
        continue;
      }
      dy_ir_idx = ir_index;
      if (range.second == 0 || (range.second > out_num)) {
        // if dynamic is 0 need gen place holder, here need append INPUT desc
        FE_LOGW("Op[%s,%s]Dynamic input[%zu] no use.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), ir_index);
        return FAILED;
      }
      AddDynamicDesc(range, ir_index, false);
    } else if (output_type == static_cast<uint32_t>(OpParamType::OPTIONAL)) {
      ConstructOptOutputArgs(ir_index);
    } else {
      return FAILED;
    }
  }
  if (!dyn_io_vv_.empty()) {
    (void)ge::AttrUtils::SetListListInt(op_desc_, kDyOutputsIndexes, dyn_io_vv_);
  }
  return SUCCESS;
}

inline bool NeedConstructByIR(const ge::OpDescPtr op_desc, bool is_dy_folded, bool is_gen_place) {
  bool ret = (!op_desc->GetIrInputs().empty() || !op_desc->GetIrOutputs().empty()) &&
             (!ge::AttrUtils::HasAttr(op_desc, kAttrNameIsFusionOp));
  ret &= (is_dy_folded || is_gen_place);
  FE_LOGD("Need by ir[%d].", ret);
  return ret;
}

void ArgsFormatConstructor::ConstructArgsDescByGraph() {
  format_desc_.Clear();
  if (need_sync_) {
    FE_LOGD("Add ffts addr arg.");
    format_desc_.Append(ge::AddrType::FFTS_ADDR);
  }
  size_t all_num = op_desc_->GetAllInputsSize();
  size_t arg_id = 0;
  for (size_t id = 0; id < all_num; ++id) {
    bool has_input = (op_desc_->GetInputDescPtr(id) != nullptr);
    FE_LOGD("Input[%zu] is used flag:%d.", id, has_input);
    if (has_input) {
      format_desc_.Append(ge::AddrType::INPUT_INSTANCE, arg_id++);
    } else if (is_input_gen_place_) {
      format_desc_.Append(ge::AddrType::PLACEHOLDER);
    }
  }
  all_num = op_desc_->GetOutputsSize();
  arg_id = 0;
  for (size_t id = 0; id < all_num; ++id) {
    auto output_desc_ptr = op_desc_->MutableOutputDesc(id);
    if (output_desc_ptr == nullptr) {
      continue;
    }
    int32_t calc_type = 0;
    (void)ge::AttrUtils::GetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY)) {
      FE_LOGD("Op[%s:%s] Output[%zu] is always empty, is_output_gen_place_[%d]", op_desc_->GetNamePtr(),
              op_desc_->GetTypePtr(), id, is_output_gen_place_);
      if (is_output_gen_place_) {
        format_desc_.Append(ge::AddrType::PLACEHOLDER);
      }
      continue;
    }
    // normal output
    format_desc_.Append(ge::AddrType::OUTPUT_INSTANCE, arg_id++);
  }
  return;
}

// INPUT_INSTANCE: arg_id represent input edge index in graph
Status ArgsFormatConstructor::ConstructInArgsDesc() {
  const auto &ir_inputs = op_desc_->GetIrInputs();
  if (by_ir_) {
    if (ConstructInArgsDescByOps(ir_inputs) == SUCCESS) {
      return SUCCESS;
    }
    FE_LOGW("Op[%s][%s] cannot be constructed by IR.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    by_ir_ = false;
  }
  if (is_dy_folded_) {
    REPORT_FE_ERROR("Node[%s][%s] needs to be dynamically folded but does not have IR.",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  return SUCCESS;
}

Status ArgsFormatConstructor::ConstructOutArgsDesc() {
  const auto &ir_outputs = op_desc_->GetIrOutputs();
  if (by_ir_) {
    if (ConstructOutArgsDescByOps(ir_outputs) == SUCCESS) {
      return SUCCESS;
    }
    FE_LOGW("Op[%s][%s] cannot be constructed by IR.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  }
  if (is_dy_folded_) {
    REPORT_FE_ERROR("Node [%s][%s] needs to be dynamically folded but does not have output IR.",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  ConstructArgsDescByGraph();
  return SUCCESS;
}

Status ArgsFormatConstructor::ConstructNodeArgsDesc() {
  FE_CHECK_NOTNULL(op_desc_);
  std::string dyn_mode;
  (void)ge::AttrUtils::GetStr(op_desc_, fe::kAttrDynamicParamMode, dyn_mode);
  is_dy_folded_ = (dyn_mode == fe::kFoldedWithDesc);
  std::string input_opt_mode;
  (void)ge::AttrUtils::GetStr(op_desc_, fe::kAttrOptionalInputMode, input_opt_mode);
  is_input_gen_place_ = (input_opt_mode == fe::kGenPlaceholder);
  std::string output_opt_mode;
  (void)ge::AttrUtils::GetStr(op_desc_, fe::kAttrOptionalOutputMode, output_opt_mode);
  is_output_gen_place_ = (output_opt_mode == fe::kGenPlaceholder);
  std::string core_type;
  (void)ge::AttrUtils::GetStr(op_desc_, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  need_sync_ = (core_type == kCoreTypeMixEnhance) && (PlatformUtils::Instance().GetFftsMode() == FFTS_MODE_FFTS_PLUS);
  need_sync_ = need_sync_ || (op_desc_->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME));
  if (need_sync_) {
    FE_LOGD("Add ffts addr arg.");
    format_desc_.Append(ge::AddrType::FFTS_ADDR);
  }
  by_ir_ = NeedConstructByIR(op_desc_, is_dy_folded_, (is_input_gen_place_ || is_output_gen_place_));
  if (ConstructInArgsDesc() != SUCCESS) {
    FE_LOGE("Node [%s][%s] failed to in args desc.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  dyn_io_v_.clear();
  dyn_io_vv_.clear();
  if (ConstructOutArgsDesc() != SUCCESS) {
    FE_LOGE("Node[%s][%s] construct out args desc failed.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return FAILED;
  }
  auto tiling_type = is_ffts_plus_ ? ge::AddrType::TILING_FFTS : ge::AddrType::TILING;
  if (fe::UnknownShapeUtils::IsUnknownShapeOp(*op_desc_)) {
    format_desc_.Append(ge::AddrType::WORKSPACE);
    format_desc_.Append(tiling_type);
  } else {
    auto work_size = op_desc_->GetWorkspaceBytes().size();
    std::vector<uint32_t> aicpu_workspace_type;
    ge::AttrUtils::GetListInt(op_desc_, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, aicpu_workspace_type);
    for (size_t i = 0; i < work_size; ++i) {
      if ((IsCustomOp(*op_desc_) || IsPrefixOpsPath(*op_desc_)) && work_size == aicpu_workspace_type.size() &&
          aicpu_workspace_type[i] == ge::AicpuWorkSpaceType::CUST_LOG) {
        FE_LOGI("Node[%s][%s] custom op tiling sink remove CUST_LOG workspace[%zu]", op_desc_->GetNamePtr(),
                op_desc_->GetTypePtr(), i);
        continue;
      }
      format_desc_.Append(ge::AddrType::WORKSPACE, i);
    }
    if (fe::OpTensorUtils::IsStaticReuseBinaryOp(op_desc_)) {
      format_desc_.Append(tiling_type);
    }
  }
  if (!is_ffts_plus_ && ge::AttrUtils::HasAttr(op_desc_, ge::GLOBALWORKSPACE_TYPE)) {
    format_desc_.Append(ge::AddrType::OVERFLOW_ADDR);
  }
  return SUCCESS;
}

Status ArgsFormatConstructor::GetArgsSize(size_t &args_size) {
  if (format_desc_.GetArgsSize(op_desc_, args_size) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

std::string ArgsFormatConstructor::GetArgsFormatString() {
  return format_desc_.ToString();
}
}  // namespace fe
