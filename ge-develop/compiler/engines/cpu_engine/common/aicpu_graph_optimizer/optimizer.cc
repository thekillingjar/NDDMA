/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimizer.h"

#include "config/config_file.h"
#include "error_code/error_code.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/util.h"
#include "common/op_slice_util.h"
#include "common/aicore_util_types.h"

using namespace std;
using namespace ge;

namespace {
const string kPlaceHolderOpType = "PlaceHolder";
const string kFunctionOp = "FunctionOp";
const string kFrameworkOp = "FrameworkOp";
constexpr uint64_t kOpCheckModeOff = 0;
constexpr uint64_t kOpCheckModeOn = 1;
constexpr int64_t kFormatAgnostic = 1;
constexpr int64_t kShape4d = 4;
const std::string kTfOpFussion = "tf_op_fussion";

const set<Format> kGeFormatSet = {Format::FORMAT_NCHW, Format::FORMAT_NHWC, Format::FORMAT_ND};

const bool kAicpuSupportStrideWrite = false;
const int32_t kNCHWDimOfN = 0;
const int32_t kNCHWDimOfC = 1;
const int32_t kNCHWDimOfH = 2;
const int32_t kNCHWDimOfW = 3;

const int32_t kNHWCDimOfN = 0;
const int32_t kNHWCDimOfH = 1;
const int32_t kNHWCDimOfW = 2;
const int32_t kNHWCDimOfC = 3;
}  // namespace

namespace aicpu {
Status Optimizer::GetFrameworkOpType(const OpDescPtr &op_desc_ptr,
                                     string &op_type) const {
  // op_desc_ptr already check not null
  const string *original_type = AttrUtils::GetStr(op_desc_ptr, kOriginalType);
  CHECK_RES_BOOL((original_type != nullptr),
      ErrorCode::GET_ATTR_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
          kOriginalType.c_str(), op_desc_ptr->GetName().c_str()))
  if (original_type->empty()) {
    AICPU_REPORT_INNER_ERR_MSG("Attr[%s] is empty, op[%s].", kOriginalType.c_str(),
        op_desc_ptr->GetName().c_str());
    return STR_IS_EMPTY;
  }
  ge::OpDescUtilsEx::SetType(const_cast<OpDescPtr &>(op_desc_ptr), *original_type);
  op_type = *original_type;
  return SUCCESS;
}

void Optimizer::GetTfOpFussionOoLevel() {                                   
  std::string opt_value;
  auto status = GetThreadLocalContext().GetOo().GetValue(kTfOpFussion, opt_value);
  AICPUE_LOGI("Get option[%s], opt_value[%s], status[%u].",
               kTfOpFussion.c_str(), opt_value.c_str(), status);
  if (opt_value == "false") {
    AICPUE_LOGI("Tf op fussion may be disable in current level.");
    is_tf_op_fussion_oo_enable_ = false;
  } else {
    is_tf_op_fussion_oo_enable_ = true;
  }
  return;
}

void Optimizer::InitOpCheckMode() {
  // get TfDebugMode from config file
  string op_check_mode;
  if (ConfigFile::GetInstance().GetValue(kOpCheckMode, op_check_mode)) {
    uint64_t result = kOpCheckModeOff;
    if (StringToNum(op_check_mode, result).state != ge::SUCCESS) {
      AICPUE_LOGW("Tran op_check_mode [%s] to integer failed. default value is 0.",
                  op_check_mode.c_str());
      return;
    }
    if (result == kOpCheckModeOn) {
      AICPUE_LOGI("OpCheckMode is on.");
      op_check_mode_ = true;
      return;
    }
    AICPUE_LOGI("OpCheckMode is off.");
    return;
  }
  AICPUE_LOGW("Get [op_check_mode] from config file failed. op check mode is off.");
}

ge::Status Optimizer::InitSlicePattern(const std::string &slice,
                                       const ge::NodePtr &node) const {
  if (slice == "") {
    return SUCCESS;
  }
  const std::map<std::string, fe::SlicePattern> str_slice_pattern_map {
        {"elemwise", fe::ELEMENT_WISE},
        {"elemwiseBroadcast", fe::ELEMENT_WISE_BROADCAST},
        {"broadcast", fe::BROADCAST},
        {"slidingWindow", fe::SLIDING_WINDOW},
        {"slidingWindowDeconv", fe::SLIDING_WINDOW_DECONV},
        {"cubeMatmul", fe::CUBE_MATMUL},
        {"reduce", fe::SLICE_PATTERN_REDUCE},
        {"resize", fe::SLICE_PATTERN_RESIZE},
        {"scatter", fe::SLICE_PATTERN_SCATTER},
        {"segment", fe::SLICE_PATTERN_SEGMENT},
  };
  AICPUE_LOGD("op[%s], slice[%s]", node->GetName().c_str(), slice.c_str());
  auto iter = str_slice_pattern_map.find(slice);
  if (iter == str_slice_pattern_map.end()) {
    AICPU_REPORT_INNER_ERR_MSG("slice[%s] is invalid", slice.c_str());
    return FAILED;
  }
  Status state = fe::OpSliceUtil::SetOpSliceInfo(node, iter->second, kAicpuSupportStrideWrite);
  if (state != SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("op[%s] SetOpSliceInfo[%s] fail[%u]",
                            node->GetName().c_str(), slice.c_str(), state);
    return state;
  }
  return SUCCESS;
}

Status Optimizer::OptimizeOriginalGraphJudgeInsert(
    const ComputeGraph &graph,
    const map<string, OpFullInfo> &all_op_info) const {
  for (const NodePtr &curr_node : graph.GetAllNodes()) {
    AICPU_CHECK_NOTNULL(curr_node)
    OpDescPtr curr_op_desc_ptr = curr_node->GetOpDesc();
    AICPU_CHECK_NOTNULL(curr_op_desc_ptr)
    string op_type = curr_op_desc_ptr->GetType();
    // if op type is placeholder or function_op or framework_op, skip it
    if (op_type == kPlaceHolderOpType || op_type == kFunctionOp ||
        op_type == kFrameworkOp) {
      AICPUE_LOGD("Current op type is [%s]. Don't need to set format.",
                  op_type.c_str());
      continue;
    }

    const string *op_attr_engine_name = AttrUtils::GetStr(curr_op_desc_ptr, ge::ATTR_NAME_OP_SPECIFIED_ENGINE_NAME);
    const string *op_attr_kernel_name = AttrUtils::GetStr(curr_op_desc_ptr, ge::ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME);
    if (((op_attr_engine_name != nullptr) && (*op_attr_engine_name == "DNN_VM_DVPP")) &&
        ((op_attr_kernel_name != nullptr) && (*op_attr_kernel_name == "dvpp_ops_kernel"))) {
      continue;
    }

    auto iter = all_op_info.find(op_type);
    if (iter != all_op_info.end()) {
      auto op_info = iter->second;
      bool format_agnostic = op_info.formatAgnostic;
      if (format_agnostic) {
        const string name = "_format_agnostic";
        bool b_ret = AttrUtils::SetInt(curr_op_desc_ptr, name, kFormatAgnostic);
        AICPU_IF_BOOL_EXEC(
            !(b_ret),
            AICPU_REPORT_INNER_ERR_MSG(
                "Call ge::AttrUtils::SetInt failed to set attr[%s], op[%s]",
                name.c_str(), curr_op_desc_ptr->GetName().c_str());
            return FAILED)
      } else {
        AICPU_CHECK_RES_WITH_LOG(
            UpdateInputFormatAndShape(op_info, curr_op_desc_ptr),
            "Call UpdateInputFormatAndShape function failed, op[%s].",
            curr_op_desc_ptr->GetType().c_str())
        AICPU_CHECK_RES_WITH_LOG(
            UpdateOutputFormatAndShape(op_info, curr_op_desc_ptr),
            "Call UpdateOutputFormatAndShape function failed, op[%s].",
            curr_op_desc_ptr->GetType().c_str())
      }
      ge::Status ret = InitSlicePattern(op_info.slicePattern, curr_node);
      if (ret != SUCCESS) {
        return ret;
      }
    }
  }
  return SUCCESS;
}

Status Optimizer::UpdateInputFormatAndShape(const OpFullInfo &op_info,
                                            const OpDescPtr &op_desc_ptr) const {
  uint32_t index = 0;
  for (GeTensorDescPtr input_desc_ptr : op_desc_ptr->GetAllInputsDescPtr()) {
    ge::Format src_format = input_desc_ptr->GetFormat();
    if (!kGeFormatSet.count(src_format)) {
      AICPUE_LOGD("input %u is not need update format and shape.", index);
      return SUCCESS;
    }

    map<string, string> formats = op_info.inOutFormat;
    string format_name = Stringcat("input", index);
    ge::Format dst_format;
    GetFormat(formats, format_name, dst_format);
    if ((dst_format == ge::FORMAT_NHWC) || (dst_format == ge::FORMAT_NCHW)) {
      (void)UpdateTensorDesc((*input_desc_ptr), src_format, dst_format);
      AICPU_CHECK_RES_WITH_LOG(op_desc_ptr->UpdateInputDesc(index, (*input_desc_ptr)),
          "Call UpdateInputDesc failed to update input[%u] desc, op[%s].",
          index, op_desc_ptr->GetName().c_str())
    }
    index++;
  }
  return SUCCESS;
}

void Optimizer::UpdateTensorDesc(GeTensorDesc &tensor_desc,
                                 const ge::Format &src_format,
                                 const ge::Format &dst_format) const {
  if (src_format != dst_format) {
    AICPUE_LOGD("update src_format[%s] to dst_format[%s]", ge::TypeUtils::FormatToSerialString(src_format).c_str(),
                ge::TypeUtils::FormatToSerialString(dst_format).c_str());
    tensor_desc.SetFormat(dst_format);
    vector<int64_t> dims = tensor_desc.GetShape().GetDims();
    if (dims.size() != kShape4d) {
      AICPUE_LOGW("Input tensor is not 4D, but it's format is 4D.");
      return;
    }
    vector<int64_t> newDims(dims);
    if (src_format == ge::FORMAT_NCHW) {
      newDims[0] = dims[kNCHWDimOfN];
      newDims[1] = dims[kNCHWDimOfH];
      newDims[2] = dims[kNCHWDimOfW];
      newDims[3] = dims[kNCHWDimOfC];
    } else if (src_format == ge::FORMAT_NHWC) {
      newDims[0] = dims[kNHWCDimOfN];
      newDims[1] = dims[kNHWCDimOfC];
      newDims[2] = dims[kNHWCDimOfH];
      newDims[3] = dims[kNHWCDimOfW];
    }
    tensor_desc.SetShape(GeShape(newDims));
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    auto ret = tensor_desc.GetShapeRange(shape_range);
    if (ret != GRAPH_SUCCESS) {
      AICPUE_LOGW("Get shape range failed");
      return;
    }
    if (shape_range.size() != kShape4d) {
      AICPUE_LOGW("shape range size is [%u], is not 4", shape_range.size());
    } else {
      std::vector<std::pair<int64_t, int64_t>> new_range;
      if (src_format == ge::FORMAT_NCHW) {
        new_range.emplace_back(shape_range[kNCHWDimOfN]);
        new_range.emplace_back(shape_range[kNCHWDimOfH]);
        new_range.emplace_back(shape_range[kNCHWDimOfW]);
        new_range.emplace_back(shape_range[kNCHWDimOfC]);
        tensor_desc.SetShapeRange(new_range);
      } else if (src_format == ge::FORMAT_NHWC) {
        new_range.emplace_back(shape_range[kNHWCDimOfN]);
        new_range.emplace_back(shape_range[kNHWCDimOfC]);
        new_range.emplace_back(shape_range[kNHWCDimOfH]);
        new_range.emplace_back(shape_range[kNHWCDimOfW]);
        tensor_desc.SetShapeRange(new_range);
      }
    }
  }
}

Status Optimizer::UpdateOutputFormatAndShape(const OpFullInfo &op_info,
                                             const OpDescPtr &op_desc_ptr) const {
  uint32_t index = 0;
  for (GeTensorDescPtr output_desc_ptr : op_desc_ptr->GetAllOutputsDescPtr()) {
    ge::Format src_format = output_desc_ptr->GetFormat();
    if (!kGeFormatSet.count(src_format)) {
      AICPUE_LOGD("output[%u] is not need update format and shape.", index);
      return SUCCESS;
    }

    map<string, string> formats = op_info.inOutFormat;
    string format_name = Stringcat("output", index);
    ge::Format dst_format;
    GetFormat(formats, format_name, dst_format);
    if ((dst_format == ge::FORMAT_NHWC) || (dst_format == ge::FORMAT_NCHW)) {
      (void)UpdateTensorDesc((*output_desc_ptr), src_format, dst_format);
      AICPU_CHECK_RES_WITH_LOG(op_desc_ptr->UpdateOutputDesc(index, (*output_desc_ptr)),
          "Call UpdateOutputDesc failed to update output[%u] desc, op[%s].",
          index, op_desc_ptr->GetName().c_str())
    }
    index++;
  }
  return SUCCESS;
}

void Optimizer::GetFormat(const map<string, string> &formats,
                          const string &format_name, ge::Format &format) const {
  auto iter = formats.find(format_name);
  if (iter != formats.end()) {
    string formatStr = iter->second;
    if (formatStr == "NHWC") {
      format = ge::FORMAT_NHWC;
    } else if (formatStr == "NCHW") {
      format = ge::FORMAT_NCHW;
    } else {
      format = ge::FORMAT_ND;
    }
  } else {
    format = ge::FORMAT_ND;
  }
}

REG_OPTION(kTfOpFussion)
  .LEVELS(ge::OoLevel::kO0)
  .DEFAULT_VALUES({{ge::OoLevel::kO0, "false"}, {ge::OoLevel::kO1, "false"},
                   {ge::OoLevel::kO2, "true"}, {ge::OoLevel::kO3, "true"}});
}  // namespace aicpu