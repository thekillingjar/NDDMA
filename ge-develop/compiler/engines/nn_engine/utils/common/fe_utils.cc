/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_utils.h"
#include <sys/time.h>
#include <atomic>
#include <sstream>
#include <thread>
#include <string>
#include <algorithm>
#include <cctype>
#include "common/string_utils.h"
#include "common/fe_inner_error_codes.h"
#include "common/math_util.h"
#include "common/util/op_info_util.h"
#include "runtime/rt.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "base/err_msg.h"

namespace fe {
namespace {
const std::vector<std::string> kRootGraphDataOpSet = {"Data", "RefData"};
const std::set<std::string> kHeavyDataOpType = {"ConstPlaceHolder"};
static const size_t kMaxTraceBufSize = 120;
const std::map<std::string, ge::DataType> STR_DTYPE_MAP{{"float32", ge::DT_FLOAT}};
const std::map<ge::DataType, std::string> DATATYPE_STRING_MAP {{ge::DT_FLOAT, "float32"}};
}

std::mutex g_report_error_msg_mutex;
int64_t GetMicroSecondTime() {
  struct timeval tv = {0, 0};
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    return 0;
  }
  if (tv.tv_sec < 0 || tv.tv_usec < 0) {
    return 0;
  }
  int64_t micro_multiples = 1000000;
  int64_t second = tv.tv_sec;
  int64_t second_to_micro = 0;
  FE_MUL_OVERFLOW(second, micro_multiples, second_to_micro)
  FE_ADD_OVERFLOW(second_to_micro, tv.tv_usec, second_to_micro)
  return second_to_micro;
}

uint64_t GetCurThreadId() {
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  std::string s_tid = oss.str();
  try {
    return std::stoull(s_tid);
  } catch (...) {
    FE_LOGW("Thread Id %s invalid.", s_tid.c_str());
    uint64_t invalid_thread_id = 0;
    return invalid_thread_id;
  }
}

std::string GetCurThreadIdStr() {
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  return oss.str();
}

uint64_t GetAtomicId() {
  static std::atomic<uint64_t> global_atomic_id(1);
  return global_atomic_id.fetch_add(1, std::memory_order_relaxed);
}

std::string FormatToStr(ge::Format format) { return ge::TypeUtils::FormatToSerialString(format); }

std::string DTypeToStr(ge::DataType d_type) { return ge::TypeUtils::DataTypeToSerialString(d_type); }

std::string GetImplTypeString(OpImplType op_impl_type) {
  auto iter = IMPL_TYPE_STRING_MAP.find(op_impl_type);
  if (iter == IMPL_TYPE_STRING_MAP.end()) {
    return "unknown-type " + std::to_string(op_impl_type);
  } else {
    return iter->second;
  }
}

std::string GetGeImplTypeString(domi::ImplyType ge_impl_type) {
  auto iter = GE_IMPL_TYPE_STRING_MAP.find(ge_impl_type);
  if (iter == GE_IMPL_TYPE_STRING_MAP.end()) {
    return "unknown-type " + std::to_string(static_cast<int64_t>(ge_impl_type));
  } else {
    return iter->second;
  }
}

bool IsInvalidImplType(const std::string &engine_name, const OpImplType &op_impl_type) {
  if (engine_name == AI_CORE_NAME && op_impl_type == EN_IMPL_HW_DSA) {
    return true;
  }
  if (engine_name == kDsaCoreName && op_impl_type != EN_IMPL_HW_DSA) {
    return true;
  }
  return false;
}

std::string GetPassTypeString(GraphFusionPassType pass_type) {
  auto iter = PASS_TYPE_STRING_MAP.find(pass_type);
  if (iter == PASS_TYPE_STRING_MAP.end()) {
    return "unknown-pass-type " + std::to_string(pass_type);
  } else {
    return iter->second;
  }
}

std::string GetBufferFusionPassTypeString(BufferFusionPassType pass_type) {
  auto iter = BUFFER_FUSION_PASS_TYPE_STRING_MAP.find(pass_type);
  if (iter == BUFFER_FUSION_PASS_TYPE_STRING_MAP.end()) {
    return "unknown-buffer-fusion-pass-type " + std::to_string(pass_type);
  } else {
    return iter->second;
  }
}

bool CheckFileEmpty(const std::string &file_name) {
  std::ifstream ifs(file_name);
  if (!ifs.is_open()) {
    REPORT_FE_ERROR("[GraphOpt][Init][CheckFileEmpty] The file [%s] does not exist or has been opened.",
                    file_name.c_str());
    return FILE_NOT_EXIST;
  }
  char c;
  ifs >> c;
  if (ifs.eof()) {
    ifs.close();
    return true;
  }
  ifs.close();
  return false;
}

bool IsRootGraphData(const std::string &op_node_type) {
  return std::find(kRootGraphDataOpSet.begin(), kRootGraphDataOpSet.end(), op_node_type) != kRootGraphDataOpSet.end();
}

bool IsHeavyDataOp(const std::string &op_type) {
  return kHeavyDataOpType.count(op_type) != 0;
}
int GetDataTypeBits(const ge::DataType data_type) {
  int data_type_size = ge::GetSizeByDataType(data_type);
  if (data_type_size <= 0) {
    return data_type_size;
  }
  if (data_type > ge::kDataTypeSizeBitOffset) {
    return data_type_size - ge::kDataTypeSizeBitOffset;
  }
  return data_type_size * ge::kBitNumOfOneByte;
}

void SaveErrorMessage(const std::string &error_code, const std::string &key, const std::string &value) {
  std::lock_guard<std::mutex> lock_guard(g_report_error_msg_mutex);
  std::vector<const char*> keys = {key.c_str()};
  std::vector<const char*> values = {value.c_str()};
  REPORT_PREDEFINED_ERR_MSG(error_code.c_str(), keys, values);
}

bool IsHeavyFormatMatmul(const ge::NodePtr &node) {
  bool ffts_switch = false;
  (void)ge::AttrUtils::GetBool(node->GetOwnerComputeGraph(), "_ffts_switch", ffts_switch);
  if (ffts_switch) {
    return false;
  }
  const std::string &op_type = node->GetOpDesc()->GetType();
  if (KFeFormatModeFilterOp.count(op_type) == 0) {
    return false;
  }
  for (auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
    if (input_desc == nullptr) {
      continue;
    }
    ge::Format cur_format = input_desc->GetFormat();
    auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(cur_format));
    if (primary_format == ge::FORMAT_FRACTAL_NZ) {
      return true;
    }
  }
  for (auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    if (output_desc == nullptr) {
      continue;
    }
    ge::Format cur_format = output_desc->GetFormat();
    auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(cur_format));
    if (primary_format == ge::FORMAT_FRACTAL_NZ) {
      return true;
    }
  }
  return false;
}

std::string GetStrTaskIdByMap(const std::map<uint64_t, int64_t> &task_scope_id_map) {
  std::string result;
  for (auto it = task_scope_id_map.begin(); it != task_scope_id_map.end(); ++it) {
    result = result + "[" + std::to_string(it->first) + "]";
    if (it != std::prev(task_scope_id_map.end())) {
      result += ",";
    }
  }
  return result;
}

void PrintOutputInplace(const ge::OpDescPtr &opdesc, const vector<vector<int64_t>> &inplace) {
  std::string str = "{";
  for (auto idx : inplace) {
    str += "{" + std::to_string(idx[0]) + "," + std::to_string(idx[1]) + "},";
  }
  str.back() = '}';
  FE_LOGD("Node[%s, %s] with outputPlaceAbility: %s", opdesc->GetTypePtr(), opdesc->GetNamePtr(), str.c_str());
}

void GetIrIdexInstance(const ge::OpDescPtr &opdesc, 
                       std::map<size_t, std::pair<size_t, size_t>> &input_ir_real_index,
                       std::map<size_t, std::pair<size_t, size_t>> &output_ir_real_index) {
  FE_LOGD("Get ir_index to real_index from GE");
  (void)ge::OpDescUtils::GetIrInputInstanceDescRange(opdesc, input_ir_real_index);
  (void)ge::OpDescUtils::GetIrOutputDescRange(opdesc, output_ir_real_index);
}

Status TransDtypeToString(const ge::DataType &dtype, string &dtype_string) {
  auto data_type_iter = DATATYPE_STRING_MAP.find(dtype);
  if (data_type_iter != DATATYPE_STRING_MAP.end()) {
    dtype_string = data_type_iter->second;
    return SUCCESS;
  }

  dtype_string = ge::TypeUtils::DataTypeToSerialString(dtype);
  auto indx = dtype_string.find("_");
  if (indx == string::npos) {
    FE_LOGW("The current dtype[%d] is invalid", dtype);
    return FAILED;
  }

  dtype_string = dtype_string.substr(indx + 1);
  transform(dtype_string.begin(), dtype_string.end(), dtype_string.begin(), ::tolower);
  return SUCCESS;
}

Status TransStringToDtype(const string &dtype_str, ge::DataType &dtype) {
  auto indx = STR_DTYPE_MAP.find(const_cast<string&>(dtype_str));
  if (indx != STR_DTYPE_MAP.end()) {
    dtype = indx->second;
    return SUCCESS;
  }
  string fe_dtype_str = const_cast<string&>(dtype_str);
  transform(fe_dtype_str.begin(), fe_dtype_str.end(), fe_dtype_str.begin(), ::toupper);
  std::string ge_dtype_string = "DT_" + fe_dtype_str;
  dtype = ge::TypeUtils::SerialStringToDataType(ge_dtype_string);
  if (dtype == ge::DT_UNDEFINED) {
    FE_LOGW("The current dtype[%s] string is invalid", dtype_str.c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
