/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/format_dtype_op_builtin_selector.h"
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/configuration.h"
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/fe_utils.h"
#include "common/format/axis_name_util.h"
#include "common/fe_op_info_common.h"
#include "common/util/json_util.h"
#include "platform/platform_info.h"
#include "graph/detail/attributes_holder.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"

namespace {
constexpr size_t kNumTwo = 2;
}
namespace fe {
namespace {
Status GetFormatDtypeForAllTensors(const vector<InputOrOutputInfoPtr> &all_tensor_info,
                                   map<string, vector<ge::Format>> &format_map, const ge::OpDescPtr &op_desc,
                                   FormatDtypeInfo &format_dtype_res, string &key) {
  for (const auto &input_or_output : all_tensor_info) {
    auto &old_data_types = input_or_output->GetDataType();
    string unique_name = input_or_output->GetUniqueName();
    key = unique_name;
    if (format_map.find(unique_name) == format_map.end()) {
      REPORT_FE_ERROR(
          "[GraphOpt][FmtJdg][GetFmtDtype] Op[name=%s,type=%s]: fail to find the key[%s] from the formatMap.",
          op_desc->GetName().c_str(), op_desc->GetType().c_str(), unique_name.c_str());
      return FAILED;
    }

    vector<ge::Format> old_formats = format_map[unique_name];
    vector<ge::Format> new_formats;
    vector<ge::DataType> new_data_types;
    if (GenerateUnionFormatDtype(old_formats, old_data_types, new_formats, new_data_types) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FmtJdg][GenFmtUnion][Op%s,type=%s] Failed to GenerateUnionFormatDtype",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    format_dtype_res.format_map[input_or_output->GetUniqueName()] = new_formats;
    format_dtype_res.data_type_map[input_or_output->GetUniqueName()] = new_data_types;
  }
  return SUCCESS;
}

uint64_t CalcTotalSize(const vector<int64_t> &axis_new_value, const ge::GeShape &new_shape, const ge::OpDesc &op_desc,
                       int data_size) {
  int64_t axis_value_size = 1;
  int64_t not_reduce_size = 1;
  for (size_t i = 0; i < new_shape.GetDimNum(); i++) {
    auto iter = find(axis_new_value.begin(), axis_new_value.end(), i);
    if (iter != axis_new_value.end()) {
      FE_MUL_OVERFLOW(axis_value_size, new_shape.GetDim(i), axis_value_size);
    } else {
      FE_MUL_OVERFLOW(not_reduce_size, new_shape.GetDim(i), not_reduce_size);
    }
  }
  /* The min size needed is size of reduce axis plus size not reduce axis multiply coefficients
   * coefficient is the max coefficient of reduce op : ReduceMean whose value is 2. */
  uint64_t tmp1 = 0;
  FE_ADD_OVERFLOW(axis_value_size, not_reduce_size, tmp1);
  uint64_t total_size = 0;
  FE_MUL_OVERFLOW(tmp1, data_size, total_size);
  FE_MUL_OVERFLOW(total_size, REDUCE_COEFFICIENTS, total_size);
  FE_LOGD("Op[name=%s,type=%s]: min size needed is %lu.", op_desc.GetName().c_str(), op_desc.GetType().c_str(),
          total_size);
  return total_size;
}
} // namespace

FormatDtypeOpBuiltinSelector::FormatDtypeOpBuiltinSelector() : FormatDtypeSelectorBase() {}

FormatDtypeOpBuiltinSelector::~FormatDtypeOpBuiltinSelector() {}

Status FormatDtypeOpBuiltinSelector::GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const ge::NodePtr &node,
                                               const bool &is_dynamic_check,
                                               map<string, vector<ge::Format>> &format_map) {
  Status status = GetAllFormatsFromOpDesc(node->GetOpDesc(), format_map);
  if (status != SUCCESS) {
    HeavyFormatInfo heavy_format_info;
    FormatDtypeInfo format_dtype_info;
    if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
        format_dtype_info) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptedFmt] Op[name=%s, type=%s]: failed to GetDynamicFormatAndDtype.",
                      node->GetNamePtr(), node->GetTypePtr());
      return FAILED;
    }
    if (!format_dtype_info.format_map.empty()) {
      format_map = std::move(format_dtype_info.format_map);
    }
  }
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                           const ge::NodePtr &node,
                                                           const bool &is_dynamic_check,
                                                           FormatDtypeInfo &format_dtype_res) {
  // 1. get the old_format from op_desc. if failed, get GetDynamicFormatAndDtype
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  FE_LOGD("Op[name=%s,type=%s]: start to GetSupportFormatDtype.", op_name.c_str(), op_type.c_str());
  FormatDtypeInfo format_dtype_info;
  Status status = GetAllFormatsFromOpDesc(op_desc, format_dtype_info.format_map);
  Status ret_status = GetAllSubFormatsFromOpDesc(op_desc, format_dtype_res.sub_format_map);
  if (status != SUCCESS || ret_status != SUCCESS) {
    HeavyFormatInfo heavy_format_info;
    if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
        format_dtype_info) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptedFmt] Op[name=%s, type=%s]: failed to GetDynamicFormatAndDtype.",
                      op_name.c_str(), op_type.c_str());
      return FAILED;
    }
  }

  // 2. get the data_types from op_kernel, union format and data_type
  string input_key;
  Status ret = GetFormatDtypeForAllTensors(op_kernel_info_ptr->GetAllInputInfo(), format_dtype_info.format_map,
                                           op_desc, format_dtype_res, input_key);
  if (ret != SUCCESS) {
    return ret;
  }

  string output_key;
  ret = GetFormatDtypeForAllTensors(op_kernel_info_ptr->GetAllOutputInfo(), format_dtype_info.format_map, op_desc,
                                    format_dtype_res, output_key);
  if (ret != SUCCESS) {
    return ret;
  }

  if (GetReduceNewFormatDType(op_kernel_info_ptr, op_desc, input_key, output_key, format_dtype_res) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Setcheck][GetSptedFmt] Op[name=%s,type=%s]: failed to GetReduceNewFormatDType.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  LogFormatMap(format_dtype_res.format_map);
  LogDataTypeMap(format_dtype_res.data_type_map);
  LogSubformatMap(format_dtype_res.sub_format_map);
  FE_LOGD("Op[name=%s,type=%s]: end to GetSupportFormatDtype.", op_name.c_str(), op_type.c_str());
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                       const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                       const ge::NodePtr &node, vector<ge::Format> &result) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  vector<ge::DataType> new_data_types;
  GetUnionFormatDtype(input_or_output_info_ptr, op_desc, result, new_data_types);
  vector<uint32_t> new_sub_formats;

  string unique_name = input_or_output_info_ptr->GetUniqueName();
  if (GetReduceNewFormatDTypeVec(op_kernel_info_ptr, node, unique_name, new_data_types, result, new_sub_formats)
      != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptFmt] Op[name=%s, type=%s]: Failed to get ReduceNewFormatDTypeVec.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s,type=%s]: support format[%s] for the %s.", op_name.c_str(), op_type.c_str(),
          GetStrByFormatVec(result).c_str(), unique_name.c_str());
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                               const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                               const ge::NodePtr &node, vector<ge::Format> &format_res,
                                                               vector<uint32_t> &sub_format_res,
                                                               uint32_t sub_format) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  vector<ge::DataType> new_data_types;
  GetUnionFormatDtype(input_or_output_info_ptr, op_desc, format_res, new_data_types);
  string unique_name = input_or_output_info_ptr->GetUniqueName();
  OpPattern op_pattern = op_kernel_info_ptr->GetOpPattern();
  if ((op_pattern == OP_PATTERN_BROADCAST || op_pattern == OP_PATTERN_BROADCAST_ENHANCED) &&
      GetBroadcastNewSubformatVec(op_kernel_info_ptr, node, unique_name, sub_format, sub_format_res) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptSubfmt] Op[name=%s,type=%s]: Failed to GetBroadcastNewSubformatVec.",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  if (GetReduceNewFormatDTypeVec(op_kernel_info_ptr, node, unique_name, new_data_types, format_res, sub_format_res)
      != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptFmt] Op[name=%s, type=%s]: Failed to get ReduceNewFormatDTypeVec.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s,type=%s]: support subformat[%s] for the %s.", op_name.c_str(), op_type.c_str(),
          GetStrBySubFormatVec(sub_format_res).c_str(), unique_name.c_str());
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                         const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                         const ge::NodePtr &node, vector<ge::DataType> &result) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  vector<ge::Format> new_formats;
  GetUnionFormatDtype(input_or_output_info_ptr, op_desc, new_formats, result);
  vector<uint32_t> new_sub_formats;

  string unique_name = input_or_output_info_ptr->GetUniqueName();
  if (GetReduceNewFormatDTypeVec(op_kernel_info_ptr, node, unique_name, result, new_formats, new_sub_formats)
      != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptType] Op[name=%s, type=%s]: Failed to get ReduceNewFormatDTypeVec.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  FE_LOGD("Op[name=%s,type=%s]: support data_types[%s] for the %s.", op_name.c_str(), op_type.c_str(),
          GetStrByDataTypeVec(result).c_str(), unique_name.c_str());
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetUnionFormatDtype(const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                                         const ge::OpDescPtr &op_desc, vector<ge::Format> &new_formats,
                                                         vector<ge::DataType> &new_data_types) {
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  string unique_name = input_or_output_info_ptr->GetUniqueName();

  // 1. get the formats form op_desc
  vector<ge::Format> old_formats;
  if (GetFormatFromOpDescByKey(op_desc, unique_name, old_formats) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetUnionFmt][Op %s,type=%s]: failed to get formats by key %s.", op_name.c_str(),
                    op_type.c_str(), unique_name.c_str());
    return FAILED;
  }

  // 2. union format and data_type, return formats
  if (GenerateUnionFormatDtype(old_formats, input_or_output_info_ptr->GetDataType(), new_formats, new_data_types) !=
      SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetUnionFmt][Op %s,type %s]: Failed to generate union for %s.", op_name.c_str(),
                    op_type.c_str(), unique_name.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetDynamicFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                           const ge::NodePtr &node, const bool &is_dynamic_check,
                                                           const HeavyFormatInfo &heavy_format_info,
                                                           FormatDtypeInfo &format_dtype_info,
                                                           uint32_t sub_format) {
  (void)is_dynamic_check;
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  string op_name = op_desc->GetName();
  string op_type = op_desc->GetType();
  FE_LOGD("Op[name=%s,type=%s]: start to GetDynamicFormatDtype.", op_name.c_str(), op_type.c_str());

  // 1. input_index_map, output_index_map
  IndexNameMap input_index_map;
  IndexNameMap output_index_map;
  if (GetInputOutputNameMap(*op_desc, op_kernel_info_ptr, input_index_map, output_index_map) != SUCCESS) {
    return FAILED;
  }

  // 2. get the origin info
  OriginInfoPtr origin_info_ptr = nullptr;
  FE_MAKE_SHARED(origin_info_ptr = make_shared<OriginInfo>(), return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (InitOriginInfo(op_desc, op_kernel_info_ptr, input_index_map, output_index_map, origin_info_ptr,
      format_dtype_info.format_map, format_dtype_info.sub_format_map) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][Built-In-Selector][Op %s,type=%s]: initialization not successfully.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  if (sub_format > UINT16_MAX || sub_format < GROUPS_DEFAULT_VALUE) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][Built-In-Selector][Op %s,type=%s]: sub_format[%u] invalid.",
                    op_name.c_str(), op_type.c_str(), sub_format);
    return FAILED;
  }
  FormatProccessArgs args;
  args.op_pattern = op_kernel_info_ptr->GetOpPattern();
  args.origin_info_ptr = origin_info_ptr;
  args.propagat_primary_format = heavy_format_info.expected_heavy_format;
  args.propagat_sub_format = sub_format;

  // 3. heavy format
  for (const auto &heavy_format : FE_HEAVY_FORMAT_VECTOR) {
    string heavy_format_str = ge::TypeUtils::FormatToSerialString(heavy_format);
    args.support_format = heavy_format;
    auto processer = BuildFormatProcess(args);
    if (processer == nullptr) {
      FE_LOGD("Op[name=%s,type=%s]: can't find the proccesser of heavy format %s.", op_name.c_str(), op_type.c_str(),
              heavy_format_str.c_str());
      continue;
    }

    FormatProccessResult result;
    FE_LOGD("Op[name=%s,type=%s]: start to process format %s.", op_name.c_str(), op_type.c_str(),
            heavy_format_str.c_str());
    if (processer->Process(*op_desc, args, result) == SUCCESS) {
      FE_LOGD("Op[name=%s,type=%s]: end to process format %s, support it.", op_name.c_str(), op_type.c_str(),
              heavy_format_str.c_str());
      if (UpdateFormatMap(op_kernel_info_ptr, result, input_index_map, output_index_map,
          format_dtype_info.format_map, format_dtype_info.sub_format_map) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][GenFmt][UptFmtMap][Op %s,type=%s]: update input and output formats not successfully.",
                        op_name.c_str(), op_type.c_str());
        return FAILED;
      }
    }
  }

  LogFormatMap(format_dtype_info.format_map);
  FE_LOGD("Op[name=%s,type=%s]: end to GetDynamicFormatDtype.", op_name.c_str(), op_type.c_str());
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                      const ge::NodePtr &node, bool is_dynamic_check,
                                                      std::vector<bool> &fallback_res) {
  Status status = GetAllFallbackFromOpDesc(node->GetOpDesc(), fallback_res);
  if (status != SUCCESS) {
    HeavyFormatInfo heavy_format_info;
    FormatDtypeInfo format_dtype_info;
    if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
        format_dtype_info) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FmtJdg][GetSptedFmt] Op[name=%s, type=%s]: failed to GetDynamicFormatAndDtype.",
                      node->GetNamePtr(), node->GetTypePtr());
      return FAILED;
    }
    if (!format_dtype_info.fallback_flags.empty()) {
      fallback_res = std::move(format_dtype_info.fallback_flags);
    }
  }
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::InitOriginInfo(const ge::OpDescPtr &op_desc,
                                                    const OpKernelInfoPtr &op_kernel_info_ptr,
                                                    const IndexNameMap &input_index_map,
                                                    const IndexNameMap &output_index_map, OriginInfoPtr origin_info_ptr,
                                                    map<string, vector<ge::Format>> &format_res,
                                                    map<string, vector<uint32_t>> &sub_format_res) const {
  vector<ge::Format> input_formats;
  vector<ge::DataType> input_dtypes;
  vector<ge::GeShape> input_shapes;
  vector<ge::Format> output_formats;
  vector<ge::GeShape> output_shapes;

  uint32_t index = 0;
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> all_input_desc_ptr = op_desc->GetAllInputsDescPtr();
  for (const auto &input_desc : all_input_desc_ptr) {
    auto origin_format = input_desc->GetOriginFormat();
    auto origin_dtype = input_desc->GetOriginDataType();
    auto origin_shape = input_desc->GetOriginShape();
    input_formats.push_back(origin_format);
    input_dtypes.push_back(origin_dtype);
    input_shapes.push_back(origin_shape);
    FE_LOGD("input[%u]: origin_format=[%s], origin_data_type=[%s], origin_shape=[%s].", index,
            ge::TypeUtils::FormatToSerialString(origin_format).c_str(),
            ge::TypeUtils::DataTypeToSerialString(origin_dtype).c_str(), GetShapeDims(origin_shape).c_str());

    InputOrOutputInfoPtr input_or_output_info_ptr =
        GetInputOrOutputUniqueName(true, input_index_map, index, op_kernel_info_ptr);
    FE_CHECK_NOTNULL(input_or_output_info_ptr);
    string unique_name = input_or_output_info_ptr->GetUniqueName();
    format_res[unique_name] = FE_ORIGIN_FORMAT_VECTOR;
    sub_format_res[unique_name] = vector<uint32_t>(1U, DEFAULT_SUB_FORMAT);
    index++;
  }

  index = 0;
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> all_output_desc_ptr = op_desc->GetAllOutputsDescPtr();
  for (const auto &output_desc : all_output_desc_ptr) {
    auto origin_format = output_desc->GetOriginFormat();
    output_formats.push_back(origin_format);
    output_shapes.push_back(output_desc->GetOriginShape());
    FE_LOGD("output[%u]: origin_shape=[%s].", index, GetShapeDims(output_desc->GetOriginShape()).c_str());
    InputOrOutputInfoPtr input_or_output_info_ptr =
        GetInputOrOutputUniqueName(false, output_index_map, index, op_kernel_info_ptr);
    FE_CHECK_NOTNULL(input_or_output_info_ptr);
    string unique_name = input_or_output_info_ptr->GetUniqueName();
    format_res[unique_name] = FE_ORIGIN_FORMAT_VECTOR;
    sub_format_res[unique_name] = vector<uint32_t>(1U, DEFAULT_SUB_FORMAT);
    index++;
  }
  origin_info_ptr->input_formats = input_formats;
  origin_info_ptr->input_dtypes = input_dtypes;
  origin_info_ptr->input_shapes = input_shapes;
  origin_info_ptr->output_formats = output_formats;
  origin_info_ptr->output_shapes = output_shapes;
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::UpdateFormatMap(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                     const FormatProccessResult &format_proccess_res,
                                                     const IndexNameMap &index_name_map,
                                                     const IndexNameMap &output_index_map,
                                                     map<string, vector<ge::Format>> &format_res,
                                                     map<string, vector<uint32_t>> &sub_format_res) const {
  if (UpdateFormatMap(true, op_kernel_info_ptr, format_proccess_res, index_name_map, format_res,
      sub_format_res) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][UptFmtMap] Update input formats failed.");
    return FAILED;
  }
  if (UpdateFormatMap(false, op_kernel_info_ptr, format_proccess_res, output_index_map, format_res,
      sub_format_res) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][UptFmtMap] Update output formats failed");
    return FAILED;
  }
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::UpdateFormatMap(const bool &is_input, const OpKernelInfoPtr &op_kernel_info_ptr,
                                                     const FormatProccessResult &format_proccess_res,
                                                     const IndexNameMap &index_map,
                                                     map<string, vector<ge::Format>> &format_res,
                                                     map<string, vector<uint32_t>> &sub_format_res) const {
  const vector<vector<ge::Format>> formats = is_input ? format_proccess_res.input_format_res :
       format_proccess_res.output_format_res;
  const vector<uint32_t> sub_formats = is_input ? format_proccess_res.input_subformat_res :
      format_proccess_res.output_subformat_res;
  if (formats.empty()) {
    FE_LOGD("[GraphOpt][GenFmt][UptFmtMap] Support format is empty.");
    return SUCCESS;
  }
  for (const auto &format_vec : formats) {
    for (size_t j = 0; j != format_vec.size(); ++j) {
      InputOrOutputInfoPtr input_or_output_info_ptr =
          GetInputOrOutputUniqueName(is_input, index_map, j, op_kernel_info_ptr);
      FE_CHECK_NOTNULL(input_or_output_info_ptr);
      const string &unique_name = input_or_output_info_ptr->GetUniqueName();
      if (input_or_output_info_ptr->GetParamType() == DYNAMIC) {
        if (j == 0) {
          format_res[unique_name].emplace_back(format_vec[j]);
        }
      } else {
        format_res[unique_name].emplace_back(format_vec[j]);
      }
    }
  }
  for (size_t i = 0; i < formats[0].size(); i++) {
    InputOrOutputInfoPtr input_or_output_info_ptr =
    GetInputOrOutputUniqueName(is_input, index_map, i, op_kernel_info_ptr);
    FE_CHECK_NOTNULL(input_or_output_info_ptr);
    const string &unique_name = input_or_output_info_ptr->GetUniqueName();
    for (const auto &sub_format : sub_formats) {
      sub_format_res[unique_name].emplace_back(sub_format);
    }
  }
  return SUCCESS;
}

InputOrOutputInfoPtr FormatDtypeOpBuiltinSelector::GetInputOrOutputUniqueName(
    const bool &is_input, const IndexNameMap &index_name_map, const size_t &index,
    const OpKernelInfoPtr &op_kernel_info_ptr) const {
  InputOrOutputInfoPtr input_or_output_info_ptr = nullptr;
  auto iter = index_name_map.find(index);
  string prefix = is_input ? "input" : "output";
  if (iter == index_name_map.end()) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][GetInOutNm] the prefix[%s] index [%zu] is not found from the op store.",
                    prefix.c_str(), index);
    return input_or_output_info_ptr;
  }
  if (is_input) {
    if (op_kernel_info_ptr->GetInputInfoByName(iter->second, input_or_output_info_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][GenFmt][GetInOutNm] the prefix[%s] index[%zu] is not found from the op store.",
                      prefix.c_str(), index);
      return input_or_output_info_ptr;
    }
  } else {
    if (op_kernel_info_ptr->GetOutputInfoByName(iter->second, input_or_output_info_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][GenFmt][GetInOutNm] the prefix[%s] index[%zu] is not found from the op store.",
                      prefix.c_str(), index);
      return input_or_output_info_ptr;
    }
  }
  return input_or_output_info_ptr;
}

Status FormatDtypeOpBuiltinSelector::GetReduceNewFormatDType(const OpKernelInfoPtr &op_kernel_info_ptr,
                                                             const ge::OpDescPtr &op_desc, const string input_key,
                                                             const string output_key,
                                                             FormatDtypeInfo &format_dtype_res) const {
  if (op_kernel_info_ptr->GetOpPattern() == OP_PATTERN_REDUCE) {
    FE_LOGD("Op[name=%s,type=%s]: is reduce op, begin to check dtype supported", op_desc->GetName().c_str(),
            op_desc->GetType().c_str());
    if (format_dtype_res.format_map.size() != kNumTwo || format_dtype_res.data_type_map.size() != kNumTwo) {
      FE_LOGD("Op[name=%s,type=%s]: format size:%zu, datatype size:%zu, not 2.", op_desc->GetName().c_str(),
              op_desc->GetType().c_str(), format_dtype_res.format_map.size(), format_dtype_res.data_type_map.size());
      return SUCCESS;
    }

    vector<ge::Format> input_formats = format_dtype_res.format_map[input_key];
    vector<ge::Format> output_formats = format_dtype_res.format_map[output_key];
    vector<ge::DataType> input_data_types = format_dtype_res.data_type_map[input_key];
    vector<ge::DataType> output_data_types = format_dtype_res.data_type_map[output_key];

    vector<ge::Format> input_new_formats;
    vector<ge::Format> output_new_formats;
    vector<uint32_t> input_new_sub_formats;
    vector<uint32_t> output_new_sub_formats;
    vector<ge::DataType> input_new_data_types;
    vector<ge::DataType> output_new_data_types;

    for (size_t i = 0; i < input_formats.size(); i++) {
      bool is_out_original_format = (find(FE_ORIGIN_FORMAT_VECTOR.begin(), FE_ORIGIN_FORMAT_VECTOR.end(),
                                          output_formats[i]) != FE_ORIGIN_FORMAT_VECTOR.end());
      if (input_formats[i] == ge::FORMAT_NC1HWC0 && is_out_original_format && input_data_types[i] != ge::DT_FLOAT16) {
        FE_LOGD("Op[%s]: input [%zu]'s format is NC1HWC0, output format is ND, and dtype is not float16, clear it.",
                op_desc->GetName().c_str(), i);
        continue;
      }
      /* when format is 5HD, the size of reduce axis and size of not redcued axis
       * must less than ub size. */
      if (input_formats[i] == ge::FORMAT_NC1HWC0 &&
        !CheckUBSizeEnable(*op_desc, input_formats[i], input_data_types[i])) {
        FE_LOGD("Op[%s]: input %zu's format is NC1HWC0, total size needed is greater than ub block size, clear it.",
                op_desc->GetName().c_str(), i);
        continue;
      }

      input_new_formats.push_back(input_formats[i]);
      input_new_sub_formats.push_back(SUPPORT_ALL_SUB_FORMAT);
      input_new_data_types.push_back(input_data_types[i]);
      output_new_formats.push_back(output_formats[i]);
      output_new_sub_formats.push_back(SUPPORT_ALL_SUB_FORMAT);
      output_new_data_types.push_back(output_data_types[i]);
    }

    format_dtype_res.format_map[input_key].swap(input_new_formats);
    format_dtype_res.format_map[output_key].swap(output_new_formats);
    format_dtype_res.sub_format_map[input_key].swap(input_new_sub_formats);
    format_dtype_res.sub_format_map[output_key].swap(output_new_sub_formats);
    format_dtype_res.data_type_map[input_key].swap(input_new_data_types);
    format_dtype_res.data_type_map[output_key].swap(output_new_data_types);
  }
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetReduceNewFormatDTypeVec(const OpKernelInfoPtr &op_kernel_info_ptr,
    const ge::NodePtr &node, const string &input_or_out_key, vector<ge::DataType> &data_types,
    vector<ge::Format> &formats, vector<uint32_t> &sub_formats) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  if (op_kernel_info_ptr->GetOpPattern() == OP_PATTERN_REDUCE) {
    FormatDtypeInfo format_dtype_res;
    bool is_dynamic_check = IsOpDynamicImpl(op_desc);
    if (GetSupportFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, format_dtype_res) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][GenFmt][GetRdcNewFmtDtyVec] Op[name=%s,type=%s]: Failed to GetSupportFormatDtype.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    if (!format_dtype_res.data_type_map.empty() &&
        (format_dtype_res.data_type_map.find(input_or_out_key) != format_dtype_res.data_type_map.end())) {
      data_types = std::move(format_dtype_res.data_type_map[input_or_out_key]);
    }
    if (!format_dtype_res.format_map.empty() &&
        (format_dtype_res.format_map.find(input_or_out_key) != format_dtype_res.format_map.end())) {
      formats = std::move(format_dtype_res.format_map[input_or_out_key]);
    }
    if (!format_dtype_res.sub_format_map.empty() &&
        (format_dtype_res.sub_format_map.find(input_or_out_key) != format_dtype_res.sub_format_map.end())) {
      sub_formats = std::move(format_dtype_res.sub_format_map[input_or_out_key]);
    }
  }
  return SUCCESS;
}

Status FormatDtypeOpBuiltinSelector::GetBroadcastNewSubformatVec(const OpKernelInfoPtr &op_kernel_info_ptr,
    const ge::NodePtr &node, const string &input_or_out_key, const uint32_t &sub_format, vector<uint32_t> &sub_formats) {
  auto op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  FormatDtypeInfo format_dtype_res;
  HeavyFormatInfo heavy_format_info;
  bool is_dynamic_check = IsOpDynamicImpl(op_desc);
  (void)GetSubFormatFromOpDescByKey(op_desc, input_or_out_key, sub_formats);

  if (sub_format == GROUPS_DEFAULT_VALUE ||
      std::find(sub_formats.begin(), sub_formats.end(), sub_format) != sub_formats.end()) {
    FE_LOGD("[GraphOpt][GenFmt][GetBrdcstNewSubfmtVec] No need to get new subformat for op_desc[%s, %s], sub_format %u.",
            op_desc->GetName().c_str(), op_desc->GetType().c_str(), sub_format);
    return SUCCESS;
  }
  if (GetDynamicFormatDtype(op_kernel_info_ptr, node, is_dynamic_check, heavy_format_info,
      format_dtype_res, sub_format) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][GetBrdcstNewSubfmtVec] Op[name=%s,type=%s]: failed to GetDynamicFormatDtype.",
                    op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  if (!format_dtype_res.sub_format_map.empty() && format_dtype_res.sub_format_map.find(input_or_out_key) !=
      format_dtype_res.sub_format_map.end()) {
    sub_formats = format_dtype_res.sub_format_map[input_or_out_key];
    (void)op_desc->SetExtAttr(EXT_DYNAMIC_SUB_FORMAT, format_dtype_res.sub_format_map);
  }
  return SUCCESS;
}

bool FormatDtypeOpBuiltinSelector::CheckUBSizeEnable(const ge::OpDesc &op_desc, const ge::Format &check_format,
                                                     const ge::DataType &check_dtype) const {
  vector<int64_t> axis_values;
  (void)ge::AttrUtils::GetListInt(op_desc, AXES_ATTR_NAME, axis_values);
  if (axis_values.empty()) {
    return true;
  }
  // 1. keep_dims must be false
  bool keep_dim = false;
  if (ge::AttrUtils::GetBool(op_desc, KEEP_DIMS, keep_dim) && keep_dim) {
    FE_LOGD("Op[name=%s,type=%s]: the attribute keep_dims is true.", op_desc.GetName().c_str(),
            op_desc.GetName().c_str());
    return true;
  }

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][GenFmt][ChkUbSizeEn] Failed to get platform info.");
    return false;
  }
  uint64_t ub_size = platform_info.ai_core_spec.ub_size;
  FE_LOGD("UB size is %lu.", ub_size);

  int data_size = ge::GetSizeByDataType(check_dtype);
  FE_LOGD("The dtype size of op[name:%s,type:%s] is %d.", op_desc.GetName().c_str(),
          op_desc.GetType().c_str(), data_size);

  for (const auto &input_desc : op_desc.GetAllInputsDescPtr()) {
    ge::GeShape origin_shape = input_desc->GetOriginShape();
    ge::Format ori_format = input_desc->GetOriginFormat();
    ge::GeShape new_shape;
    ShapeAndFormat shape_and_format_info = {origin_shape, new_shape, ori_format, check_format, check_dtype};
    Status ret = GetShapeAccordingToFormat(shape_and_format_info);
    if (ret != SUCCESS) {
      FE_LOGW("Get new shape not successfully, old format is %u, new format is %u", input_desc->GetOriginFormat(),
              check_format);
      return false;
    }
    // get new axis info
    vector<int64_t> axis_new_value;
    if (AxisNameUtil::GetNewAxisAttributeValue(op_desc, ori_format, check_format, origin_shape, axis_new_value) !=
        SUCCESS) {
      FE_LOGW("Get reduce op [%s] new axis info failed!", op_desc.GetName().c_str());
      return false;
    }

    uint64_t total_size = CalcTotalSize(axis_new_value, new_shape, op_desc, data_size);
    if (total_size >= ub_size) {
      return false;
    }
  }
  return true;
}
}  // namespace fe
