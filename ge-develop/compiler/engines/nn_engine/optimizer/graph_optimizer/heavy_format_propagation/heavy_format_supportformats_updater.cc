/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/heavy_format_propagation/heavy_format_supportformats_updater.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"

namespace fe {
HeavyFormatSupportFormatsUpdater::HeavyFormatSupportFormatsUpdater(FormatDtypeQuerierPtr format_dtype_querier_ptr,
                                                                   FormatDtypeSetterPtr format_dtype_setter_ptr)
    : format_dtype_querier_ptr_(format_dtype_querier_ptr), format_dtype_setter_ptr_(format_dtype_setter_ptr) {}

HeavyFormatSupportFormatsUpdater::~HeavyFormatSupportFormatsUpdater() {}

/* Before doing op_select_format, we need to update sub-format.
 * Because the purpose of second time op_select_format is to get
 * the supporting formats for specific sub-format. If we use the
 * original sub-format(0), there will be no change when we do the
 * second time query. */
void HeavyFormatSupportFormatsUpdater::UpdateSubFormatForTensors(const ge::OpDescPtr &op_desc_ptr,
                                                                 const HeavyFormatInfo& heavy_format_info) const {
  for (size_t i = 0; i < op_desc_ptr->GetAllInputsSize(); i++) {
    auto input = op_desc_ptr->MutableInputDesc(i);
    if (input != nullptr) {
      int32_t ori_format = ge::GetPrimaryFormat(static_cast<int32_t>(input->GetFormat()));
      int32_t c0_bit_val = GetC0BitByDataType(input->GetDataType());
      input->SetFormat(static_cast<ge::Format>(ge::GetFormatFromSubAndC0(ori_format, heavy_format_info.sub_format,
                                                                         c0_bit_val)));
    }
  }

  for (size_t i = 0; i < op_desc_ptr->GetAllOutputsDescSize(); i++) {
    auto output = op_desc_ptr->MutableOutputDesc(i);
    if (output != nullptr) {
      int32_t prm_format = ge::GetPrimaryFormat(static_cast<int32_t>(output->GetFormat()));
      int32_t c0_bit_val = GetC0BitByDataType(output->GetDataType());
      output->SetFormat(static_cast<ge::Format>(ge::GetFormatFromSubAndC0(prm_format, heavy_format_info.sub_format,
                                                                          c0_bit_val)));
    }
  }
}

Status HeavyFormatSupportFormatsUpdater::UpdateSupportFormats(const ge::NodePtr& node_ptr,
                                                              const OpKernelInfoPtr& op_kernel_info_ptr,
                                                              const std::vector<IndexNameMap>& tensor_map,
                                                              const HeavyFormatInfo& heavy_format_info) {
  auto op_desc_ptr = node_ptr->GetOpDesc();
  auto op_name = op_desc_ptr->GetName();
  auto op_type = op_desc_ptr->GetType();
  // 1. If the heavy_format is not fz/fz_3d, or the op is not dynamic_format and op.patter=Broadcast,
  // no need to update the support_formats
  if (!IsFzRelaFormat(heavy_format_info) || !IsSelectFormatOrBroadcast(op_desc_ptr, op_kernel_info_ptr)) {
    return SUCCESS;
  }

  // 2. get the support formats
  vector<vector<InputOrOutputInfoPtr>> input_and_output_kernel;
  input_and_output_kernel.emplace_back();
  input_and_output_kernel.emplace_back();
  Status ret = GetAllInputAndOutputKernelInfo(op_kernel_info_ptr, node_ptr, tensor_map, input_and_output_kernel);
  if (ret != SUCCESS) {
    return FAILED;
  }
  if (input_and_output_kernel.size() != INPUT_OUTPUT_INDEX_BOTTOM) {
    FE_LOGW("Size of input kernel vector %zu is not correct for node %s.", input_and_output_kernel.size(),
            node_ptr->GetName().c_str());
    return FAILED;
  }

  std::vector<InputOrOutputInfoPtr> input_or_output_info_vec =
      heavy_format_info.is_input ? input_and_output_kernel[INPUT_INDEX] : input_and_output_kernel[OUTPUT_INDEX];
  InputOrOutputInfoPtr input_or_output_info = input_or_output_info_vec.at(heavy_format_info.anchor_index);
  vector<ge::Format> kernel_formats;
  if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, input_or_output_info, node_ptr,
                                                   kernel_formats) != SUCCESS) {
    return FAILED;
  }

  // 3. update support formats and dtypes
  auto propaga_heavy_format = static_cast<ge::Format>(
      ge::GetFormatFromSub(heavy_format_info.expected_heavy_format, heavy_format_info.sub_format));
  if (!NeedUpdateSupportFormats(op_desc_ptr, heavy_format_info, kernel_formats, propaga_heavy_format)) {
    return SUCCESS;
  }

  FE_LOGD("Op[name=%s, type=%s]: need to update supported formats, propaga_heavy_format=%s for %s.", op_name.c_str(),
          op_type.c_str(), FormatToStr(propaga_heavy_format).c_str(), input_or_output_info->GetUniqueName().c_str());
  UpdateSubFormatForTensors(op_desc_ptr, heavy_format_info);

  ret = format_dtype_setter_ptr_->SetSupportFormatDtypeByNode(node_ptr, heavy_format_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SptFmtUpDtr][UptSptFmt] Op[name=%s, type=%s]: failed to set the supported formats",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }
  (void)ge::AttrUtils::SetStr(op_desc_ptr, ATTR_NAME_FE_PROPAGAT_HEAVY_FORMAT,
                              ge::TypeUtils::FormatToSerialString(propaga_heavy_format));

  return SUCCESS;
}

bool HeavyFormatSupportFormatsUpdater::IsFzRelaFormat(const HeavyFormatInfo& heavy_format_info) const {
  return std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(),
                   heavy_format_info.expected_heavy_format) != FE_GROUP_RELA_FORMAT_VECTOR.end();
}

bool HeavyFormatSupportFormatsUpdater::IsSelectFormatOrBroadcast(const ge::OpDescPtr& op_desc_ptr,
                                                                 const OpKernelInfoPtr& op_kernel_info_ptr) {
  bool is_dynamic_check = IsOpDynamicImpl(op_desc_ptr);
  bool is_op_pattern_broadcast =
      format_dtype_setter_ptr_->IsOpPatternBroadcast(op_kernel_info_ptr, is_dynamic_check);
  bool is_op_select_format =
      format_dtype_setter_ptr_->IsSelectFormat(op_kernel_info_ptr, is_dynamic_check);
  return is_op_pattern_broadcast || is_op_select_format;
}

bool HeavyFormatSupportFormatsUpdater::NeedUpdateSupportFormats(const ge::OpDescPtr& op_desc_ptr,
                                                                const HeavyFormatInfo& heavy_format_info,
                                                                const vector<ge::Format>& kernel_formats,
                                                                ge::Format propaga_heavy_format) {
  if (!ge::AttrUtils::HasAttr(op_desc_ptr, ATTR_NAME_FE_PROPAGAT_HEAVY_FORMAT)) {
    if (std::find(kernel_formats.begin(), kernel_formats.end(), heavy_format_info.expected_heavy_format) ==
            kernel_formats.end() ||
        heavy_format_info.sub_format <= 1) {
      return false;
    }
    FE_LOGD("Op[name=%s,type=%s]: no attr %s.", op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
            ATTR_NAME_FE_PROPAGAT_HEAVY_FORMAT.c_str());
    return true;
  }

  string update_format_str;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, ATTR_NAME_FE_PROPAGAT_HEAVY_FORMAT, update_format_str);
  auto propaga_format_str = ge::TypeUtils::FormatToSerialString(propaga_heavy_format);
  if (update_format_str == propaga_format_str) {
    FE_LOGD("Op[name=%s,type=%s]: the attribute %s %s is equal to propaga_format %s.", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str(), ATTR_NAME_FE_PROPAGAT_HEAVY_FORMAT.c_str(), update_format_str.c_str(),
            propaga_format_str.c_str());
    return false;
  }
  return true;
}
}  // namespace fe
