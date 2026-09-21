/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_OP_CUSTOMIZE_FORMAT_DTYPE_OP_CUSTOMIZE_SELECTOR_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_OP_CUSTOMIZE_FORMAT_DTYPE_OP_CUSTOMIZE_SELECTOR_H_

#include "adapter/adapter_itf/op_store_adapter.h"
#include "format_selector/common/format_dtype_selector_base.h"

namespace fe {
class FormatDtypeOpCustomizeSelector : public FormatDtypeSelectorBase {
 public:
  explicit FormatDtypeOpCustomizeSelector(const std::string &engine_name);
  ~FormatDtypeOpCustomizeSelector() override;

  Status GetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                               const bool &is_dynamic_check, FormatDtypeInfo &format_dtype_info) override;

  Status GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                           const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                           vector<ge::Format> &result) override;

  Status GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                             const bool &is_dynamic_check, map<string, vector<ge::Format>> &format_map) override;

  Status GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                   const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                                   vector<ge::Format> &format_res, vector<uint32_t> &sub_format_res,
                                   uint32_t sub_format = 1) override;

  Status GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                             const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                             vector<ge::DataType> &result) override;

  Status GetDynamicFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                               const bool &is_dynamic_check, const HeavyFormatInfo &heavy_format_info,
                               FormatDtypeInfo &format_dtype_info, uint32_t sub_format = 1) override;

  Status GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                          bool is_dynamic_check, std::vector<bool> &fallback_res) override;
 private:
  Status ParseJsonStr(const ge::NodePtr &node, const string &json_str, const bool &is_dynamic_check,
                      FormatDtypeInfo &format_dtype_info);
  Status ConvertFormatDtype(const string &format_vec_str, const string &sub_format_vec_str,
                            const string &data_type_vec_str, uint32_t &format_size_of_first_input_or_output,
                            vector<ge::Format> &format_vec, vector<uint32_t> &sub_format_vec,
                            vector<ge::DataType> &data_type_vec);

  Status UpdateDTypeAndFormat(set<uint32_t> &remain_index_set, std::map<std::string, vector<ge::Format>> &format_map,
                              std::map<std::string, vector<uint32_t>> &sub_format_map,
                              std::map<std::string, vector<ge::DataType>> &data_type_map) const;

  Status ParseFallbackJsonInfo(const ge::NodePtr &node, bool is_dynamic_check, const nlohmann::json &fallback_json,
                               FormatDtypeInfo &format_dtype_info) const;

  std::string engine_name_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_OP_CUSTOMIZE_FORMAT_DTYPE_OP_CUSTOMIZE_SELECTOR_H_
