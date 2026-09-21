/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_FORMAT_DTYPE_OP_BUILTIN_SELECTOR_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_FORMAT_DTYPE_OP_BUILTIN_SELECTOR_H_

#include <map>
#include <string>
#include <vector>
#include "adapter/adapter_itf/op_store_adapter.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/util/op_info_util.h"
#include "common/string_utils.h"
#include "format_selector/builtin/process/format_process_registry.h"
#include "format_selector/common/format_dtype_selector_base.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"

namespace fe {
using IndexNameMap = map<uint32_t, string>;

class FormatDtypeOpBuiltinSelector : public FormatDtypeSelectorBase {
 public:
  FormatDtypeOpBuiltinSelector();
  ~FormatDtypeOpBuiltinSelector() override;

  Status GetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                               const bool &is_dynamic_check, FormatDtypeInfo &format_dtype_res) override;

  Status GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                             const bool &is_dynamic_check, map<string, vector<ge::Format>> &format_map) override;

  Status GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                           const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                           vector<ge::Format> &result) override;

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
  Status GetUnionFormatDtype(const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::OpDescPtr &op_desc,
                             vector<ge::Format> &new_formats, vector<ge::DataType> &new_data_types);
  Status InitOriginInfo(const ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                        const IndexNameMap &input_index_map, const IndexNameMap &output_index_map,
                        OriginInfoPtr origin_info_ptr, map<string, vector<ge::Format>> &format_res,
                        map<string, vector<uint32_t>> &sub_format_res) const;

  Status UpdateFormatMap(const OpKernelInfoPtr &op_kernel_info_ptr, const FormatProccessResult &format_proccess_res,
                         const IndexNameMap &index_name_map, const IndexNameMap &output_index_map,
                         map<string, vector<ge::Format>> &format_res,
                         map<string, vector<uint32_t>> &sub_format_res) const;

  Status UpdateFormatMap(const bool &is_input, const OpKernelInfoPtr &op_kernel_info_ptr,
                         const FormatProccessResult &format_proccess_res, const IndexNameMap &index_map,
                         map<string, vector<ge::Format>> &format_res,
                         map<string, vector<uint32_t>> &sub_format_res) const;

  InputOrOutputInfoPtr GetInputOrOutputUniqueName(const bool &is_input, const IndexNameMap &index_name_map,
                                                  const size_t &index, const OpKernelInfoPtr &op_kernel_info_ptr) const;

  Status GetReduceNewFormatDType(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::OpDescPtr &op_desc,
                                 const string input_key, const string output_key,
                                 FormatDtypeInfo &format_dtype_res) const;

  /* if reduce size greater than ub block size, can not support the format */
  bool CheckUBSizeEnable(const ge::OpDesc &op_desc, const ge::Format &check_format,
                         const ge::DataType &check_dtype) const;

  Status GetReduceNewFormatDTypeVec(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                    const string &input_or_out_key, vector<ge::DataType> &data_types,
                                    vector<ge::Format> &formats, vector<uint32_t> &sub_formats);

  Status GetBroadcastNewSubformatVec(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                     const string &input_or_out_key, const uint32_t &sub_format,
                                     vector<uint32_t> &sub_formats);
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_FORMAT_DTYPE_OP_BUILTIN_SELECTOR_H_
