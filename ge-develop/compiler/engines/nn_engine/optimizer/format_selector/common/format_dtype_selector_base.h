/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_COMMON_FORMAT_DTYPE_SELECTOR_BASE_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_COMMON_FORMAT_DTYPE_SELECTOR_BASE_H_

#include <map>
#include <string>
#include <vector>
#include "common/fe_inner_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "common/string_utils.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
struct FormatDtypeInfo {
  std::map<std::string, vector<ge::Format>> format_map;
  std::map<std::string, vector<uint32_t>> sub_format_map;
  std::map<std::string, vector<ge::DataType>> data_type_map;
  std::vector<bool> fallback_flags;
};
class FormatDtypeSelectorBase {
 public:
  FormatDtypeSelectorBase();
  virtual ~FormatDtypeSelectorBase();

  /**
   * Get the support formats and dtyps from the op_desc, if failed, get the
   * dynamic formats and dtyps. Used for CheckSubStoreSupported.
   * @param op_kernel_info_ptr op kernel info
   * @param op_desc op desc
   * @param format_map formats result
   * @param data_type_map dtypes result
   * @return SUCCESS or FAILED
   */
  virtual Status GetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                       const bool &is_dynamic_check, FormatDtypeInfo &format_dtype_info) = 0;

  virtual Status GetAllSupportFormat(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                     const bool &is_dynamic_check, map<string, vector<ge::Format>> &format_map) = 0;

  /**
   * Get the support formats from the op_desc by the input_or_output_info.
   * Used for OpFormatDtypeJudge and HeavyFormatDistribution.
   * @param input_or_output_info_ptr input_or_output_info
   * @param node node ptr
   * @param result formats
   * @return SUCCESS or FAILED
   */
  virtual Status GetSupportFormats(const OpKernelInfoPtr &op_kernel_info_ptr,
                                   const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                                   vector<ge::Format> &result) = 0;

  /**
   * Get the support subformats from the op_desc by the input_or_output_info.
   * Used for OpFormatDtypeJudge and HeavyFormatDistribution.
   * @param input_or_output_info_ptr input_or_output_info
   * @param node node ptr
   * @param result subformats
   * @return SUCCESS or FAILED
   */
  virtual Status GetSupportFormatSubFormat(const OpKernelInfoPtr &op_kernel_info_ptr,
                                           const InputOrOutputInfoPtr &input_or_output_info_ptr,
                                           const ge::NodePtr &node, vector<ge::Format> &format_res,
                                           vector<uint32_t> &sub_format_res,
                                           uint32_t sub_format) = 0;

  /**
   * Get the support dtypes from the op_desc by the input_or_output_info.
   * Used for OpFormatDtypeJudge and HeavyFormatDistribution.
   * @param input_or_output_info_ptr input_or_output_info
   * @param node nodeptr
   * @param result d_types
   * @return SUCCESS or FAILED
   */
  virtual Status GetSupportDataTypes(const OpKernelInfoPtr &op_kernel_info_ptr,
                                     const InputOrOutputInfoPtr &input_or_output_info_ptr, const ge::NodePtr &node,
                                     vector<ge::DataType> &result) = 0;

  /**
   * Get the dynmaic formats and dtypes by infering or caliing the
   * SelectTbeOpFormat function of TeFusion.
   * @param op_kernel_info_ptr op kernel info
   * @param NodePtr
   * @param format_map formats result
   * @param data_type_map dtypes result
   * @return SUCCESS or FAILED
   */
  virtual Status GetDynamicFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                       const bool &is_dynamic_check, const HeavyFormatInfo &heavy_format_info,
                                       FormatDtypeInfo &dtype_format_info, uint32_t sub_format) = 0;

  virtual Status GetFallbackFlags(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &node,
                                  bool is_dynamic_check, std::vector<bool> &fallback_res) = 0;

  /**
   * Set the support formats and dtypes for the op_desc.
   * @param op_kernel_info_ptr op kernel info
   * @param op_desc op desc
   * @return SUCCESS or FAILED
   */
  Status SetSupportFormatDtype(const OpKernelInfoPtr &op_kernel_info_ptr, const HeavyFormatInfo &heavy_format_info,
                               const ge::NodePtr &node, const bool &is_dynamic_check);

 protected:
  Status GetAllFormatsFromOpDesc(const ge::OpDescPtr &op_desc, map<string, vector<ge::Format>> &result);
  Status GetAllFallbackFromOpDesc(const ge::OpDescPtr &op_desc, vector<bool> &fallback_flags);
  Status GetAllSubFormatsFromOpDesc(const ge::OpDescPtr &op_desc, map<string, vector<uint32_t>> &result);
  Status GetAllDataTypesFromOpDesc(const ge::OpDescPtr &op_desc, map<string, vector<ge::DataType>> &result);
  Status GetFormatFromOpDescByKey(const ge::OpDescPtr &op_desc, const string &key, vector<ge::Format> &result);
  Status GetSubFormatFromOpDescByKey(const ge::OpDescPtr &op_desc, const string &key, vector<uint32_t> &result);
  Status GetDataTypeFromOpDescByKey(const ge::OpDescPtr &op_desc, const string &key, vector<ge::DataType> &result);
  const string EXT_DYNAMIC_SUB_FORMAT = "ext_dynamic_sub_format";

 private:
  /**
   * Save the format_map and data_type_map to the ext attribute of the op_desc.
   * @param format_map format map
   * @param data_type_map data_type map
   * @param op_desc  op desc
   * @return SUCCESS or FAILED
   */
  Status SaveDynamicFormatDtype(const FormatDtypeInfo &format_dtype_info,
                                const ge::OpDescPtr &op_desc) const;

  const string EXT_DYNAMIC_FORMAT = "ext_dynamic_format";
  const string EXT_DYNAMIC_DATATYPE = "ext_dynamic_datatype";
  const string EXT_DYNAMIC_FALLBACK = "ext_dynamic_fallback";
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_COMMON_FORMAT_DTYPE_SELECTOR_BASE_H_
