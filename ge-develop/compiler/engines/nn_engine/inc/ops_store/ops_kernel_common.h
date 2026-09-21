/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_OPS_KERNEL_COMMON_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_OPS_KERNEL_COMMON_H_

#include "ops_store/op_kernel_info.h"
#include "common/util/op_info_util.h"

namespace fe {
using IndexNameMap = std::map<uint32_t, std::string>;

const std::map<OpParamType, std::string> kParamTypeStrMap {
              {DYNAMIC, "dynamic"}, {OPTIONAL, "optional"},
              {REQUIRED, "required"}, {RESERVED, "reserved"}};

enum InputOrOutputIndex { INPUT_INDEX = 0, OUTPUT_INDEX = 1, INPUT_OUTPUT_INDEX_BOTTOM = 2 };
struct UnSupportedReason {
  std::string reason;
  uint64_t reason_id = static_cast<uint64_t>(OpNotSupportedReasonID::EN_REASON_ID_RESERVED);
};

bool IsHeavyOp(ge::OpDescPtr op_desc_ptr);
bool IsHeavyOp(ge::NodePtr node_ptr);

bool IsAllowNzMatmul(const ge::NodePtr &node_ptr);

bool IsDynamicInputOrOutput(const InputOrOutputInfoPtr& input_output_info);

std::string GetOpParamTypeStr(OpParamType op_param_type);

/*
 *  @ingroup fe
 *  @brief   get input index and name in op kernel info map
 *  @param   [in]  op_desc
 *  @param   [in]  op_kernel_info
 *  @param   [out] input_map
 *  @return  SUCCESS or FAILED
 */
Status GetInputIndexNameMap(const ge::OpDesc &op_desc, const OpKernelInfo &op_kernel_info, IndexNameMap &input_map);

/*
 *  @ingroup fe
 *  @brief   get output index and name in op kernel info map
 *  @param   [in]  op_desc
 *  @param   [in]  op_kernel_info
 *  @param   [out] output_map
 *  @return  SUCCESS or FAILED
 */
Status GetOutputIndexNameMap(const ge::OpDesc &op_desc, const OpKernelInfo &op_kernel_info, IndexNameMap &output_map);

void CheckSpecialCases(const std::vector<InputOrOutputInfoPtr>& input_or_output_info, IndexNameMap& index_name_map,
                       uint32_t index, uint32_t op_desc_input_or_output_size, bool& has_found);

bool CheckInputSubString(const std::string& op_desc_input_name, const std::string& info_input_name);

bool CheckVirtualSoftsyncOp(const OpKernelInfoPtr &op_kernel_ptr, const ge::OpDescPtr &op_desc_ptr);

/* Get All input and output kernel info */
Status GetAllInputAndOutputKernelInfo(const OpKernelInfoPtr &op_kernel_info_ptr, const ge::NodePtr &current_node,
                                      const std::vector<IndexNameMap> &tensor_map,
                                      std::vector<std::vector<InputOrOutputInfoPtr>> &input_and_output_kernel);

Status GetInputOutputNameMap(const ge::OpDesc &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                             IndexNameMap &input_map, IndexNameMap &output_map);
Status GetOutputNameMap(const ge::OpDesc& op_desc, const OpKernelInfoPtr& op_kernel_info_ptr,
                        IndexNameMap& output_map);
bool GetInputOutputNameMap(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                           IndexNameMap &input_map, IndexNameMap &output_map,
                           UnSupportedReason &reason);

void GenerateOpSupportInfo(const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_dynamic_impl,
                           const std::map<string, vector<ge::Format>> &format_map,
                           const std::map<string, vector<ge::DataType>> &datatype_map,
                           std::string &op_support_info);
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_OP_INFO_COMMON_H_
