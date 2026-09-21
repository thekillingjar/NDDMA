/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_COMMON_UPDATE_TENSOR_DESC_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_COMMON_UPDATE_TENSOR_DESC_H_

#include "common/op_info_common.h"

namespace fe {
void CountNdInput(const UpdateInfo &update_info, ge::Format &ori_format,
                  ge::NodePtr &node, size_t &count_input_is_nd);

ge::GeShape GetNewShape(const ge::OpDescPtr& op_desc_ptr, const ge::GeShape& old_shape,
                        const ge::Format& old_format, const ge::Format& new_format,
                        const int64_t& op_imply_type, const ge::DataType& current_data_type,
                        const int64_t& group);

Status PadNDToOtherFormat(const UpdateInfo &update_info, const ge::Format &op_kernel_format, ge::Format &ori_format);

int64_t GetReshapeTypeMask(const UpdateInfo &update_info, const ge::Format &ori_format,
                           const ge::Format &primary_format);

Status UpdateNewShapeAndFormat(const UpdateInfo& update_info, ge::Format op_kernel_format, const int64_t& group,
                             const ge::GeShape& original_shape, const ge::GeShape& new_shape,
                             const std::string& op_type, const std::string& op_name);

Status UpdateNewShape(const UpdateInfo& update_info, ge::Format op_kernel_format, ge::DataType op_kernel_dtype,
                      int64_t group, int64_t op_imply_type_input);

Status CalcNewShapeAndUpdate(const UpdateInfo& update_info, ge::Format op_kernel_format, ge::DataType op_kernel_dtype);
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_COMMON_UPDATE_TENSOR_DESC_H_
