/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_BUILDER_H_
#define DVPP_ENGINE_COMMON_DVPP_BUILDER_H_

#include "util/dvpp_error_code.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace dvpp {
class DvppBuilder {
 public:
  /**
   * @brief Construct
   */
  DvppBuilder() = default;

  /**
   * @brief Destructor
   */
  virtual ~DvppBuilder() = default;

  // Copy constructor prohibited
  DvppBuilder(const DvppBuilder& dvpp_builder) = delete;

  // Move constructor prohibited
  DvppBuilder(const DvppBuilder&& dvpp_builder) = delete;

  // Copy assignment prohibited
  DvppBuilder& operator=(const DvppBuilder& dvpp_builder) = delete;

  // Move assignment prohibited
  DvppBuilder& operator=(DvppBuilder&& dvpp_builder) = delete;

  /**
   * @brief calculate running size of op
   *        then GE will alloc the memory from runtime
   * @param node node info, return task memory size in node attr
   * @return status whether success
   */
  DvppErrorCode CalcOpRunningParam(ge::Node& node) const;

  /**
   * @brief make the task info details
   *        then GE will alloc the address and send task by runtime
   * @param node node info
   * @param context run context from GE
   * @param tasks make the task return to GE
   * @return status whether success
   */
  virtual DvppErrorCode GenerateTask(const ge::Node& node,
      ge::RunContext& context, std::vector<domi::TaskDef>& tasks) = 0;

 private:
  DvppErrorCode SetOutPutMemorySize(ge::OpDescPtr& op_desc_ptr) const;

  /**
   * @brief calculate output memory size
   * @param output_desc output desc
   * @param output_memory_size output memory size
   * @return status whether success
   */
  DvppErrorCode CalcOutputMemorySize(
      ge::GeTensorDesc& output_desc, int64_t& output_memory_size) const;

  DvppErrorCode CalcTotalSizeByDimsAndType(const std::vector<int64_t>& dims,
      const ge::DataType& data_type, int64_t& total_size) const;

  /**
   * @brief set op channel resource
   * @param op_desc_ptr OpDesc pointer
   * @return status whether success
   */
  virtual DvppErrorCode SetAttrResource(ge::OpDescPtr& op_desc_ptr) const = 0;
}; // class DvppBuilder
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_BUILDER_H_
