/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_OPS_LIB_H_
#define DVPP_ENGINE_COMMON_DVPP_OPS_LIB_H_

#include <mutex>
#include <nlohmann/json.hpp>
#include "common/dvpp_ops.h"
#include "util/dvpp_error_code.h"
#include "graph/node.h"

namespace dvpp {
class DvppOpsLib {
 public:
  /**
   * @brief Get DvppOpsLib instance
   * @return DvppOpsLib instance
   */
  static DvppOpsLib& Instance();

  /**
   * @brief Destructor
   */
  virtual ~DvppOpsLib() = default;

  // Copy constructor prohibited
  DvppOpsLib(const DvppOpsLib& dvpp_ops_lib) = delete;

  // Move constructor prohibited
  DvppOpsLib(const DvppOpsLib&& dvpp_ops_lib) = delete;

  // Copy assignment prohibited
  DvppOpsLib& operator=(const DvppOpsLib& dvpp_ops_lib) = delete;

  // Move assignment prohibited
  DvppOpsLib& operator=(DvppOpsLib&& dvpp_ops_lib) = delete;

  DvppErrorCode Initialize(std::string& file_name);

  std::map<std::string, DvppOpInfo> AllOps() const;

  bool GetDvppOpInfo(std::string& op_name, DvppOpInfo& dvpp_op_info) const;

  /**
   * @brief create dvpp op desc
   * @param op_type op type
   * @param op_desc_ptr op info
   * @return status whether success
   */
  DvppErrorCode CreateDvppOpDesc(
      const std::string& op_type, ge::OpDescPtr& op_desc_ptr);

  /**
   * @brief create reduce mean op desc
   * @param op_desc_ptr op info
   * @return status whether success
   */
  DvppErrorCode CreateAICpuReduceMeanDesc(ge::OpDescPtr& op_desc_ptr);

  /**
   * @brief create if op desc
   * @param op_type op type
   * @param op_desc_ptr op info
   * @return status whether success
   */
  DvppErrorCode CreateIfOpDesc(
      const std::string& op_type, ge::OpDescPtr& op_desc_ptr);

 private:
  /**
   * @brief Constructor
   */
  DvppOpsLib() = default;

  /**
   * @brief read json file
   * @param file_name dvpp ops info json file
   * @return whether success
   */
  DvppErrorCode ReadOpsInfoFromJsonFile(std::string& file_name);

  DvppErrorCode GetAllOpsInfo(DvppOpsInfoLib& dvpp_ops_info_lib);

  void CreateAdjustContrastWithMeanDesc(ge::OpDescPtr& op_desc_ptr);

  void CreateRgbToGrayscaleDesc(ge::OpDescPtr& op_desc_ptr);

  /**
   * @brief create op name for only ID
   * @param op_type op type
   * @return op name
   */
  std::string CreateDvppOpName(const std::string& op_type);

 private:
  nlohmann::json dvpp_ops_info_json_file_;
  std::map<std::string, DvppOpInfo> all_ops_;
  std::mutex op_num_mutex_;
}; // class DvppOpsLib
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_OPS_LIB_H_
