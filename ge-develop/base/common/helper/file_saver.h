/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_AUTH_FILE_SAVER_H_
#define GE_COMMON_AUTH_FILE_SAVER_H_

#include <string>
#include <vector>

#include "framework/common/helper/om_file_helper.h"
#include "framework/common/types.h"
#include "ge/ge_ir_build.h"
#include "graph/buffer.h"
#include "mmpa/mmpa_api.h"

namespace ge {
class FileSaver {
 public:
  /// @ingroup domi_common
  /// @brief save model, no encryption
  /// @return Status  result
  static Status SaveToFile(const std::string &file_path, const ge::ModelData &model,
                           const ModelFileHeader *const model_file_header = nullptr);

  static Status SaveToFile(const std::string &file_path, const ModelFileHeader &model_file_header,
                           const ModelPartitionTable &model_partition_table,
                           const std::vector<ModelPartition> &partition_datas);

  static Status SaveToFile(const std::string &file_path, const ModelFileHeader &file_header,
                           const std::vector<ModelPartitionTable *> &model_partition_tables,
                           const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                           const bool is_partition_align = false,
                           const uint32_t align_bytes = 32U);

  static Status SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                         ModelPartitionTable &model_partition_table,
                                         const std::vector<ModelPartition> &partition_datas,
                                         ge::ModelBufferData& model);

  static Status SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                         const std::vector<ModelPartitionTable *> &model_partition_tables,
                                         const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                                         ge::ModelBufferData &model);

  static Status SaveToFile(const std::string &file_path, const void *const data, const uint64_t len,
                           const bool append = false);

  static void PrintModelSaveLog();

  static void SetHostPlatformParamInitialized(const bool host_platform_param_initialized) {
    host_platform_param_initialized_ = host_platform_param_initialized;
  }

 protected:
  /// @ingroup domi_common
  /// @brief Check validity of the file path
  /// @return Status  result
  static Status CheckPathValid(const std::string &file_path);

  static Status WriteData(const void * const data, uint64_t size, const int32_t fd);

  static Status OpenFile(int32_t &fd, const std::string &file_path, const bool append = false);

  /// @ingroup domi_common
  /// @brief save model to file
  /// @param [in] file_path  file output path
  /// @param [in] file_header  file header info
  /// @param [in] data  model data
  /// @param [in] len  model length
  /// @return Status  result
  static Status SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                   const void *const data, const uint64_t len);

  static Status SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                   const ModelPartitionTable &model_partition_table,
                                   const std::vector<ModelPartition> &partition_datas);
  static Status SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                   const std::vector<ModelPartitionTable *> &model_partition_tables,
                                   const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                                   const bool is_partition_align = false,
                                   const uint32_t align_bytes = 32U);
 private:
  static Status SaveWithAlignFill(uint32_t size, uint32_t align_bytes, const int32_t fd);
  static bool host_platform_param_initialized_;
};
}  // namespace ge
#endif  // GE_COMMON_AUTH_FILE_SAVER_H_
