/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_

#include <string>
#include <vector>

#include "ge/ge_ir_build.h"
#include "framework/common/types.h"
#include "framework/common/ge_types.h"
#include "graph/model.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
struct ModelPartition {
  ModelPartitionType type;
  const uint8_t *data = nullptr;
  uint64_t size = 0UL;
};

struct OmFileContext {
  std::vector<ModelPartition> partition_datas_;
  std::vector<char_t> partition_table_;
  uint64_t model_data_len_ = 0UL;
};

class GE_FUNC_VISIBILITY OmFileLoadHelper {
 public:
  Status Init(const ModelData &model);

  Status Init(uint8_t *const model_data, const uint32_t model_data_size);

  Status Init(uint8_t *const model_data, const uint32_t model_data_size, const uint32_t model_num);

  Status Init(uint8_t *const model_data, const uint64_t model_data_size,
              const ModelFileHeader *file_header = nullptr);

  Status Init(uint8_t *const model_data,
              const uint64_t model_data_size,
              const uint32_t model_num,
              const ModelFileHeader *file_header = nullptr);

  Status GetModelPartition(const ModelPartitionType type, ModelPartition &partition);

  Status GetModelPartition(const ModelPartitionType type, ModelPartition &partition, const size_t model_index) const;

  const std::vector<ModelPartition> &GetModelPartitions(const size_t model_index) const;

  Status CheckModelCompatibility(const Model &model) const;

  static bool CheckPartitionTableNum(const uint32_t partition_num);

  std::vector<OmFileContext> model_contexts_;

 private:
  Status LoadModelPartitionTable(const uint8_t *const model_data, const uint64_t model_data_size,
                                 const size_t model_index, size_t &mem_offset,
                                 const ModelFileHeader *file_header = nullptr);

  Status LoadModelPartitionTable(const uint8_t *const model_data,
                                 const uint64_t model_data_size,
                                 const uint32_t model_num,
                                 const ModelFileHeader *file_header = nullptr);
  OmFileContext context_;
  bool is_inited_{false};
};

class GE_FUNC_VISIBILITY OmFileSaveHelper {
 public:
  ModelFileHeader &GetModelFileHeader() { return model_header_; }

  ModelPartitionTable *GetPartitionTable();

  Status AddPartition(const ModelPartition &partition);

  Status AddPartition(const ModelPartition &partition, const size_t cur_index);

  Status SaveModel(const char_t *const output_file, ModelBufferData &model, const bool save_to_file = true,
                   const bool is_partition_align = false, const uint32_t align_bytes = 32U);

  ModelPartitionTable *GetPartitionTable(const size_t cur_ctx_index,
                                         const bool is_partition_align = false, const uint32_t align_bytes = 32U);

 private:
  ModelFileHeader model_header_;
  std::vector<OmFileContext> model_contexts_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
