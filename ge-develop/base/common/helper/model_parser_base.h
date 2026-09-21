/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_MODEL_PARSER_BASE_H_
#define GE_COMMON_MODEL_PARSER_BASE_H_

#include <memory>

#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "framework/common/ge_model_inout_types.h"

namespace ge {
class ModelParserBase {
 public:
  /**
   * @ingroup hiai
   * @brief Parsing a model file
   * @param [in] model_path  model path
   * @param [in] priority    modle priority
   * @param [out] model_data model data
   * @return Status  result
   */
  static Status LoadFromFile(const char_t *const model_path, const int32_t priority, ModelData &model_data);

  /**
   * @ingroup domi_ome
   * @brief Parse model contents from the ModelData
   * @param [in] model  model data read from file
   * @param [out] model_data  address of the model data
   * @param [out] model_len  model actual length
   * If the input is an encrypted model, it needs to be deleted
   * @return SUCCESS success
   * @return others failure
   * @author
   */
  static Status ParseModelContent(const ModelData &model, uint8_t *&model_data, uint64_t &model_len);

  static bool IsDynamicModel(const ModelFileHeader &file_header);

  static Status GetModelInputDesc(const uint8_t * const data, const size_t size, ModelInOutInfo &info);

  static Status GetModelOutputDesc(const uint8_t * const data, const size_t size, ModelInOutInfo &info);

  static Status GetDynamicBatch(const uint8_t * const data, const size_t size, ModelInOutInfo &info);

  static Status GetDynamicHW(const uint8_t * const data, const size_t size, ModelInOutInfo &info);

  static Status GetDynamicDims(const uint8_t * const data, const size_t size, ModelInOutInfo &info);

  static Status GetDataNameOrder(const uint8_t * const data, const size_t size, ModelInOutInfo &info);

  static Status GetDynamicOutShape(const uint8_t * const data, const size_t size, ModelInOutInfo &info);
};
}  //  namespace ge
#endif  // GE_COMMON_MODEL_PARSER_BASE_H_
