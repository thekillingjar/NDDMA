/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_AIPP_UTILS_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_AIPP_UTILS_H_

#include <map>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/op_desc.h"
#include "proto/insert_op.pb.h"

namespace ge {
constexpr uint32_t kAippOriginInputIndex = 0U;
constexpr uint32_t kAippInfoNum = 6U;
constexpr uint32_t kAippInfoFormat = 0U;
constexpr uint32_t kAippInfoDataType = 1U;
constexpr uint32_t kAippInfoTensorName = 2U;
constexpr uint32_t kAippInfoTensorSize = 3U;
constexpr uint32_t kAippInfoDimNum = 4U;
constexpr uint32_t kAippInfoShape = 5U;

class AippUtils {
 public:
  AippUtils() = default;
  ~AippUtils() = default;

  static Status ConvertAippParams2AippInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info);
  static Status SetAippInfoAndTypeFromOpDesc(const std::map<std::string, uint32_t> &data_index_map,
                                             const OpDescPtr &op_desc, const uint32_t index,
                                             std::map<uint32_t, AippConfigInfo> &aipp_infos,
                                             std::map<uint32_t, std::pair<InputAippType, size_t>> &aipp_types);
  static Status GetAippInfo(const std::map<uint32_t, AippConfigInfo> &aipp_infos, const uint32_t index,
                            AippConfigInfo &aipp_info);
  static Status GetAippType(const std::map<uint32_t, std::pair<InputAippType, size_t>> &aipp_types,
                            const uint32_t index, InputAippType &aipp_type, size_t &aipp_data_index);

  static Status SetAippInputOutputInfoFromOpDesc(const OpDescPtr &op_desc, const uint32_t index,
      std::map<uint32_t, OriginInputInfo> &orig_input_info_map,
      std::map<uint32_t, std::pair<std::vector<InputOutputDims>, std::vector<InputOutputDims>>> &aipp_dims_info);

  static Status GetOrigInputInfo(const std::map<uint32_t, OriginInputInfo> &orig_input_info_map, const uint32_t index,
      OriginInputInfo &orig_input_info);

  static Status GetAllAippInputOutputDims(
      const std::map<uint32_t,
      std::pair<std::vector<InputOutputDims>, std::vector<InputOutputDims>>> &aipp_dims_info,
      const uint32_t index, std::vector<InputOutputDims> &input_dims, std::vector<InputOutputDims> &output_dims);

 private:
  static Status SetAippInfoImpl(const NamedAttrs &aipp_attr,
                                const OpDescPtr &op_desc, const uint32_t index,
                                std::map<uint32_t, AippConfigInfo> &aipp_infos);
  static Status SetAippTypeImpl(const std::map<std::string, uint32_t> &data_index_map,
                                const std::string &data_mode, const OpDescPtr &op_desc, const uint32_t index,
                                std::map<uint32_t, std::pair<InputAippType, size_t>> &aipp_types);
  static void SetMatrixInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info);
  static void SetBiasInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info);
  static void SetChnInfo(const domi::AippOpParams &aipp_params, AippConfigInfo &aipp_info);
  static Status ParseAIPPInfo(const std::string &in_out_info, InputOutputDims &dims_info);
  static void SetOrigInputInfo(const std::string &input, const uint32_t index,
      std::map<uint32_t, OriginInputInfo> &orig_input_info_map);
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_AIPP_UTILS_H_
