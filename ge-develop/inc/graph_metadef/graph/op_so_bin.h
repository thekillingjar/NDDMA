/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_OP_SO_BIN_H
#define INC_GRAPH_OP_SO_BIN_H

#include <string>
#include <utility>
#include "graph/types.h"
#include "graph/def_types.h"

namespace ge {
typedef struct {
  std::string cpu_info;
  std::string os_info;
  std::string opp_version;
  std::string compiler_version;
} SoInOmInfo;

enum class SoBinType : uint16_t {
  kSpaceRegistry = 0,  // for use of rt2 infer shape/tiling so, managed in OpImplSpaceRegistry
  kOpMasterDevice = 1,  // for use of tiling so on device
  kAutofuse = 2  // for use of autofuse so offline saving and loading
};

class OpSoBin {
public:
  OpSoBin(const std::string &so_name, const std::string &vendor_name,
      std::unique_ptr<char_t[]> data, uint32_t data_len)
      : so_name_(so_name), vendor_name_(vendor_name), data_(std::move(data)),
        data_size_(data_len) {}

 OpSoBin(const std::string &so_name, const std::string &vendor_name, std::unique_ptr<char_t[]> data, uint32_t data_len,
         SoBinType so_bin_type)
     : so_name_(so_name), vendor_name_(vendor_name), data_(std::move(data)), data_size_(data_len),
       so_bin_type_(so_bin_type) {}

 ~OpSoBin() = default;

  const std::string &GetSoName() const { return so_name_; }
  const std::string &GetVendorName() const { return vendor_name_; }
  const uint8_t *GetBinData() const { return ge::PtrToPtr<void, const uint8_t>(data_.get()); }
  uint8_t *MutableBinData() const {
    return PtrToPtr<void, uint8_t>(data_.get());
  }
  size_t GetBinDataSize() const { return data_size_; }
  SoBinType GetSoBinType() const { return so_bin_type_; }
  OpSoBin(const OpSoBin &) = delete;
  const OpSoBin &operator=(const OpSoBin &) = delete;

private:
  std::string so_name_;
  std::string vendor_name_;
  std::unique_ptr<char_t[]> data_;
  uint32_t data_size_;
  SoBinType so_bin_type_{SoBinType::kSpaceRegistry};
};

using OpSoBinPtr = std::shared_ptr<ge::OpSoBin>;
}  // namespace ge

#endif  // INC_GRAPH_OP_SO_BIN_H
