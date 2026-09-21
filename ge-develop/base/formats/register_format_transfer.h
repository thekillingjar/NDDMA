/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_REGISTER_FORMAT_TRANSFER_H_
#define INC_REGISTER_REGISTER_FORMAT_TRANSFER_H_

#include <memory>
#include <vector>
#include <functional>

#include "graph/types.h"
#include "ge/ge_api_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"

namespace ge {
namespace formats {
struct TransArgs {
  const uint8_t *data;
  Format src_format; // src full format
  Format dst_format; // dst full format
  Format src_primary_format; // src primary format
  Format dst_primary_format; // dst primary format
  Format src_sub_format; // src sub format
  Format dst_sub_format; // dst sub format
  int64_t src_c0_format; // src s0 format
  int64_t dst_c0_format; // dst c0 format
  // For scenes that need to supplement the shape, for example, 5D to 4D
  // It is not possible to convert the format normally if you only get the src_shape,
  // and must get the shape before you mend the shape.
  // So the parameters here need to be passed in both src_shape and dst_shape
  std::vector<int64_t> src_shape;
  std::vector<int64_t> dst_shape;
  DataType src_data_type;
};

struct TransResult {
  std::shared_ptr<uint8_t> data;
  // data length in bytes
  size_t length;
};

struct CastArgs {
  const uint8_t *data;
  size_t src_data_size;
  DataType src_data_type;
  DataType dst_data_type;
};

class FormatTransfer {
 public:
  FormatTransfer() = default;
  virtual ~FormatTransfer() = default;
  virtual Status TransFormat(const TransArgs &args, TransResult &result) = 0;
  virtual Status TransShape(Format src_format, const std::vector<int64_t> &src_shape, DataType data_type,
                            Format dst_format, std::vector<int64_t> &dst_shape) = 0;

 private:
  FormatTransfer(const FormatTransfer &) = delete;
  FormatTransfer &operator=(const FormatTransfer &) = delete;
};

using FormatTransferBuilder = std::function<std::shared_ptr<FormatTransfer>()>;

class FormatTransferRegister {
 public:
  FormatTransferRegister(FormatTransferBuilder builder, const Format src, const Format dst) noexcept;
  ~FormatTransferRegister() = default;
};

/// Build a formattransfer according to 'args'
/// @param args
/// @param result
/// @return
std::shared_ptr<FormatTransfer> BuildFormatTransfer(const TransArgs &args);

bool FormatTransferExists(const TransArgs &args);
}  // namespace formats
}  // namespace ge

#define REGISTER_FORMAT_TRANSFER(TransferClass, format1, format2)                 \
namespace {                                                                       \
  std::shared_ptr<FormatTransfer> Transfer_##format1##_##format2() {              \
    return ge::MakeShared<TransferClass>();                                       \
  }                                                                               \
  FormatTransferRegister format_transfer_##TransferClass##_##format1##_##format2( \
      &Transfer_##format1##_##format2, (format1), (format2));                     \
}
#endif  // INC_REGISTER_REGISTER_FORMAT_TRANSFER_H_