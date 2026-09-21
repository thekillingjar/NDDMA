/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_FZ_C04_H_
#define GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_FZ_C04_H_

#include <vector>
#include "formats/register_format_transfer.h"

namespace ge {
namespace formats {
class FormatTransfer4DToFZC04 : public FormatTransfer {
 public:
  Status TransFormat(const ge::formats::TransArgs &args, ge::formats::TransResult &result) override;
  Status TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                    const DataType data_type, const Format dst_format,
                    std::vector<int64_t> &dst_shape) override;
 private:
  Status BuildTransArgsNchwToFzC04(const ge::formats::TransArgs &src_dst_args, TransResult &nchw_result_holder,
                                   ge::formats::TransArgs &nchw_dst_args) const;
  Status TransShapeFromSrcToNchw(const Format src_format, const std::vector<int64_t> &src_shape,
                                 const DataType data_type, std::vector<int64_t> &nchw_shape) const;
};

class FormatTransferFZC04To4D : public FormatTransfer {
 public:
  Status TransFormat(const ge::formats::TransArgs &args, ge::formats::TransResult &result) override;
  Status TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                    const DataType data_type, const Format dst_format,
                    std::vector<int64_t> &dst_shape) override;
};
}  // namespace formats
}  // namespace ge

#endif  // GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_NCHW_FZ_C04_H_
