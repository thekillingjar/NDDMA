/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_HWCN_C1HWNCOC0_H_
#define GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_HWCN_C1HWNCOC0_H_

#include <vector>
#include "formats/register_format_transfer.h"

namespace ge {
namespace formats {
class FormatTransferHwcnC1hwncoc0 : public FormatTransfer {
 public:
  Status TransFormat(const TransArgs &args, TransResult &result) override;
  Status TransShape(const Format src_format, const std::vector<int64_t> &src_shape,
                    const DataType data_type, const Format dst_format,
                    std::vector<int64_t> &dst_shape) override;
};
}  // namespace formats
}  // namespace ge

#endif  // GE_COMMON_FORMATS_FORMAT_TRANSFERS_FORMAT_TRANSFER_HWCN_C1HWNCOC0_H_
