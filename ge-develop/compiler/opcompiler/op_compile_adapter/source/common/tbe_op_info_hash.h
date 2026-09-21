/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TBE_OP_INFO_HASH_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TBE_OP_INFO_HASH_H_

#include "tensor_engine/tbe_op_info.h"

namespace te {
namespace fusion {
// Hash function for TbeInfo
struct HashTbeOpInfo {
    size_t operator()(const TbeOpInfo &tbeOpInfo) const;
    size_t operator()(const TbeOpInfoPtr &tbeOpInfo) const;
};

// EqualTo function for TbeOpInfo
struct EqualToTbeOpInfo {
    bool operator()(const TbeOpInfo &a, const TbeOpInfo &b) const;
    bool operator()(const TbeOpInfoPtr &a, const TbeOpInfoPtr &b) const;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TBE_OP_INFO_HASH_H_