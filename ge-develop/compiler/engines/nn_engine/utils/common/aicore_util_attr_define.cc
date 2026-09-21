/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/aicore_util_attr_define.h"

namespace fe {
const std::string kKernelName = "_kernelname";

const std::string kOpPattern = "_pattern";

const std::string kCanNotReuseOm = "_can_not_reuse_om";

const std::string AOE_TYPE = "aoe_type";

const std::string IS_NEED_TUNEFORMAT = "is_need_tuneformat";

const std::string AOE_TUNEFORMAT_REQ = "tuneformat_req";

const std::string AOE_TUNEFORMAT = "tuneformat";

const std::string ATTR_NAME_ATOMIC_CLEAN_NODE = "atomic_clean_node_ptr";

const std::string ATTR_NAME_MEMSET_NODE = "memset_node_ptr";

const std::string ATTR_NAME_ORIGINAL_NODE = "original_node_ptr";

const std::string ATOMIC_CLEAN_OP_TYPE = "AtomicAddrClean";

const std::string MEMSET_OP_TYPE = "MemSet";

const std::string DYNAMIC_ATOMIC_CLEAN_OP_TYPE = "DynamicAtomicAddrClean";

const std::string TBE_OP_ATOMIC_OUTPUT_INDEX = "tbe_op_atomic_output_index";

const std::string kDeletedAtomicOutputIndex = "_deleted_atomic_output_index";

const std::string TBE_OP_ATOMIC_WORKSPACE_INDEX = "tbe_op_atomic_workspace_index";

const std::string TBE_OP_ATOMIC_WSP_MODE = "wspMode";

const std::string TBE_OP_ATOMIC_DTYPES = "tbe_op_atomic_dtypes";

const std::string TBE_OP_ATOMIC_INT64_VALUES = "tbe_op_atomic_int64_values";

const std::string TBE_OP_ATOMIC_FLOAT_VALUES = "tbe_op_atomic_float_values";

const std::string TBE_OP_ATOMIC_WORKSPACE_FLAG = "tbe_op_atomic_workspace_flag";

const std::string COMPILE_INFO_JSON = "compile_info_json";

const std::string COMPILE_INFO_KEY = "compile_info_key";

const std::string kThreadAddrOffset = "_thread_addr_offset";

const std::string ATTR_NAME_FALLBACK_ACLNN = "_fallback_aclnn";

// todo tmp using, need change to ATTR_NAME_DYNAMIC_TILING_DEPEND_OP
const std::string kDynamicTilingDependOp = "_dynamic_tiling_depend_op";

const std::string kOpDfxOptions = "_op_dfx_options";

const std::string kOpDfxBufferSize = "_op_dfx_buffer_size";
} // namespace fe
