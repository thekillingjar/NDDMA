/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_op.h"
#include "types/op_attr.h"
#include "common/common_inner.h"
#include "common/prof_api_reg.h"
#include "single_op/acl_op_executor_impl.h"

namespace {
    constexpr char_t const *OP_NAME_CAST = "Cast";
    constexpr char_t const *ATTR_NAME_TRUNCATE = "truncate";
    constexpr char_t const *ATTR_NAME_DST_TYPE = "dst_type";
    constexpr int32_t CAST_INPUT_NUM = 1;
    constexpr int32_t CAST_OUTPUT_NUM = 1;
}

aclError aclopCastImpl(const aclTensorDesc *srcDesc,
                   const aclDataBuffer *srcBuffer,
                   const aclTensorDesc *dstDesc,
                   aclDataBuffer *dstBuffer,
                   uint8_t truncate,
                   aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCast);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dstDesc);
    const aclTensorDesc *inputDesc[CAST_INPUT_NUM] = {srcDesc};
    const aclTensorDesc *outputDesc[CAST_OUTPUT_NUM] = {dstDesc};
    const aclDataBuffer *inputs[CAST_OUTPUT_NUM] = {srcBuffer};
    aclDataBuffer *outputs[CAST_OUTPUT_NUM] = {dstBuffer};
    aclopAttr opAttr;
    if (acl::GetIfCastHasTruncateAttr()) {
        ACL_LOG_INFO("Need to set truncate attr in aclopCast");
        (void)opAttr.SetAttr(ATTR_NAME_TRUNCATE, static_cast<bool>(truncate));
    }
    (void)opAttr.SetAttr(ATTR_NAME_DST_TYPE, static_cast<int64_t>(dstDesc->dataType));
    return aclopExecuteV2Impl(OP_NAME_CAST,
                          CAST_INPUT_NUM,
                          const_cast<aclTensorDesc **>(inputDesc),
                          const_cast<aclDataBuffer **>(inputs),
                          CAST_OUTPUT_NUM,
                          const_cast<aclTensorDesc **>(outputDesc),
                          outputs,
                          &opAttr,
                          stream);
}

aclError aclopCreateHandleForCastImpl(aclTensorDesc *srcDesc,
                                  aclTensorDesc *dstDesc,
                                  uint8_t truncate,
                                  aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCreateHandleForCast);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dstDesc);
    const aclTensorDesc *inputDesc[CAST_INPUT_NUM] = {srcDesc};
    const aclTensorDesc *outputDesc[CAST_OUTPUT_NUM] = {dstDesc};
    aclopAttr opAttr;
    if (acl::GetIfCastHasTruncateAttr()) {
        ACL_LOG_INFO("Need to set truncate attr in aclopCreateHandleForCast");
        (void)opAttr.SetAttr(ATTR_NAME_TRUNCATE, static_cast<bool>(truncate));
    }
    (void)opAttr.SetAttr(ATTR_NAME_DST_TYPE, static_cast<int64_t>(dstDesc->dataType));
    return aclopCreateHandleImpl(OP_NAME_CAST,
                             CAST_INPUT_NUM,
                             inputDesc,
                             CAST_OUTPUT_NUM,
                             outputDesc,
                             &opAttr,
                             handle);
}
