/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_op_inner.h"
#include <sstream>

namespace {
    constexpr size_t MAX_ACLOP_STRING_LENGTH = 800U;
}

namespace acl {
AclOp::AclOp(const AclOp& aclOp)
{
    Init(aclOp);
}

AclOp &AclOp::operator=(const AclOp &aclOp) &
{
    if (this != &aclOp) {
        Init(aclOp);
    }
    return *this;
}

AclOp::~AclOp()
{
    if (!this->isCopyConstructor) {
        return;
    }
    if (this->inputDesc != nullptr) {
        for (int32_t i = 0; i < this->numInputs; ++i) {
            if (this->inputDesc[i] != nullptr) {
                delete this->inputDesc[i];
            }
        }
        free(const_cast<aclTensorDesc **>(this->inputDesc));
        this->inputDesc = nullptr;
    }
    if (this->outputDesc != nullptr) {
        for (int32_t i = 0; i < this->numOutputs; ++i) {
            if (this->outputDesc[i] != nullptr) {
                delete this->outputDesc[i];
            }
        }
        free(const_cast<aclTensorDesc **>(this->outputDesc));
        this->outputDesc = nullptr;
    }
    ACL_DELETE_AND_SET_NULL(this->opAttr);
}

std::string AclOp::DebugString() const
{
    std::stringstream ss;
    ss << "OpType: " << opType << ", ";
    for (int32_t i = 0; i < numInputs; ++i) {
        const aclTensorDesc *const desc = inputDesc[i];
        ss << "InputDesc[" << i << "]: ";
        ss << desc->DebugString() << " ";
    }

    for (int32_t i = 0; i < numOutputs; ++i) {
        const aclTensorDesc *const desc = outputDesc[i];
        ss << "OutputDesc[" << i << "]: ";
        ss << desc->DebugString() << " ";
    }

    if (opAttr != nullptr) {
        const std::string attrStr = opAttr->DebugString();
        ss << "Attr: " << attrStr;
    }
    if (ss.str().length() > MAX_ACLOP_STRING_LENGTH) {
        // in case of being truncated out of log limit 1024
        size_t index = 0U;
        while (index < ss.str().length()) {
            ACL_LOG_DEBUG("%s", ss.str().substr(index, MAX_ACLOP_STRING_LENGTH).c_str());
            index += MAX_ACLOP_STRING_LENGTH;
        }
    }

    return ss.str();
}

void AclOp::Init(const AclOp& aclOp)
{
    this->isCopyConstructor = true;
    this->opType = aclOp.opType;
    this->numInputs = aclOp.numInputs;
    this->numOutputs = aclOp.numOutputs;
    if ((aclOp.inputDesc != nullptr) && (aclOp.numInputs > 0)) {
        const size_t len = static_cast<size_t>(aclOp.numInputs) * sizeof(uintptr_t);
        aclTensorDesc **desc = static_cast<aclTensorDesc **>(malloc(len));
        ACL_REQUIRES_NOT_NULL_RET_VOID(desc);
        this->inputDesc = static_cast<const aclTensorDesc * const *>(desc);
        for (int32_t i = 0; i < this->numInputs; ++i) {
            if (aclOp.inputDesc[i] != nullptr) {
                desc[i] = new(std::nothrow) aclTensorDesc(*aclOp.inputDesc[i]);
                ACL_REQUIRES_NOT_NULL_RET_VOID(desc[i]);
            } else {
                desc[i] = nullptr;
            }
        }
    }
    if ((aclOp.outputDesc != nullptr) && (aclOp.numOutputs > 0)) {
        const size_t len = static_cast<size_t>(aclOp.numOutputs) * sizeof(uintptr_t);
        aclTensorDesc **desc = static_cast<aclTensorDesc **>(malloc(len));
        ACL_REQUIRES_NOT_NULL_RET_VOID(desc);

        this->outputDesc = static_cast<const aclTensorDesc * const *>(desc);
        for (int32_t i = 0; i < this->numOutputs; ++i) {
            if (aclOp.outputDesc[i] != nullptr) {
                desc[i] = new(std::nothrow) aclTensorDesc(*aclOp.outputDesc[i]);
                ACL_REQUIRES_NOT_NULL_RET_VOID(desc[i]);
            } else {
                desc[i] = nullptr;
            }
        }
    }
    if (aclOp.opAttr != nullptr) {
        this->opAttr = new(std::nothrow) aclopAttr(*aclOp.opAttr);
        ACL_REQUIRES_NOT_NULL_RET_VOID(this->opAttr);
    }
    this->inputs = aclOp.inputs;
    this->outputs = aclOp.outputs;
    this->engineType = aclOp.engineType;
    this->opPath = aclOp.opPath;
    this->compileType = aclOp.compileType;
    this->isCompile = aclOp.isCompile;
    this->exeucteType = aclOp.exeucteType;
    this->isMatched = aclOp.isMatched;
    this->isDynamic = aclOp.isDynamic;
    this->opModel = aclOp.opModel;
}

void AclOp::BackupConst() const
{
    for (int32_t i = 0; i < numInputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(inputDesc));
        tmp[i]->BackupConst();
    }
    for (int32_t i = 0; i < numOutputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(outputDesc));
        tmp[i]->BackupConst();
    }
}

void AclOp::RecoverConst() const
{
    for (int32_t i = 0; i < numInputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(inputDesc));
        tmp[i]->RecoverConst();
    }
    for (int32_t i = 0; i < numOutputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(outputDesc));
        tmp[i]->RecoverConst();
    }
}

void AclOp::BackupDimsAndShapeRanges() const
{
    for (int32_t i = 0; i < numInputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(inputDesc));
        tmp[i]->BackupDimsAndShapeRanges();
    }
    for (int32_t i = 0; i < numOutputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(outputDesc));
        tmp[i]->BackupDimsAndShapeRanges();
    }
}

void AclOp::RecoverDimsAndShapeRanges() const
{
    for (int32_t i = 0; i < numInputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(inputDesc));
        tmp[i]->RecoverDimsAndShapeRanges();
    }
    for (int32_t i = 0; i < numOutputs; ++i) {
        aclTensorDesc **const tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(outputDesc));
        tmp[i]->RecoverDimsAndShapeRanges();
    }
}
} // namespace acl

