/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <securec.h>
#include "graph/ascend_string.h"
#include "register/tilingdata_base.h"
#include "base/asc/tilingdata_base_impl.h"


namespace optiling {

void TilingDef::SaveToBuffer(void *pdata, size_t capacity) {
    TilingDefImpl::SaveToBuffer(*this, pdata, capacity);
}

std::vector<FieldInfo> TilingDef::GetFieldInfo() const {
    return TilingDefImpl::GetFieldInfo(*this);
}

const char *TilingDef::GetTilingClassName() const {
    return TilingDefImpl::GetTilingClassName(*this);
}

size_t TilingDef::GetDataSize() const {
    return TilingDefImpl::GetDataSize(*this);
}

void TilingDef::SetDataPtr(void *dataPtr) {
    TilingDefImpl::SetDataPtr(*this, dataPtr);
}

void TilingDef::CheckAlignAndGenPlaceHolder(const char *name, size_t typeSize) {
    TilingDefImpl::CheckAlignAndGenPlaceHolder(*this, name, typeSize);
}

void TilingDef::InitData() {
    TilingDefImpl::InitData(*this);
}

void TilingDef::GeLogError(const std::string &str) const {
    TilingDefImpl::GeLogError(str);
}

CTilingDataClassFactory &CTilingDataClassFactory::GetInstance()
{
    static CTilingDataClassFactory instance;
    return instance;
}

void CTilingDataClassFactory::RegisterTilingData(const char *op_type,
                                                 const TilingDataConstructor constructor) {
    CTilingDataClassFactoryImpl::GetInstance().RegisterTilingData(op_type, constructor);
}

std::shared_ptr<TilingDef> CTilingDataClassFactory::CreateTilingDataInstance(const char *op_type) {
    return CTilingDataClassFactoryImpl::GetInstance().CreateTilingDataInstance(op_type);
}

uint32_t __attribute__((weak)) TilingDataStructBase::RecordTilingStruct(const char* name, const char* file, \
    uint32_t line) {
    return TilingDataStructBaseImpl::GetInstance().RecordTilingStruct(name, file, line);
}
}  // end of namespace optiling
