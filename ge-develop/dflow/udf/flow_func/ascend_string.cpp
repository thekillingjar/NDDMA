/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func/ascend_string.h"
#include "common/udf_log.h"

namespace FlowFunc {
AscendString::AscendString(const char * const name) {
    if (name != nullptr) {
        try {
            name_ = std::make_shared<std::string>(name);
        } catch (std::exception &e) {
            UDF_LOG_ERROR("[New][String]AscendString[%s] make shared failed, error=%s.", name, e.what());
        }
    }
}

AscendString::AscendString(const char * const name, size_t length) {
    if (name != nullptr) {
        try {
            name_ = std::make_shared<std::string>(name, length);
        } catch (std::exception &e) {
            UDF_LOG_ERROR("[New][String]AscendString make shared failed, length=%zu, error=%s.", length, e.what());
        }
    }
}

const char *AscendString::GetString() const {
    const static char *empty_value = "";
    return (name_ == nullptr) ? empty_value : name_->c_str();
}

size_t AscendString::GetLength() const {
    if (name_ == nullptr) {
        return 0UL;
    }
    return name_->length();
}

bool AscendString::operator<(const AscendString &d) const {
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return false;
    } else if (name_ == nullptr) {
        return true;
    } else if (d.name_ == nullptr) {
        return false;
    } else {
        return (*name_) < (*(d.name_));
    }
}

bool AscendString::operator>(const AscendString &d) const {
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return false;
    } else if (name_ == nullptr) {
        return false;
    } else if (d.name_ == nullptr) {
        return true;
    } else {
        return (*name_) > (*(d.name_));
    }
}

bool AscendString::operator==(const AscendString &d) const {
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return true;
    } else if (name_ == nullptr) {
        return false;
    } else if (d.name_ == nullptr) {
        return false;
    } else {
        return (*name_) == (*(d.name_));
    }
}

bool AscendString::operator<=(const AscendString &d) const {
    if (name_ == nullptr) {
        return true;
    } else if (d.name_ == nullptr) {
        return false;
    } else {
        return (*name_) <= (*(d.name_));
    }
}

bool AscendString::operator>=(const AscendString &d) const {
    if (d.name_ == nullptr) {
        return true;
    } else if (name_ == nullptr) {
        return false;
    } else {
        return (*name_) >= (*(d.name_));
    }
}

bool AscendString::operator!=(const AscendString &d) const {
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return false;
    } else if (name_ == nullptr) {
        return true;
    } else if (d.name_ == nullptr) {
        return true;
    } else {
        return (*name_) != (*(d.name_));
    }
}
}
