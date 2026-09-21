/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_model.h"
#include <sstream>
#include "framework/common/util.h"
#include "utils/attr_utils.h"

namespace acl {
std::string OpModelDef::DebugString() const
{
    std::stringstream ss;
    ss << "[OpModelDef] Path: " << modelPath;
    ss << ", OpType: " << opType << ", ";
    for (size_t i = 0UL; i < inputDescArr.size(); ++i) {
        const aclTensorDesc &desc = inputDescArr.at(i);
        ss << "InputDesc[" << i << "]: ";
        ss << desc.DebugString() << " ";
    }

    for (size_t i = 0UL; i < outputDescArr.size(); ++i) {
        const aclTensorDesc &desc = outputDescArr.at(i);
        ss << "OutputDesc[" << i << "]: ";
        ss << desc.DebugString() << " ";
    }

    ss << ", Attr: " << opAttr.DebugString();
    ss << ", isStaticModelWithFuzzCompile: " << isStaticModelWithFuzzCompile;
    return ss.str();
}

aclError ReadOpModelFromFile(const std::string &path, OpModel &opModel)
{
    int32_t fileSize;
    char_t *data = nullptr;
    ACL_LOG_DEBUG("start to call ge::ReadBytesFromBinaryFile");
    if (!ge::ReadBytesFromBinaryFile(path.c_str(), &data, fileSize)) {
        ACL_LOG_CALL_ERROR("[Read][Bytes]Read model file failed. path = %s", path.c_str());
        return ACL_ERROR_READ_MODEL_FAILURE;
    }

    const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
    opModel.data = p_data;
    opModel.size = static_cast<uint32_t>(fileSize);
    opModel.name = path;

    ACL_LOG_INFO("Read model file succeeded. file = %s, length = %d", path.c_str(), fileSize);
    return ACL_SUCCESS;
}
} // namespace acl