/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "util_stub.h"
using namespace std;
using namespace ge;
namespace aicpu {

void GetDataTypeStub(const std::map<std::string, std::string> &dataTypes,
                 const std::string &dataTypeName,
                 std::set<ge::DataType> &dataType) 
{
    dataType.insert(DataType::DT_FLOAT16);
}

void GetDataTypeStub(const std::map<std::string, std::string> &dataTypes,
                 const std::string &dataTypeName,
                 std::vector<ge::DataType> &dataType) {}

const std::string RealPathStub(const std::string &path)
{
    const char *curDirName = getenv("HOME");
    if (curDirName == nullptr) {
        return "/home/";
    }
    std::string dirName(curDirName);
    return dirName;
}
}