/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <set>
#include <vector>
#include <map>
#include "ge/ge_api_error_codes.h"
#include "graph/types.h"
#include "graph/compute_graph.h"
#include "ge/ge_api_types.h"

namespace aicpu {
    void GetDataTypeStub(const std::map<std::string, std::string> &dataTypes,
                 const std::string &dataTypeName,
                 std::set<ge::DataType> &dataType);
    void GetDataTypeStub(const std::map<std::string, std::string> &dataTypes,
                 const std::string &dataTypeName,
                 std::vector<ge::DataType> &dataType);

    const std::string RealPathStub(const std::string &path);
}
