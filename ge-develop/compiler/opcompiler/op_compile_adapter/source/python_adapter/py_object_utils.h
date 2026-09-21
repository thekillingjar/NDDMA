/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PY_OBJECT_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PY_OBJECT_UTILS_H_

#include <Python.h>
#include "tensor_engine/tbe_op_info.h"

namespace te {
namespace fusion {
class PyObjectUtils {
public:
    static PyObject *GenPySocInfo();
    static PyObject *GenPyOptionDict();
    static PyObject *GenPyOptionsInfo(const std::vector<ConstTbeOpInfoPtr> &tbeOpInfoVec);
    static bool GenPyOptionPair(const std::string &key, const std::string &val, PyObject *pySocParamDict);
    static PyObject *GenSuperKernelOptionsInfo(const std::map<std::string, std::string> &skOptions);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PY_OBJECT_UTILS_H_
