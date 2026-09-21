/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_PREBUILD_MANAGER_H
#define TE_PREBUILD_MANAGER_H

#include <string>
#include <mutex>
#include <map>
#include <Python.h>
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
class PreBuildManager {
public:
    PreBuildManager() {};
    ~PreBuildManager() {};

    static PreBuildManager& GetInstance();

    /**
     * @brief pre build tbe op
     * @return [out] bool
     */
    bool PreBuildTbeOpInner(TbeOpInfo &opinfo, uint64_t taskId, uint64_t graphId);

private:
    bool DoPrebuildOp(const OpBuildTaskPtr &opTask);

    void CheckOpImplMode(const TbeOpInfoPtr &tbeOpInfo);

    static bool NeedSkipPrebuild(const OpBuildTaskPtr &opTask, TbeOpInfo &tbeOpInfo);

    static bool CanReusePreBuildCache(const TbeOpInfoPtr &tbeOpInfoPtr, const OpBuildTaskPtr &opTask,
                                      OpBuildTaskResultPtr &opRes);
    static void UpdateOpTaskForPreBuildCache(const OpBuildTaskPtr &opTask, const OpBuildTaskResultPtr &opResPtr);
    static bool CanReusePreBuildDiskCache(const OpBuildTaskPtr &opTask);

    static bool SetPrebuildTargetsToCtx(const TbeOpInfo &tbeOpInfo, PyObject *contextDict);

private:
    mutable std::mutex opImplModeSupportedMapMutex_;
    std::map<std::string, bool> opImplModeSupportedMap_;
};
} // namespace fusion
} // namespace te
#endif