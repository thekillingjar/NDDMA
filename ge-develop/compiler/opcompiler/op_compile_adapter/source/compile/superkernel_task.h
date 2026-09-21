/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SUPERKERNEL_TASK_H
#define SUPERKERNEL_TASK_H

#include <Python.h>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <nlohmann/json.hpp>
#include "tensor_engine/fusion_types.h"
#include "inc/te_fusion_types.h"


namespace te {
namespace fusion {

using SuperKernelTaskMap =
    std::unordered_map<std::string, std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>>;

class SuperKernelTaskManager {
public:
    static SuperKernelTaskManager &GetInstance();

    OpBuildResCode BuildSuperKernel(OpBuildTaskPtr &opTask);
    bool SuperKernelTask(OpBuildTaskPtr &opTask);
    void GetPendingTask(const std::string &uniqueName,
        std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>> *&pMatch);
    int64_t NormalizeEventId(int64_t curEventId, int64_t &eventCnt,
        std::unordered_map<int64_t, int64_t> &uniqEventIdMapping) const;
    void GetCVCntFromNode(const ge::Node *node, int64_t &maxAicCnt, int64_t &maxVecCnt) const;
    static bool CanReuseCache(const std::string &kernelName, SuperKernelTaskMap &spkMap, const OpBuildTaskPtr &opTask);
    /*
     * @brief: clear instance source when session end
     */
    void Finalize();
    SuperKernelTaskManager()
    {
    }

    ~SuperKernelTaskManager()
    {
    }

    mutable std::mutex mtx_;
    SuperKernelTaskMap superKernelMap_;
};

}
}

#endif