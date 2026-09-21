/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "superkernel_task.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_util_constants.h"
#include "compile/fusion_manager.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_config_info.h"
#include "python_adapter/python_adapter_manager.h"
#include "python_adapter/py_object_utils.h"

namespace te {
namespace fusion {

SuperKernelTaskManager &SuperKernelTaskManager::GetInstance()
{
    static SuperKernelTaskManager managerInstance;
    return managerInstance;
}

void SuperKernelTaskManager::Finalize()
{
    superKernelMap_.clear();
}

OpBuildResCode SuperKernelTaskManager::BuildSuperKernel(OpBuildTaskPtr &opTask)
{
    TE_DBGLOG("start to call BuildSuperKernel, taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
    TE_FUSION_TIMECOST_START(BuildSuperKernel);
    const std::lock_guard<std::mutex> spkLock(mtx_);
    bool res = SuperKernelTask(opTask);
    TE_FUSION_TIMECOST_END(BuildSuperKernel, "BuildSuperKernel");
    if (!res) {
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    }
    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_DBGLOG("Get TeFusionManager lock");
    TeFusionManager *teFusionM = TeFusionManager::GetInstance();
    res = teFusionM->SaveBuildTask(opTask);
    if (!res) {
        TE_ERRLOG("Save super kernel task failed. taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
        return OP_BUILD_FAIL;
    }
    TE_DBGLOG("Success to call BuildSuperKernel, taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
    return OP_BUILD_SUCC;
}

void SuperKernelTaskManager::GetPendingTask(const std::string &uniqueName,
    std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>> *&pMatch)
{
    auto opIter = superKernelMap_.find(uniqueName);
    if (opIter != superKernelMap_.end()) {
        pMatch = &opIter->second;
        TE_INFOLOG("Get super kernel finished task from cached task list");
    }
}

bool SuperKernelTaskManager::CanReuseCache(const std::string &kernelName, SuperKernelTaskMap &spkMap,
    const OpBuildTaskPtr &opTask)
{
    auto opIter = spkMap.find(kernelName);
    if (opIter == spkMap.end()) {
        return false;
    }

    auto &match = opIter->second;
    if (match.first == nullptr) {
        // kernel not generated yet, there's another equal task running background.
        // save current task to the queue, when previous equal task finished, all tasks in this
        // queue became finished
        if (match.second.size() <= 0) {
            TE_WARNLOG("No matched task. Size = %zu.", match.second.size());
            return false;
        }
        TE_DBGLOG("Op[%s] background build task running[%lu],[%s] enqueue taskID[%lu:%lu] [%s]",
                  kernelName.c_str(), match.second.size(), match.second[0]->kernel.c_str(),
                  opTask->graphId, opTask->taskId, opTask->kernel.c_str());

        // set current op's kernel name
        opTask->kernel = match.second[0]->kernel;
        opTask->status = OP_TASK_STATUS::OP_TASK_PENDING;
        match.second.push_back(opTask);
        return true;
    }

    opTask->opRes = match.first;
    TE_DBGLOG("Get same superkernel op, taskID[%lu:%lu], opRes->result is %s",
              opTask->graphId, opTask->taskId, opTask->opRes->result.c_str());
    if (opTask->opRes->statusCode != 0) {
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    } else {
        opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
    }
    return true;
}

int64_t SuperKernelTaskManager::NormalizeEventId(int64_t curEventId, int64_t &eventCnt,
    std::unordered_map<int64_t, int64_t> &uniqEventIdMapping) const
{
    if (uniqEventIdMapping.count(curEventId) > 0) {
        return uniqEventIdMapping[curEventId];
    }
    uniqEventIdMapping[curEventId] = eventCnt;
    ++eventCnt;
    return uniqEventIdMapping[curEventId];
}

void SuperKernelTaskManager::GetCVCntFromNode(const ge::Node *node, int64_t &maxAicCnt, int64_t &maxVecCnt) const {
    std::string custAicNum;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kAicCntKeyOp, custAicNum);
    if (custAicNum != "" && std::stoi(custAicNum) >= 0) {
        int64_t tmpAicCnt = std::stoi(custAicNum);
        maxAicCnt = tmpAicCnt > maxAicCnt ? tmpAicCnt : maxAicCnt;
        TE_DBGLOG("[SuperKernel] OpNode: %s, get ai_core_cnt: %s", node->GetName().c_str(), custAicNum.c_str());
    }
    std::string custAivNum;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kAivCntKeyOp, custAivNum);
    if (custAivNum != "" && std::stoi(custAivNum) >= 0) {
        int64_t tmpVecCnt = std::stoi(custAivNum);
        maxVecCnt = tmpVecCnt > maxVecCnt ? tmpVecCnt : maxVecCnt;
        TE_DBGLOG("[SuperKernel] OpNode: %s, get vector_core_cnt: %s", node->GetName().c_str(), custAivNum.c_str());
    }
}

bool SuperKernelTaskManager::SuperKernelTask(OpBuildTaskPtr &opTask)
{
    json opListJson;
    json superKernelInfo;
    json normalizedJson;
    std::unordered_map<int64_t, int64_t> uniqEventIdMapping;
    int64_t eventCnt = 0;

    std::map<std::string, std::string> skOptions;
    TeConfigInfo::Instance().GetHardwareInfoMap(skOptions);

    int64_t skOpAicCnt = 0;
    int64_t skOpAivCnt = 0;
    for (const auto &currentNode : opTask->opNodes) {
        std::string keyName;
        bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(currentNode, keyName);
        TE_FUSION_CHECK(!bres || keyName.empty(), {
            TE_ERRLOG("Failed to get node key name.");
            return false;
        });
        bool isHcclOp = false;
        (void)ge::AttrUtils::GetBool(currentNode->GetOpDesc(), "_hccl", isHcclOp);
        ConstTbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(currentNode);
        if (opInfo == nullptr && !isHcclOp) {
            TE_ERRLOG("Node[%s] get tbe op information by node failed.", currentNode->GetName().c_str());
            return false;
        }
        if (!isHcclOp) {
            GetCVCntFromNode(currentNode, skOpAicCnt, skOpAivCnt);
        }
        // record currentNode info in json
        json currentNodeJson;
        json curOpNormalizedJson;
        std::string tmpKernelName;
        auto currentNodeOpDesc = currentNode->GetOpDesc();
        (void)ge::AttrUtils::GetStr(currentNodeOpDesc, SPK_KERNELNAME, tmpKernelName);
        std::string jsonFilePath;
        std::string binFilePath;
        std::string subKernelName = tmpKernelName;
        std::size_t pos = tmpKernelName.find("__kernel");
        if (pos != std::string::npos) {
            subKernelName = tmpKernelName.substr(0, pos);
        }
        (void)ge::AttrUtils::GetStr(currentNodeOpDesc, "json_file_path", jsonFilePath);
        (void)ge::AttrUtils::GetStr(currentNodeOpDesc, "bin_file_path", binFilePath);
        TE_DBGLOG("Get json file path [%s], bin file path [%s].", jsonFilePath.c_str(), binFilePath.c_str());
        if (jsonFilePath.empty() || binFilePath.empty()) {
            TE_ERRLOG("Failed to get compile res by kernelname[%s].", subKernelName.c_str());
            return false;
        }
        curOpNormalizedJson["sub_kernel_hash"] = tmpKernelName.empty() ? binFilePath : tmpKernelName;
        currentNodeJson[BIN_PATH] = binFilePath;
        currentNodeJson[JSON_PATH] = jsonFilePath;
        std::string taskType = "normal";
        (void)ge::AttrUtils::GetStr(currentNodeOpDesc, "SPK_task_type", taskType);
        currentNodeJson["task_type"] = taskType;
        curOpNormalizedJson["task_type"] = taskType;
        int64_t opStreamId = currentNodeOpDesc->GetStreamId();
        currentNodeJson["stream_id"] = opStreamId;
        std::vector<int64_t> receive_tasks;
        (void)ge::AttrUtils::GetListInt(currentNodeOpDesc, "_sk_rcv_event_ids", receive_tasks);
        json rcvList;
        json rcvListNormalized;
        for (auto &i : receive_tasks) {
            rcvList.push_back(i);
            rcvListNormalized.push_back(NormalizeEventId(i, eventCnt, uniqEventIdMapping));
        }
        if (!receive_tasks.empty()) {
            currentNodeJson["recv_event_list"] = rcvList;
            curOpNormalizedJson["recv_event_list"] = rcvListNormalized;
        }
        std::vector<int64_t> send_tasks;
        (void)ge::AttrUtils::GetListInt(currentNodeOpDesc, "_sk_send_event_ids", send_tasks);
        json sendList;
        json sendListNormalized;
        for (auto &i : send_tasks) {
            sendList.push_back(i);
            sendListNormalized.push_back(NormalizeEventId(i, eventCnt, uniqEventIdMapping));
        }
        if (!send_tasks.empty()) {
            currentNodeJson["send_event_list"] = sendList;
            curOpNormalizedJson["send_event_list"] = sendListNormalized;
        }
        opListJson.push_back(currentNodeJson);
        normalizedJson.push_back(curOpNormalizedJson);
    }
    if (skOpAicCnt > 0 && skOpAivCnt > 0) {
        skOptions[kAiCoreCnt] = std::to_string(skOpAicCnt);
        skOptions[kCubeCoreCnt] = std::to_string(skOpAicCnt);
        skOptions[kVectorCoreCnt] = std::to_string(skOpAivCnt);
    }
    superKernelInfo[OP_LIST] = opListJson;
    std::string superKernelName;
    std::string tmpTaskInfo = normalizedJson.dump();
    if (!PythonApiCall::Instance().GenerateStrSha256HashValue(tmpTaskInfo, superKernelName)) {
        TE_ERRLOGF("taskID[%lu:%lu] do not generate kernel name hash by json(%s).",
            opTask->graphId, opTask->taskId, tmpTaskInfo.c_str());
        return false;
    }
    std::string kernelName = SUPERKERNEL_PREFIX + superKernelName;
    opTask->kernel = kernelName;
    opTask->superKernelUniqueKey = superKernelName;
    superKernelInfo[KERNEL_NAME] = kernelName;

    std::string spOptions;
    (void)ge::AttrUtils::GetStr(opTask->opNodes[0]->GetOpDesc(), SPK_OPTIONS, spOptions);
    if (!spOptions.empty()) {
        superKernelInfo[ASCENDC_SPK_OPTIONS] = spOptions;
    }

    opTask->jsonStr = superKernelInfo.dump();
    TE_INFOLOGF("taskID[%lu:%lu]: SuperKernel[%s] compile json is %s, normalized json is %s",
                opTask->graphId, opTask->taskId, kernelName.c_str(), opTask->jsonStr.c_str(), tmpTaskInfo.c_str());
    PyLockGIL pyLockGIL;
    skOptions["op_debug_dir"] = TeConfigInfo::Instance().GetKernelMetaParentDir();
    skOptions["device_id"] = TeConfigInfo::Instance().GetDeviceId();
    skOptions["soc_version"] = TeConfigInfo::Instance().GetShortSocVersion();
    PyObject *optionValues = PyObjectUtils::GenSuperKernelOptionsInfo(skOptions);
    TE_FUSION_CHECK(optionValues == nullptr, {
        TE_ERRLOG("Failed to generate fusion options info.");
        return false;
    });
    AUTO_PY_DECREF(optionValues);
    if (CanReuseCache(superKernelName, superKernelMap_, opTask)) {
        TE_INFOLOG("Superkernel cache found, node name=[%s], kernel name=[%s], taskID[%lu:%lu].",
                    opTask->opNodes[0]->GetName().c_str(), opTask->kernel.c_str(), opTask->graphId, opTask->taskId);
        return true;
    }
    TE_INFOLOG("Superkernel compile task dispatched, kernel_name: %s.", superKernelName.c_str());
    bool res = PythonAdapterManager::Instance().BuildOpAsync(opTask, "dispatch_super_kernel_task", "sO",
                                                             opTask->jsonStr.c_str(), optionValues);
    std::string superKernelUniqueKey = opTask->superKernelUniqueKey;
    if (res && !superKernelUniqueKey.empty()) {
        superKernelMap_.emplace(superKernelUniqueKey,
                                std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTask})));
        TE_INFOLOG("Super kernel task dispatched, taskID[%lu:%lu]. kernel name=[%s].",
                   opTask->graphId, opTask->taskId, opTask->kernel.c_str());
        return true;
    }
    return false;
}
}
}