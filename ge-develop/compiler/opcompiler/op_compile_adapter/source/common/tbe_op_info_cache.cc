/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_op_info_cache.h"
#include "inc/te_fusion_log.h"
#include "common/fusion_common.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"

namespace te {
namespace fusion {
TbeOpInfoCache::TbeOpInfoCache() {}
TbeOpInfoCache::~TbeOpInfoCache() {}

TbeOpInfoCache& TbeOpInfoCache::Instance()
{
    static TbeOpInfoCache tbeOpInfoCacheInstance;
    return tbeOpInfoCacheInstance;
}

void TbeOpInfoCache::FinalizeSessionInfo(const std::string &session_graph_id) {
    {
        std::lock_guard<std::mutex> lock_guard(tbeOpInfoMutex_);
        for (auto it = tbeOpInfoMap_.begin(); it != tbeOpInfoMap_.end();) {
            if (0 == std::strncmp(it->first.c_str(), session_graph_id.c_str(), std::min(session_graph_id.length(), it->first.length()))) {
                TE_DBGLOG("FinalizeSessionInfo erased tbeopinfo %s", it->first.c_str());
                it = tbeOpInfoMap_.erase(it);
            } else {
                TE_DBGLOG("FinalizeSessionInfo tbeopinfo %s", it->first.c_str());
                it++;
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock_guard(secondTbeOpInfoMutex_);
        for (auto it = secondTbeOpInfoMap_.begin(); it != secondTbeOpInfoMap_.end();) {
            if (0 == std::strncmp(it->first.c_str(), session_graph_id.c_str(), std::min(session_graph_id.length(), it->first.length()))) {
                TE_DBGLOG("FinalizeSessionInfo erased sec tbeopinfo %s", it->first.c_str());
                it = secondTbeOpInfoMap_.erase(it);
            } else {
                TE_DBGLOG("FinalizeSessionInfo sec tbeopinfo %s", it->first.c_str());
                it++;
            }
        }
    }
}

bool TbeOpInfoCache::SetTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr)
{
    std::lock_guard<std::mutex> lock_guard(tbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = tbeOpInfoMap_.find(opKeyName);
    if (iter != tbeOpInfoMap_.cend()) {
        TE_DBGLOG("Tbe op info of [%s] is already existed.", opKeyName.c_str());
        return false;
    }
    tbeOpInfoMap_.emplace(opKeyName, tbeOpInfoPtr);
    return true;
}

void TbeOpInfoCache::UpdateTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr)
{
    std::lock_guard<std::mutex> lock_guard(tbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::iterator iter = tbeOpInfoMap_.find(opKeyName);
    if (iter != tbeOpInfoMap_.end()) {
        TE_DBGLOG("The op info of [%s] already exists and will be replaced.", opKeyName.c_str());
        iter->second = tbeOpInfoPtr;
    } else {
        TE_DBGLOG("Tbe operation information of [%s] is not found and it will be added.", opKeyName.c_str());
        tbeOpInfoMap_.emplace(opKeyName, tbeOpInfoPtr);
    }
}

ConstTbeOpInfoPtr TbeOpInfoCache::GetTbeOpInfo(const std::string &opKeyName) const
{
    std::lock_guard<std::mutex> lock_guard(tbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = tbeOpInfoMap_.find(opKeyName);
    if (iter == tbeOpInfoMap_.cend()) {
        TE_DBGLOG("Tbe op info of [%s] is not found.", opKeyName.c_str());
        return nullptr;
    }
    return iter->second;
}

TbeOpInfoPtr TbeOpInfoCache::MutableTbeOpInfo(const std::string &opKeyName) const
{
    std::lock_guard<std::mutex> lock_guard(tbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = tbeOpInfoMap_.find(opKeyName);
    if (iter == tbeOpInfoMap_.cend()) {
        TE_DBGLOG("Tbe op info of [%s] is not found.", opKeyName.c_str());
        return nullptr;
    }
    return iter->second;
}

ConstTbeOpInfoPtr TbeOpInfoCache::GetTbeOpInfoByNode(const ge::Node *node) const
{
    std::string opKeyName;
    if (!GetOpKeyNameByNode(node, opKeyName)) {
        return nullptr;
    }
    TE_DBGLOG("Op key name is [%s].", opKeyName.c_str());
    return GetTbeOpInfo(opKeyName);
}

TbeOpInfoPtr TbeOpInfoCache::MutableTbeOpInfoByNode(const ge::Node *node) const
{
    std::string opKeyName;
    if (!GetOpKeyNameByNode(node, opKeyName)) {
        return nullptr;
    }
    TE_DBGLOG("Op key name is [%s].", opKeyName.c_str());
    return MutableTbeOpInfo(opKeyName);
}

bool TbeOpInfoCache::SetSecondTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr)
{
    std::lock_guard<std::mutex> lock_guard(secondTbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = secondTbeOpInfoMap_.find(opKeyName);
    if (iter != secondTbeOpInfoMap_.cend()) {
        TE_DBGLOG("Tbe op info of [%s] is already existed.", opKeyName.c_str());
        return false;
    }
    secondTbeOpInfoMap_.emplace(opKeyName, tbeOpInfoPtr);
    return true;
}

void TbeOpInfoCache::UpdateSecondTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr)
{
    std::lock_guard<std::mutex> lock_guard(secondTbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::iterator iter = secondTbeOpInfoMap_.find(opKeyName);
    if (iter != secondTbeOpInfoMap_.end()) {
        TE_DBGLOG("Tbe op info of [%s] already exists and will be replaced.", opKeyName.c_str());
        iter->second = tbeOpInfoPtr;
    } else {
        TE_DBGLOG("The operation information of [%s] is not found and it will be added.", opKeyName.c_str());
        secondTbeOpInfoMap_.emplace(opKeyName, tbeOpInfoPtr);
    }
}

ConstTbeOpInfoPtr TbeOpInfoCache::GetSecondTbeOpInfo(const std::string &opKeyName) const
{
    std::lock_guard<std::mutex> lock_guard(secondTbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = secondTbeOpInfoMap_.find(opKeyName);
    if (iter == secondTbeOpInfoMap_.cend()) {
        TE_DBGLOG("The second gen tbe op info of [%s] is not found.", opKeyName.c_str());
        return nullptr;
    }
    return iter->second;
}

TbeOpInfoPtr TbeOpInfoCache::MutableSecondTbeOpInfo(const std::string &opKeyName) const
{
    std::lock_guard<std::mutex> lock_guard(secondTbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = secondTbeOpInfoMap_.find(opKeyName);
    if (iter == secondTbeOpInfoMap_.cend()) {
        TE_DBGLOG("The second gen tbe op info of [%s] is not found.", opKeyName.c_str());
        return nullptr;
    }
    return iter->second;
}

ConstTbeOpInfoPtr TbeOpInfoCache::GetSecondTbeOpInfoByNode(const ge::Node *node) const
{
    std::string opKeyName;
    if (!GetOpKeyNameByNode(node, opKeyName)) {
        return nullptr;
    }
    TE_DBGLOG("Op key name is [%s].", opKeyName.c_str());
    return GetSecondTbeOpInfo(opKeyName);
}

TbeOpInfoPtr TbeOpInfoCache::MutableSecondTbeOpInfoByNode(const ge::Node *node) const
{
    std::string opKeyName;
    if (!GetOpKeyNameByNode(node, opKeyName)) {
        return nullptr;
    }
    TE_DBGLOG("Op key name is [%s].", opKeyName.c_str());
    return MutableSecondTbeOpInfo(opKeyName);
}

bool TbeOpInfoCache::GetOpFuncName(const std::string &opKeyName, std::string &opFuncName) const
{
    std::lock_guard<std::mutex> lock_guard(tbeOpInfoMutex_);
    const std::map<std::string, TbeOpInfoPtr>::const_iterator iter = tbeOpInfoMap_.find(opKeyName);
    if (iter == tbeOpInfoMap_.cend()) {
        TE_DBGLOG("Tbe op info of [%s] is not found.", opKeyName.c_str());
        return false;
    }
    if (iter->second == nullptr) {
        TE_DBGLOG("Tbe op info of [%s] is null.", opKeyName.c_str());
        return false;
    }
    (void)iter->second->GetFuncName(opFuncName);
    return true;
}

bool TbeOpInfoCache::GetOpFuncNameByNode(const ge::Node *node, std::string &opFuncName) const
{
    ConstTbeOpInfoPtr tbeOpInfo = GetTbeOpInfoByNode(node);
    if (tbeOpInfo == nullptr) {
        return false;
    }
    (void)tbeOpInfo->GetFuncName(opFuncName);
    return true;
}

bool TbeOpInfoCache::GetTbeOpInfoByName(const std::string &opName, TbeOpInfoPtr &tbeOpInfo) const
{
    TbeOpInfoPtr secondOpInfo = MutableSecondTbeOpInfo(opName);
    TbeOpInfoPtr opInfo = MutableTbeOpInfo(opName);
    if (secondOpInfo == nullptr) {
        if (opInfo == nullptr) {
            TE_ERRLOG("Failed to get tbeopinfo. opName[%s] not found in secondaryGeneralizeMap and preops",
                      opName.c_str());
            return false;
        }
        tbeOpInfo = opInfo;
    } else {
        tbeOpInfo = secondOpInfo;
        if (opInfo != nullptr && tbeOpInfo != nullptr) {
            tbeOpInfo->SetOpCoreType(opInfo->GetOpCoreType());
            TE_DBGLOGF("Op[%s] sync update preops coretype to secondaryGeneralizeMap coretype [%s].",
                       opName.c_str(), opInfo->GetOpCoreType().c_str());
        }
    }
    return true;
}

bool TbeOpInfoCache::GetOpKeyNameByNode(const ge::Node *node, std::string &opKeyName)
{
    if (node == nullptr || node->GetOpDesc() == nullptr) {
        TE_WARNLOG("Node or opDesc is null.");
        return false;
    }

    opKeyName = node->GetName();
    ge::ComputeGraphPtr ownerGraph = node->GetOwnerComputeGraph();
    if (ownerGraph == nullptr) {
        TE_WARNLOG("Owner graph of node[%s] is null.", node->GetName().c_str());
        return false;
    }
    std::string sessionGraphId;
    if (ge::AttrUtils::GetStr(ownerGraph, ge::ATTR_NAME_SESSION_GRAPH_ID, sessionGraphId) && !sessionGraphId.empty()) {
        opKeyName = sessionGraphId + "_" + opKeyName;
    } else if (ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_SESSION_GRAPH_ID, sessionGraphId) &&
             !sessionGraphId.empty()) {
        opKeyName = sessionGraphId + "_" + opKeyName;
    }
    return true;
}

bool TbeOpInfoCache::GetTbeOpInfoByNode(const ge::Node *node, TbeOpInfoPtr &tbeOpInfo) const
{
    std::string keyName;
    bool bres = GetOpKeyNameByNode(node, keyName);
    if (!bres) {
        TE_ERRLOG("Failed to get node key name.");
        return false;
    }

    return GetTbeOpInfoByName(keyName, tbeOpInfo);
}

bool TbeOpInfoCache::GetOpImplModeByOpNode(const ge::Node *opNode, std::string &implMode) const
{
    ConstTbeOpInfoPtr pOpInfo = GetTbeOpInfoByNode(opNode);
    if (pOpInfo == nullptr) {
        TE_ERRLOGF("Node %s get tbe opInfo failed.", opNode->GetName().c_str());
        return false;
    }
    implMode = pOpInfo->GetOpImplMode();
    TE_DBGLOG("Get op impl mode[%s].", implMode.c_str());
    return true;
}
}
}
