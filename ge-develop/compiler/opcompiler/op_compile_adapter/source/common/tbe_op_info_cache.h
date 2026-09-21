/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TBEOPINFO_CACHE_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TBEOPINFO_CACHE_H_

#include <map>
#include <string>
#include <mutex>
#include <algorithm>
#include "tensor_engine/tbe_op_info.h"

namespace te {
namespace fusion {
class TbeOpInfoCache {
public:
    TbeOpInfoCache(const TbeOpInfoCache &) = delete;
    TbeOpInfoCache &operator=(const TbeOpInfoCache &) = delete;
    void FinalizeSessionInfo(const std::string &session_graph_id);
    static TbeOpInfoCache& Instance();

    bool SetTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr);
    void UpdateTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr);
    ConstTbeOpInfoPtr GetTbeOpInfo(const std::string &opKeyName) const;
    TbeOpInfoPtr MutableTbeOpInfo(const std::string &opKeyName) const;
    ConstTbeOpInfoPtr GetTbeOpInfoByNode(const ge::Node *node) const;
    TbeOpInfoPtr MutableTbeOpInfoByNode(const ge::Node *node) const;
    bool GetOpFuncName(const std::string &opKeyName, std::string &opFuncName) const;
    bool GetOpFuncNameByNode(const ge::Node *node, std::string &opFuncName) const;

    bool SetSecondTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr);
    void UpdateSecondTbeOpInfo(const std::string &opKeyName, const TbeOpInfoPtr &tbeOpInfoPtr);
    ConstTbeOpInfoPtr GetSecondTbeOpInfo(const std::string &opKeyName) const;
    TbeOpInfoPtr MutableSecondTbeOpInfo(const std::string &opKeyName) const;
    ConstTbeOpInfoPtr GetSecondTbeOpInfoByNode(const ge::Node *node) const;
    TbeOpInfoPtr MutableSecondTbeOpInfoByNode(const ge::Node *node) const;
    bool GetTbeOpInfoByName(const std::string &opName, TbeOpInfoPtr &tbeOpInfo) const;
    bool GetTbeOpInfoByNode(const ge::Node *node, TbeOpInfoPtr &tbeOpInfo) const;
    static bool GetOpKeyNameByNode(const ge::Node *node, std::string &opKeyName);
    bool GetOpImplModeByOpNode(const ge::Node *opNode, std::string &implMode) const;

private:
    TbeOpInfoCache();
    ~TbeOpInfoCache();

    mutable std::mutex tbeOpInfoMutex_;
    std::map<std::string, TbeOpInfoPtr> tbeOpInfoMap_;
    mutable std::mutex secondTbeOpInfoMutex_;
    std::map<std::string, TbeOpInfoPtr> secondTbeOpInfoMap_;
};
}  // namespace fusion
}  // namespace te
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TBEOPINFO_CACHE_H_