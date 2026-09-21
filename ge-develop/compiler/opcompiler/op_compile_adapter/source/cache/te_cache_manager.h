/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_CACHE_TE_CACHE_MANAGER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_CACHE_TE_CACHE_MANAGER_H_

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
class TeCacheManager {
public:
    static TeCacheManager& Instance();
    bool Initialize();
    bool IsCacheEnable() const;
    CompileResultPtr MatchCompileCache(const std::string &kernelName, bool spkMode);
    bool SetCompileResult(const CompileResultPtr &compileResultPtr);
    PreCompileResultPtr MatchPreCompileCache(const std::string &kernelName);
    bool SetPreCompileResult(const std::string &kernelName, const PreCompileResultPtr &preCompileResultPtr);
    void CheckAndMakeCacheDir();
    void SetOpArgsCache(const std::string &key_name, const std::string &op_args);
    bool GetOpArgsCache(const std::string &key_name, std::string &op_args);
    void ClearOpArgsCache(const std::string &session_graph_id);
    void Finalize();

private:
    TeCacheManager() : cache_mode_(CompileCacheMode::Disable) {}
    ~TeCacheManager()
    {
        cache_mode_ = CompileCacheMode::Disable;
    }

    CompileResultPtr GetCompileResultFromCacheMap(const std::string &kernelName);
    void SetCompileResultIntoCacheMap(const std::string &kernelName, const CompileResultPtr &compileResultPtr);
    PreCompileResultPtr GetPreCompileResultFromCacheMap(const std::string &kernelName);
    void SetPreCompileResultIntoCacheMap(const std::string &kernelName, const PreCompileResultPtr &preCompileResultPtr);
    static bool CopyCompileRetIntoCacheDir(const CompileResultPtr &compileResultPtr,
                                           const std::string &cacheJsonFilePath, const std::string &cacheBinFilePath);
    static bool SavePreCompileRetIntoCacheDir(const PreCompileResultPtr &preCompileResultPtr,
                                              const std::string &jsonCachePath);

    static bool VerifyBinFileSha256(const CompileResultPtr &compileResultPtr);

    bool SetCacheDirPath();

private:
    CompileCacheMode cache_mode_;
    mutable std::mutex cache_param_mutex_;
    std::string cache_dir_path_;
    std::string cache_parent_dir_path_;
    mutable std::mutex compile_cache_mutex_;
    std::map<std::string, CompileResultPtr> compile_ret_cache_map_;
    mutable std::mutex precompile_cache_mutex_;
    std::map<std::string, PreCompileResultPtr> precompile_ret_cache_map_;
    mutable std::mutex op_args_mutex_;
    std::map<std::string, std::string> op_args_cache_map_;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_CACHE_TE_CACHE_MANAGER_H_