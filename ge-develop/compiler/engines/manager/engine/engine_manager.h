/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PLUGIN_ENGINE_ENGINE_MANAGE_H_
#define GE_PLUGIN_ENGINE_ENGINE_MANAGE_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/engine/dnnengine.h"

namespace ge {
using DNNEnginePtr = std::shared_ptr<DNNEngine>;
class GE_FUNC_VISIBILITY EngineManager {
 public:
  static Status RegisterEngine(const std::string &engine_name, DNNEnginePtr engine_ptr);
  static DNNEnginePtr GetEngine(const std::string &engine_name);
  static std::unique_ptr<std::map<std::string, DNNEnginePtr>> engine_map_;
};
void RegisterAiCoreEngine();
void RegisterVectorEngine();
void RegisterDsaEngine();
void RegisterDvppEngine();
void RegisterAiCpuEngine();
void RegisterAiCpuTFEngine();
void RegisterAICpuAscendFftsPlusEngine();
void RegisterAICpuFftsPlusEngine();
void RegisterGeLocalEngine();
void RegisterHostCpuEngine();
void RegisterRtsEngine();
void RegisterRtsFftsPlusEngine();
void RegisterHcclEngine();
void RegisterFftsPlusEngine();
extern "C" {
GE_FUNC_VISIBILITY void GetDNNEngineObjs(std::map<std::string, DNNEnginePtr> &engines);
}
}  // namespace ge
#endif  // GE_PLUGIN_ENGINE_ENGINE_MANAGE_H_
