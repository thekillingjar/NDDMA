/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_binary_resource_manager.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/def_types.h"

namespace nnopbase {
namespace {
ge::graphStatus GetStr(const std::tuple<const uint8_t*, const uint8_t*> &input, std::string &str)
{
  const uint8_t *start = std::get<0U>(input);
  const uint8_t *end = std::get<1U>(input);
  if ((end < start) || (start == nullptr) || (end == nullptr)) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Parse json failed, end is %p, start is %p!", end, start);
    return ge::GRAPH_PARAM_INVALID;
  }

  str = std::string(start, end);
  GELOGD("Parse str addr is %p, len is %zu, %s.", start, ge::PtrToValue(end) - ge::PtrToValue(start), str.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseJson(const std::tuple<const uint8_t*, const uint8_t*> &input, nlohmann::json &res)
{
  std::string jsonStr;
  GE_ASSERT_GRAPH_SUCCESS(GetStr(input, jsonStr));
  try {
    res = nlohmann::json::parse(jsonStr);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Parse json failed, resion %s, json info %s.", e.what(), jsonStr.c_str());
    return ge::GRAPH_PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseBinary(const std::tuple<const uint8_t*, const uint8_t*> &input, Binary &binaryInfo)
{
  const uint8_t *start = std::get<0U>(input);
  const uint8_t *end = std::get<1U>(input);
  if ((end < start) || (start == nullptr) || (end == nullptr)) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Parse json failed, end is %p, start is %p!", end, start);
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t len = ge::PtrToValue(end) - ge::PtrToValue(start);
  if (len > static_cast<size_t>(UINT32_MAX)) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Parse json failed, binary len %zu is larger than uint32_max.", len);
    return ge::GRAPH_PARAM_INVALID;
  }
  binaryInfo.len = static_cast<uint32_t>(len);
  binaryInfo.content = start;
  GELOGD("Parse binary info, addr is %p, len is %us.", binaryInfo.content, binaryInfo.len);
  return ge::GRAPH_SUCCESS;
}
} // namepsace

void OpBinaryResourceManager::AddOpFuncHandle(const ge::AscendString &opType,
                                              const std::vector<void *> &opResourceHandle)
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = resourceHandle_.find(opType);
  if (it != resourceHandle_.end()) {
    return;
  }
  GELOGI("Add op %s func handle, num is %zu!", opType.GetString(), opResourceHandle.size());
  for (auto func : opResourceHandle) {
    (void)resourceHandle_[opType].emplace_back(func);
  }
}

// 首个信息一定存在，是算子描述json，后续是成对的二进制信息
ge::graphStatus OpBinaryResourceManager::AddBinary(const ge::AscendString &opType,
    const std::vector<std::tuple<const uint8_t*, const uint8_t*>> &opBinary)
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = opBinaryDesc_.find(opType);
  if (it != opBinaryDesc_.end()) {
    return ge::GRAPH_SUCCESS;
  }

  // 首个信息是op的描述信息
  if (opBinary.size() >= 1U) {
    nlohmann::json opDesc;
    GE_ASSERT_GRAPH_SUCCESS(ParseJson(opBinary[0], opDesc), "Parse op %s json failed!", opType.GetString());
    opBinaryDesc_[opType] = opDesc;
  }

  size_t i = 1U;
  while (i + 1U < opBinary.size()) {
    nlohmann::json binaryDesc;
    Binary binaryInfo;
    GE_ASSERT_GRAPH_SUCCESS(ParseJson(opBinary[i], binaryDesc), "Parse op %s binary json file [%zu] failed!",
                            opType.GetString(), i / 2U); // 2 for idx
    GE_ASSERT_GRAPH_SUCCESS(ParseBinary(opBinary[i + 1U], binaryInfo), "Parse op %s binary file [%zu] failed!",
                            opType.GetString(), i / 2U); // 2 for idx

    std::string filePath;
    try {
      filePath = binaryDesc["filePath"].get<std::string>();
    } catch (const nlohmann::json::exception &e) {
      GELOGE(ge::GRAPH_PARAM_INVALID, "Parse op %s json filePath from binary json failed, reason %s.",
             opType.GetString(), e.what());
      return ge::GRAPH_PARAM_INVALID;
    }
    pathToBinary_[filePath.c_str()] = std::tuple<nlohmann::json, Binary>(binaryDesc, binaryInfo);
    GELOGI("Add op %s binary, filePath %s, bin addr is %p, bin len %u.", opType.GetString(), filePath.c_str(),
           binaryInfo.content, binaryInfo.len);

    std::vector<std::string> keys;
    try {
      auto supportInfo = binaryDesc["supportInfo"];
      keys = supportInfo["simplifiedKey"].get<std::vector<std::string>>();
    } catch (const nlohmann::json::exception &e) {
      GELOGW("Get op %s json simplifiedKey from binary json failed, reason %s.", opType.GetString(), e.what());
    }
    for (auto key : keys) {
      GELOGI("Add op %s binary, simplifiedKey %s, filePath %s, bin addr %p, bin len %u.", opType.GetString(),
             key.c_str(), filePath.c_str(), binaryInfo.content, binaryInfo.len);
      keyToPath_[key.c_str()] = filePath.c_str();
    }

    i += 2U; // 2 for json & binary
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpBinaryResourceManager::AddRuntimeKB(const ge::AscendString &opType,
    const std::vector<std::tuple<const uint8_t*, const uint8_t*>> &opRuntimeKb)
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = runtimeKb_.find(opType);
  if (it != runtimeKb_.end()) {
    return ge::GRAPH_SUCCESS;
  }
  for (const auto &kbInfo : opRuntimeKb) {
    std::string kbStr;
    GE_ASSERT_GRAPH_SUCCESS(GetStr(kbInfo, kbStr), "Parse op %s runtime kb json file!", opType.GetString());
    (void)runtimeKb_[opType].emplace_back(kbStr.c_str());
  }
  GELOGI("Add op %s runtime kb num %zu!", opType.GetString(), runtimeKb_[opType].size());
  return ge::GRAPH_SUCCESS;
}

const std::map<const ge::AscendString, nlohmann::json> &OpBinaryResourceManager::GetAllOpBinaryDesc() const
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  GELOGI("Get all op binary desc, num is %zu.", opBinaryDesc_.size());
  return opBinaryDesc_;
}

ge::graphStatus OpBinaryResourceManager::GetOpBinaryDesc(const ge::AscendString &opType, nlohmann::json &binDesc) const
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = opBinaryDesc_.find(opType);
  if (it == opBinaryDesc_.end()) {
    // 返回错误码表示该optype不存在，但不打印error日志，可以在执行时调用，根据返回判断是否存在静态二进制
    GELOGW("Get op %s json info failed!", opType.GetString());
    return ge::GRAPH_PARAM_INVALID;
  }
  binDesc = it->second;
  GELOGI("Get op %s binary desc.", opType.GetString());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpBinaryResourceManager::GetOpBinaryDescByPath(const ge::AscendString &jsonFilePath,
                                                               std::tuple<nlohmann::json, Binary> &binInfo) const
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = pathToBinary_.find(jsonFilePath);
  if (it == pathToBinary_.end()) {
    GELOGW("Get binaryInfo by json path failed, path is %s.", jsonFilePath.GetString());
    return ge::GRAPH_PARAM_INVALID;
  }
  binInfo = it->second;
  GELOGI("Get binary info, json path is %s.", jsonFilePath.GetString());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpBinaryResourceManager::GetOpBinaryDescByKey(const ge::AscendString &simplifiedKey,
                                                              std::tuple<nlohmann::json, Binary> &binInfo) const
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = keyToPath_.find(simplifiedKey);
  const auto &simplified = simplifiedKey;
  if (it == keyToPath_.end()) {
    GELOGW("Get binaryInfo by simplified failed, simplified is %s.", simplified.GetString());
    return ge::GRAPH_PARAM_INVALID;
  }
  GELOGI("Get binary info, simplified is %s.", simplified.GetString());
  return GetOpBinaryDescByPath(it->second, binInfo);
}

ge::graphStatus OpBinaryResourceManager::GetOpRuntimeKB(const ge::AscendString &opType,
                                                        std::vector<ge::AscendString> &kbList) const
{
  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &it = runtimeKb_.find(opType);
  if (it == runtimeKb_.end()) {
    GELOGW("Get op %s RuntimeKB info failed.", opType.GetString());
    return ge::GRAPH_PARAM_INVALID;
  }
  kbList = it->second;
  GELOGI("Get op %s RuntimeKB info, num is %zu.", opType.GetString(), kbList.size());
  return ge::GRAPH_SUCCESS;
}
} // nnopbase
