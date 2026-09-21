/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_tiling_registry.h"
#include "framework/common/debug/ge_log.h"

namespace optiling {
size_t ByteBufferGetAll(ByteBuffer &buf, ge::char_t *dest, size_t dest_len) {
  size_t nread = 0;
  size_t rn = 0;
  do {
    rn = static_cast<size_t>(buf.readsome(dest + nread, static_cast<std::streamsize>(dest_len - nread)));
    nread += rn;
  } while ((rn > 0) && (dest_len > nread));

  return nread;
}

ByteBuffer &ByteBufferPut(ByteBuffer &buf, const uint8_t *data, size_t data_len) {
  (void)buf.write(reinterpret_cast<const ge::char_t *>(data), static_cast<std::streamsize>(data_len));
  (void)buf.flush();
  return buf;
}

std::unordered_map<std::string, OpTilingFunc> &OpTilingRegistryInterf::RegisteredOpInterf() {
  static std::unordered_map<std::string, OpTilingFunc> interf;
  return interf;
}

OpTilingRegistryInterf::OpTilingRegistryInterf(std::string op_type, OpTilingFunc func) {
  auto &interf = RegisteredOpInterf();
  (void)interf.emplace(op_type, func);
  GELOGI("Register tiling function: op_type:%s, funcPointer:%p, registered count:%zu", op_type.c_str(),
         func.target<OpTilingFuncPtr>(), interf.size());
}

std::unordered_map<std::string, OpTilingFuncV2> &OpTilingRegistryInterf_V2::RegisteredOpInterf() {
  static std::unordered_map<std::string, OpTilingFuncV2> interf;
  GELOGI("Generated interface by new method, registered count: %zu", interf.size());
  return interf;
}

OpTilingRegistryInterf_V2::OpTilingRegistryInterf_V2(const std::string &op_type, OpTilingFuncV2 func) {
  auto &interf = RegisteredOpInterf();
  (void)interf.emplace(op_type, std::move(func));
  GELOGI("Registering tiling function with new method: op_type=%s, registered_count=%zu", op_type.c_str(), interf.size());
}

OpTilingFuncInfo::OpTilingFuncInfo(const std::string &op_type)
    : op_type_(op_type),
      tiling_func_(nullptr),
      tiling_func_v2_(nullptr),
      tiling_func_v3_(nullptr),
      parse_func_v3_(nullptr) {}

bool OpTilingFuncInfo::IsFunctionV4() {
  return this->tiling_func_v4_ != nullptr && this->parse_func_v4_ != nullptr;
}
bool OpTilingFuncInfo::IsFunctionV3() {
  return this->tiling_func_v3_ != nullptr && this->parse_func_v3_ != nullptr;
}
bool OpTilingFuncInfo::IsFunctionV2() {
  return this->tiling_func_v2_ != nullptr;
}
bool OpTilingFuncInfo::IsFunctionV1() {
  return this->tiling_func_ != nullptr;
}
void OpTilingFuncInfo::SetOpTilingFunc(OpTilingFunc &tiling_func) {
  this->tiling_func_ = tiling_func;
}
void OpTilingFuncInfo::SetOpTilingFuncV2(OpTilingFuncV2 &tiling_func) {
  this->tiling_func_v2_ = tiling_func;
}
void OpTilingFuncInfo::SetOpTilingFuncV3(OpTilingFuncV3 &tiling_func, OpParseFuncV3 &parse_func) {
  this->tiling_func_v3_ = tiling_func;
  this->parse_func_v3_ = parse_func;
}
void OpTilingFuncInfo::SetOpTilingFuncV4(OpTilingFuncV4 &tiling_func, OpParseFuncV4 &parse_func) {
  this->tiling_func_v4_ = tiling_func;
  this->parse_func_v4_ = parse_func;
}
const OpTilingFunc& OpTilingFuncInfo::GetOpTilingFunc() {
  return this->tiling_func_;
}
const OpTilingFuncV2& OpTilingFuncInfo::GetOpTilingFuncV2() {
  return this->tiling_func_v2_;
}
const OpTilingFuncV3& OpTilingFuncInfo::GetOpTilingFuncV3() {
  return this->tiling_func_v3_;
}
const OpParseFuncV3& OpTilingFuncInfo::GetOpParseFuncV3() {
  return this->parse_func_v3_;
}
const OpTilingFuncV4& OpTilingFuncInfo::GetOpTilingFuncV4() {
  return this->tiling_func_v4_;
}
const OpParseFuncV4& OpTilingFuncInfo::GetOpParseFuncV4() {
  return this->parse_func_v4_;
}

std::unordered_map<std::string, OpTilingFuncInfo> &OpTilingFuncRegistry::RegisteredOpFuncInfo() {
  static std::unordered_map<std::string, OpTilingFuncInfo> op_func_map;
  return op_func_map;
}

OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type, OpTilingFunc tiling_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFunc(tiling_func);
    (void)op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFunc(tiling_func);
  }
  GELOGI("Register op tiling function V1 for op_type:%s", op_type.c_str());
}
OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type, OpTilingFuncV2 tiling_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFuncV2(tiling_func);
    (void)op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFuncV2(tiling_func);
  }
  GELOGI("Register op tiling function V2 for op_type:%s", op_type.c_str());
}

OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type,
                                           OpTilingFuncV3 tiling_func, OpParseFuncV3 parse_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFuncV3(tiling_func, parse_func);
    (void)op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFuncV3(tiling_func, parse_func);
  }
  GELOGI("Register op tiling and parse function V3 for op_type:%s", op_type.c_str());
}

OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type,
                                           OpTilingFuncV4 tiling_func, OpParseFuncV4 parse_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFuncV4(tiling_func, parse_func);
    (void)op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFuncV4(tiling_func, parse_func);
  }
  GELOGI("Registering tiling and parsing function V4 for op_type: %s", op_type.c_str());
}
}  // namespace optiling
