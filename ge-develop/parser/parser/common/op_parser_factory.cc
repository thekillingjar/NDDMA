/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_parser_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils_inner.h"
#include "base/err_msg.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY CustomParserAdapterRegistry *CustomParserAdapterRegistry::Instance() {
  static CustomParserAdapterRegistry instance;
  return &instance;
}

FMK_FUNC_HOST_VISIBILITY void CustomParserAdapterRegistry::Register(const domi::FrameworkType framework,
                                                                    CustomParserAdapterRegistry::CREATOR_FUN fun) {
  if (funcs_.find(framework) != funcs_.end()) {
    GELOGW("Framework type %s has already registed.", TypeUtilsInner::FmkTypeToSerialString(framework).c_str());
    return;
  }
  funcs_[framework] = fun;
  GELOGI("Register %s custom parser adapter success.", TypeUtilsInner::FmkTypeToSerialString(framework).c_str());
  return;
}
FMK_FUNC_HOST_VISIBILITY CustomParserAdapterRegistry::CREATOR_FUN
CustomParserAdapterRegistry::GetCreateFunc(const domi::FrameworkType framework) {
  if (funcs_.find(framework) == funcs_.end()) {
    GELOGW("Framework type %s has not registed.", TypeUtilsInner::FmkTypeToSerialString(framework).c_str());
    return nullptr;
  }
  return funcs_[framework];
}

FMK_FUNC_HOST_VISIBILITY std::shared_ptr<OpParserFactory> OpParserFactory::Instance(
    const domi::FrameworkType framework) {
  // Each framework corresponds to one op parser factory,
  // If instances are static data members of opparserfactory, the order of their construction is uncertain.
  // Instances cannot be a member of a class because they may be used before initialization, resulting in a run error.
  static std::map<domi::FrameworkType, std::shared_ptr<OpParserFactory>> instances;

  std::map<domi::FrameworkType, std::shared_ptr<OpParserFactory>>::const_iterator iter = instances.find(framework);
  if (iter == instances.end()) {
    std::shared_ptr<OpParserFactory> instance(new (std::nothrow) OpParserFactory());
    if (instance == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "create OpParserFactory failed");
      GELOGE(INTERNAL_ERROR, "[Create][OpParserFactory] failed.");
      return nullptr;
    }
    instances[framework] = instance;
    return instance;
  }

  return iter->second;
}

FMK_FUNC_HOST_VISIBILITY std::shared_ptr<OpParser> OpParserFactory::CreateOpParser(const std::string &op_type) {
  // First look for CREATOR_FUN based on OpType, then call CREATOR_FUN to create OpParser.
  std::map<std::string, CREATOR_FUN>::const_iterator iter = op_parser_creator_map_.find(op_type);
  if (iter != op_parser_creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERR_MSG("E19999", "param op_type invalid, Not supported type: %s", op_type.c_str());
  GELOGE(FAILED, "[Check][Param] OpParserFactory::CreateOpParser: Not supported type: %s", op_type.c_str());
  return nullptr;
}

FMK_FUNC_HOST_VISIBILITY std::shared_ptr<OpParser> OpParserFactory::CreateFusionOpParser(const std::string &op_type) {
  // First look for CREATOR_FUN based on OpType, then call CREATOR_FUN to create OpParser.
  std::map<std::string, CREATOR_FUN>::const_iterator iter = fusion_op_parser_creator_map_.find(op_type);
  if (iter != fusion_op_parser_creator_map_.end()) {
    return iter->second();
  }
  REPORT_INNER_ERR_MSG("E19999", "param op_type invalid, Not supported fusion op type: %s", op_type.c_str());
  GELOGE(FAILED, "[Check][Param] OpParserFactory::CreateOpParser: Not supported fusion op type: %s", op_type.c_str());
  return nullptr;
}

// This function is only called within the constructor of the global opparserregisterar object,
// and does not involve concurrency, so there is no need to lock it
FMK_FUNC_HOST_VISIBILITY void OpParserFactory::RegisterCreator(const std::string &type, CREATOR_FUN fun,
                                                               bool is_fusion_op) {
  std::map<std::string, CREATOR_FUN> *op_parser_creator_map = &op_parser_creator_map_;
  if (is_fusion_op) {
    op_parser_creator_map = &fusion_op_parser_creator_map_;
  }

  GELOGD("OpParserFactory::RegisterCreator: op type:%s, is_fusion_op:%d.", type.c_str(), is_fusion_op);
  (*op_parser_creator_map)[type] = fun;
}

FMK_FUNC_HOST_VISIBILITY bool OpParserFactory::OpParserIsRegistered(const std::string &op_type, bool is_fusion_op) {
  if (is_fusion_op) {
    std::map<std::string, CREATOR_FUN>::const_iterator iter = fusion_op_parser_creator_map_.find(op_type);
    if (iter != fusion_op_parser_creator_map_.end()) {
      return true;
    }
  } else {
    std::map<std::string, CREATOR_FUN>::const_iterator iter = op_parser_creator_map_.find(op_type);
    if (iter != op_parser_creator_map_.end()) {
      return true;
    }
  }
  return false;
}
}  // namespace ge
