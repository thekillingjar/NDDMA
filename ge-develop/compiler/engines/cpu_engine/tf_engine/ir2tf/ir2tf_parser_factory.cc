/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir2tf_parser_factory.h"
#include "common/util/log.h"

namespace aicpu {
Ir2tfParserFactory& Ir2tfParserFactory::Instance()
{
  static Ir2tfParserFactory instance;
  return instance;
}

std::shared_ptr<Ir2tfBaseParser> Ir2tfParserFactory::CreateIRParser(const std::string &op_type)
{
  auto iter = creator_map_.find(op_type);
  if (iter != creator_map_.end()) {
    return iter->second();
  }
  AICPUE_LOGI("IR2TFParserFactory::CreateIRParser: No matched parser: [%s], use base parser.",
              op_type.c_str());
  return Ir2tfBaseParser::Instance();
}

void Ir2tfParserFactory::RegisterCreator(const std::string &op_type, CREATOR_FUN fun)
{
  if (creator_map_.find(op_type) != creator_map_.end()) {
    AICPUE_LOGW("Ir2tfBaseParser::RegisterCreator: [%s] creator already exist, it will be covered.",
                op_type.c_str());
  }
  creator_map_[op_type] = fun;
}
}
