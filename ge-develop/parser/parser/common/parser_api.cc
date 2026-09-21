/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/omg/parser/parser_api.h"

#include "common/util.h"
#include "common/acl_graph_parser_util.h"
#include "tbe_plugin_loader.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_registration_tbe.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "ge/ge_api_types.h"

namespace ge {
static bool parser_initialized = false;
// Initialize PARSER, load custom op plugin
// options will be used later for parser decoupling
Status ParserInitializeImpl(const std::map<std::string, std::string> &options) {
  GELOGT(TRACE_INIT, "ParserInitialize start");
  // check init status
  if (parser_initialized) {
    GELOGW("ParserInitialize is called more than once");
    return SUCCESS;
  }
  AclGraphParserUtil acl_graph_parse_util;
  if (acl_graph_parse_util.AclParserInitialize(options, true) != domi::SUCCESS) {
    GELOGE(GRAPH_FAILED, "Parser Initialize failed.");
    return GRAPH_FAILED;
  }
  auto iter = options.find(ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES);
  if (iter != options.end()) {
    ge::GetParserContext().enable_scope_fusion_passes = iter->second;
    ge::parser::GetParserContextInstance().enable_scope_fusion_passes = iter->second;
  }
  // set init status
  if (!parser_initialized) {
    // Initialize success, first time calling initialize
    parser_initialized = true;
  }

  GELOGT(TRACE_STOP, "ParserInitialize finished");
  return SUCCESS;
}

// Initialize PARSER, load custom op plugin
// options will be used later for parser decoupling
Status ParserInitialize(const std::map<std::string, std::string> &options) {
  return ParserInitializeImpl(options);
}

Status ParserInitialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  std::map<std::string, std::string> str_options;
  for (const auto &option : options) {
    const std::string &key =
        std::string(option.first.GetString(), option.first.GetLength());
    const std::string &val =
        std::string(option.second.GetString(), option.second.GetLength());
    str_options[key] = val;
  }
  return ParserInitializeImpl(str_options);
}

Status ParserFinalize() {
  GELOGT(TRACE_INIT, "ParserFinalize start");
  // check init status
  if (!parser_initialized) {
    GELOGW("ParserFinalize is called before ParserInitialize");
    return SUCCESS;
  }

  GE_CHK_STATUS(TBEPluginLoader::Instance().Finalize());
  if (parser_initialized) {
    parser_initialized = false;
  }
  return SUCCESS;
}
}  // namespace ge
