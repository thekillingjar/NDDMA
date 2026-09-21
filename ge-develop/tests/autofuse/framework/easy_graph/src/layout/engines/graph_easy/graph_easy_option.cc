/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"

EG_NS_BEGIN

namespace {
struct Format {
  const char *format;
  const char *postfix;
};

std::map<LayoutFormat, Format> formats = {{LayoutFormat::ASCII, {"ascii", "txt"}},
                                          {LayoutFormat::BOXART, {"boxart", "txt"}},
                                          {LayoutFormat::SVG, {"svg", "svg"}},
                                          {LayoutFormat::DOT, {"dot", "dot"}},
                                          {LayoutFormat::HTML, {"html", "html"}}};

std::string GetLayoutOutputArg(const GraphEasyOption &options, const std::string &graphName) {
  if (options.output_ == LayoutOutput::CONSOLE)
    return "";
  return std::string(" --output ") + options.output_path_ + graphName + "." + formats[options.format_].postfix;
}

std::string GetLayoutFomartArg(const GraphEasyOption &options) {
  return std::string(" --as=") + formats[options.format_].format;
}
}  // namespace

const GraphEasyOption &GraphEasyOption::GetDefault() {
  static GraphEasyOption option;
  return option;
}

std::string GraphEasyOption::GetLayoutCmdArgs(const std::string &graphName) const {
  return GetLayoutFomartArg(*this) + GetLayoutOutputArg(*this, graphName);
}

EG_NS_END
