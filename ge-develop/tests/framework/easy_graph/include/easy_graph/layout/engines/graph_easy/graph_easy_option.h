/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H37156CC2_92BD_44DA_8DA7_A11629E762BE
#define H37156CC2_92BD_44DA_8DA7_A11629E762BE

#include "easy_graph/layout/layout_option.h"
#include <string>

EG_NS_BEGIN

enum class FlowDir {
  LR = 0,
  TB,
};

enum class LayoutType {
  FREE = 0,
  REGULAR,
};

enum class LayoutFormat {
  ASCII = 0,
  BOXART,
  SVG,
  DOT,
  HTML,
};

enum class LayoutOutput {
  CONSOLE = 0,
  FILE,
};

struct GraphEasyOption : LayoutOption {
  static const GraphEasyOption &GetDefault();

  std::string GetLayoutCmdArgs(const std::string &graphName) const;

  LayoutFormat format_{LayoutFormat::BOXART};
  LayoutOutput output_{LayoutOutput::CONSOLE};
  FlowDir dir_{FlowDir::LR};
  LayoutType type_{LayoutType::FREE};
  size_t scale_{1};
  std::string output_path_{"./"};
};

EG_NS_END

#endif
