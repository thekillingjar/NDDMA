/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_EXTRA_INFO_CONFIG_H_
#define ATT_EXTRA_INFO_CONFIG_H_
#include <string>
namespace att {
struct ExtraInfoConfig {
  std::string tiling_data_type_name{"TilingData"};
  bool do_api_tiling{false};       // 控制高阶api tiling是否需要生成
  bool do_axes_calc{false};        // 控制外轴、尾轴等逻辑是否需要生成
};
}  // namespace att
#endif // ATT_EXTRA_INFO_CONFIG_H_