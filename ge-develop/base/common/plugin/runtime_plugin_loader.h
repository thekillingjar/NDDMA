/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_RUNTIME_PlUGIN_LOADER_H_
#define GE_COMMON_RUNTIME_PlUGIN_LOADER_H_

#include <memory>
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "ge/ge_api_error_codes.h"
#include "common/ge_visibility.h"
#include "common/ge_types.h"
#include "graph/compute_graph.h"

namespace ge {
class VISIBILITY_EXPORT RuntimePluginLoader {
 public:
  static RuntimePluginLoader &GetInstance();

  ge::graphStatus Initialize(const std::string &path_base);

 private:
  std::unique_ptr<ge::PluginManager> plugin_manager_;
};
}  // namespace ge
#endif  // GE_COMMON_RUNTIME_PlUGIN_LOADER_H_