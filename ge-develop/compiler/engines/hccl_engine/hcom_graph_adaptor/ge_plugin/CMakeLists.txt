# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
target_sources(hcom_graph_adaptor  PRIVATE
    plugin_manager.cc
    hcom_plugin.cc
)
target_sources(hcom_executor  PRIVATE
    plugin_manager.cc
    hcom_plugin.cc
)
add_subdirectory(common)
add_subdirectory(hcom)
add_subdirectory(auto_tuning)
add_subdirectory(calculation_tool)

