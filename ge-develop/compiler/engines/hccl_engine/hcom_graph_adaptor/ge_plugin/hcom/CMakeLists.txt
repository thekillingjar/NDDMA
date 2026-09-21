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
    hcom_all_reduce_fusion.cc
    hcom_broadcast_fusion.cc
    hcom_reduce_fusion.cc
    hcom_alltoallvc_fusion.cc
    hcom_fusion_optimizer.cc
    hcom_allgather_fusion.cc
    hcom_reducescatter_fusion.cc
    hcom_graph_optimizer.cc
    hcom_ops_kernel_info_store.cc
    offline_build_config_parse.cc
    lowering/hcom_build_graph.cc
    lowering/hcom_launch_kernel.cc
    lowering/hcom_node_converter.cc
    hcom_graph_mc2.cc
    hcom_graph_superkernel.cc
)
target_sources(hcom_executor  PRIVATE
    hcom_ops_kernel_info_store.cc
    offline_build_config_parse.cc
)
target_sources(hcom_opskernel_builder PRIVATE
    hcom_ops_kernel_builder.cc
    offline_build_config_parse.cc
    hcom_graph_mc2.cc
)
target_sources(hcom_gradtune_opskernel_builder PRIVATE
    offline_build_config_parse.cc
)