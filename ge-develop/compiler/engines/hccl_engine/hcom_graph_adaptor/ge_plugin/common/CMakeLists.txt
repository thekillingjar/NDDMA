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
    op_fusion_base.cc
    ops_kernel_info_store_base.cc
    hcom_op_utils.cc
    hcom_base_funs.cc
    hcom_acl_adapter.cc
    hcom_log.cc
    dlhccl_function.cc
    adapter_dlhcclfunc.cc
)

target_sources(hcom_executor  PRIVATE
    ops_kernel_info_store_base.cc
    hcom_op_utils.cc
    hcom_base_funs.cc
    hcom_acl_adapter.cc
    hcom_log.cc
    dlhccl_function.cc
    adapter_dlhcclfunc.cc
)

target_sources(hcom_opskernel_builder PRIVATE
    hcom_op_utils.cc
    hcom_base_funs.cc
    ops_kernel_builder_base.cc
    hcom_acl_adapter.cc
    hcom_log.cc
)

target_sources(hcom_gradtune_opskernel_builder PRIVATE
    hcom_op_utils.cc
    hcom_base_funs.cc
    ops_kernel_builder_base.cc
    hcom_log.cc
)