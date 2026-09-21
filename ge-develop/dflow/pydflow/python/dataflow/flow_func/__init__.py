#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from dataflow.flow_func.flow_func import FLOW_FUNC_SUCCESS, FLOW_FUNC_FAILED, FLOW_FUNC_ERR_PARAM_INVALID, \
    FLOW_FUNC_ERR_ATTR_NOT_EXITS, FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH, FLOW_FUNC_ERR_TIME_OUT_ERROR, \
    FLOW_FUNC_ERR_DRV_ERROR, FLOW_FUNC_ERR_QUEUE_ERROR, FLOW_FUNC_ERR_MEM_BUF_ERROR, FLOW_FUNC_ERR_EVENT_ERROR, \
    FLOW_FUNC_ERR_USER_DEFINE_START, FLOW_FUNC_ERR_USER_DEFINE_END, FlowFuncLogger, FlowMsg, PyMetaParams, \
    MetaRunContext, MSG_TYPE_RAW_MSG, MSG_TYPE_TENSOR_DATA, FLOW_FLAG_EOS, FLOW_FLAG_SEG, logger, \
    BalanceConfig, AffinityPolicy, MSG_TYPE_PICKLED_MSG, MSG_TYPE_TORCH_TENSOR_MSG, MSG_TYPE_USER_DEFINE_START, \
    FlowMsgQueue, FLOW_FUNC_STATUS_REDEPLOYING, FLOW_FUNC_STATUS_EXIT

from dataflow.flow_func.func_wrapper import proc_wrapper, init_wrapper
from dataflow.flow_func.func_register import FlowFuncRegister, FlowFuncInfos