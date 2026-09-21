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

"""
build pass specific declaration and schedules.
"""

from tbe.common.register.fusion_pass import conv2d_data_rm_fusion_pass
from tbe.common.register.fusion_pass import cube_transfer_shape_pass
from tbe.common.register.fusion_pass import elemwise_transfer_shape_pass
from tbe.common.register.fusion_pass import fc_transfer_shape_pass
from tbe.common.register.fusion_pass import matmul_transfer_shape_pass