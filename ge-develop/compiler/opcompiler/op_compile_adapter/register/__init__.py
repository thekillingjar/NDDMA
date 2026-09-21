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
tbe register
"""
from tbe.common.register.register_api import register_op_compute
from tbe.common.register.register_api import get_op_compute
from tbe.common.register.register_api import register_operator
from tbe.common.register.register_api import get_operator
from tbe.common.register.register_api import register_param_generalization
from tbe.common.register.register_api import get_param_generalization
from tbe.common.register.register_api import register_fusion_pass
from tbe.common.register.register_api import get_all_fusion_pass
from tbe.common.register.register_api import set_fusion_buildcfg
from tbe.common.register.register_api import get_fusion_buildcfg
from tbe.common.register.register_api import reset
from tbe.common.register.register_api import register_tune_space
from tbe.common.register.register_api import get_tune_space
from tbe.common.register.register_api import register_tune_param_check_supported
from tbe.common.register.register_api import get_tune_param_check_supported
from tbe.common.register.register_api import get_op_register_pattern
from tbe.common.register.register_api import register_pass_for_fusion
from tbe.common.register.register_api import register_op_param_pass

from tbe.common.register.class_manager import InvokeStage
from tbe.common.register.class_manager import Priority
from tbe.common.register.class_manager import FusionPassItem
from tbe.common.register.class_manager import OpCompute
from tbe.common.register.class_manager import Operator

from . import fusion_pass
