#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch_npu
import torchair


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.grouped_conv = nn.Conv2d(3, 3, 3, 3, 0, groups=3, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

    def forward(self, x):
        x = self.grouped_conv(x)
        x = self.pool(x)
        return x


if __name__ == "__main__":
    model = Model().npu()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    x = torch.randn(1, 3, 9, 9).npu()
    model = torch.compile(model, backend=npu_backend)
    res = model(x)
    print(res)
