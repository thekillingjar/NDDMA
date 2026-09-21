# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# -*- coding:utf-8 -*-

# 此例中，构造一个完全符合pass 的场景，pass可顺利执行
import torch
import torch.nn as nn
import torch_npu
import torchair


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.add(torch.matmul(x, y), z)


if __name__ == "__main__":
    model = Model()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    x, y = torch.randn(2, 3), torch.randn(3, 2)
    z = torch.randn(2, 2)
    model = torch.compile(model, backend=npu_backend)
    res = model(x, y, z)
    print(res)
