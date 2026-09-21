# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.add(torch.matmul(x, y), z)


def convert():
    model = Model()
    model.eval()
    x, y = torch.randn(2, 3), torch.randn(3, 2)
    z = torch.randn(2, 2)
    # 当前atc工具opset_version最高支持18,若torch版本默认导出opset_version过高，请如下显示指定opset_version
    # torch.onnx.export(model, (x, y, z), "model.onnx", opset_version=11, do_constant_folding=False)
    torch.onnx.export(model, (x, y, z), "model.onnx", do_constant_folding=False)


if __name__ == "__main__":
    convert()
