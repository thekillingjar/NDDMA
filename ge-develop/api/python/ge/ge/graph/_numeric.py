#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numeric module for numeric operations in GraphEngine."""

import struct
from typing import List

__all__ = ["float_to_fp16_bits", "float_list_to_fp16_bits"]


def float_to_fp16_bits(value: float) -> int:
    """Convert float to fp16 bits."""
    f32 = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    sign = (f32 >> 31) & 0x1
    exponent = (f32 >> 23) & 0xff
    mantissa = f32 & 0x7fffff

    if exponent == 255:
        half_exp = 0x1F
        half_mant = 0x200 if mantissa != 0 else 0
    else:
        exponent -= 127
        if exponent > 15:
            half_exp = 0x1F
            half_mant = 0
        elif exponent < -14:
            if exponent < -24:
                half_exp = 0
                half_mant = 0
            else:
                mantissa |= 0x800000
                shift = -exponent - 14
                shifted = mantissa >> (shift + 13)
                round_bit = (mantissa >> (shift + 12)) & 0x1
                remainder = mantissa & ((1 << (shift + 12)) - 1)
                if round_bit and ((shifted & 0x1) or remainder):
                    shifted += 1
                half_exp = 0
                half_mant = shifted
        else:
            half_exp = exponent + 15
            half_mant = mantissa >> 13
            round_bits = mantissa & 0x1FFF
            if round_bits > 0x1000 or (round_bits == 0x1000 and (half_mant & 0x1)):
                half_mant += 1
                if half_mant == 0x400:
                    half_mant = 0
                    half_exp += 1
                    if half_exp == 0x1F:
                        half_mant = 0
    return (sign << 15) | ((half_exp & 0x1F) << 10) | (half_mant & 0x3FF)


def float_list_to_fp16_bits(values: List[float]) -> List[int]:
    """Convert float list to fp16 bits list."""
    return [float_to_fp16_bits(v) for v in values]
