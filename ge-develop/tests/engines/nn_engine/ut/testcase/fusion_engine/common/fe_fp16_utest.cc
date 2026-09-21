/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>

#define protected public
#define private public
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph/utils/graph_utils.h"
#include "graph/op_kernel_bin.h"
#include "common/graph_comm.h"
#include "common/fe_fp16_t.h"
#include "common/dump_util.h"
#include "common/util/op_info_util.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class FEFP16_UTEST : public testing::Test
{
protected:
    void SetUp()
    {
    }

    void TearDown()
    {
    }
};

TEST_F(FEFP16_UTEST, ManRoundToNearest01)
{
    bool bit0 = false;
    bool bit1 = true;
    bool bitleft = false;
    uint16_t man = 0x3D4C;
    uint16_t ret = ManRoundToNearest(bit0, bit1, bitleft, man);
    bool flag = false;
    if (ret != 0) {
      flag =true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(FEFP16_UTEST, GetManBitLength01)
{
    bool bit0 = false;
    bool bit1 = true;
    bool bitleft = false;
    uint16_t man = 0x3D4C;
    int16_t ret = GetManBitLength(man);
    bool flag = false;
    if (ret != 0) {
      flag =true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(FEFP16_UTEST, operator01)
{
    bool bit0 = false;
    bool bit1 = true;
    bool bitleft = false;
    float tmp1 = -128;
    float tmp2 = 114566333.666;
    float tmp3 = -1234545.5;
    uint16_t tmp4 = 0xFFFF4D32;
    uint16_t tmp5 = 0x7FFFFFDD;
    uint16_t tmp6 = 3;
    uint16_t tmp7 = 0xEFFF4D32;
    fe::fp16_t fp1 = tmp1;
    fe::fp16_t fp2 = tmp2;
    fe::fp16_t fp3 = tmp3;
    fe::fp16_t fp4 = tmp4;
    fe::fp16_t fp5 = tmp5;
    fe::fp16_t fp6 = tmp6;
    fe::fp16_t fp7 = tmp7;
    int16_t dst6 = static_cast<int16_t>(fp6);
    uint16_t dst4 = static_cast<uint16_t>(fp4);
    float dst1 = static_cast<float>(fp1);
    bool flag = false;
    if (dst4 != 0) {
      flag =true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(FEFP16_UTEST, operator02)
{
    const uint16_t tmp_val0 = 0x0;
    const uint16_t tmp_val1 = 0xEF32;
    const uint16_t tmp_val2 = 6;
    float tmp_val4 = 6;
    fe::fp16_t fp1 = tmp_val0;
    fe::fp16_t fp2= tmp_val1;
    fe::fp16_t fp3= tmp_val4;
    uint16_t dst4 = fp3.ToUInt16();
    std::cout << "fe_fp16 value = " << dst4 << std::endl;
    bool flag = false;
    EXPECT_EQ(flag, false);
}
