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
#include "common/memory/mem_type_utils.h"

namespace ge {
class UtestMemTypeUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestMemTypeUtils, RtMemTypeToExternalMemType_Success) {
  MemoryType external_mem_type;
  EXPECT_EQ(MemTypeUtils::RtMemTypeToExternalMemType(RT_MEMORY_HBM, external_mem_type), GRAPH_SUCCESS);
  EXPECT_EQ(external_mem_type, MemoryType::MEMORY_TYPE_DEFAULT);

  EXPECT_EQ(MemTypeUtils::RtMemTypeToExternalMemType(RT_MEMORY_DEFAULT, external_mem_type), GRAPH_SUCCESS);
  EXPECT_EQ(external_mem_type, MemoryType::MEMORY_TYPE_DEFAULT);

  EXPECT_EQ(MemTypeUtils::RtMemTypeToExternalMemType(RT_MEMORY_P2P_DDR, external_mem_type), GRAPH_SUCCESS);
  EXPECT_EQ(external_mem_type, MemoryType::MEMORY_TYPE_P2P);
}

TEST_F(UtestMemTypeUtils, RtMemTypeToExternalMemType_Failed) {
  MemoryType external_mem_type;
  EXPECT_NE(MemTypeUtils::RtMemTypeToExternalMemType(RT_MEMORY_TS, external_mem_type), GRAPH_SUCCESS);
}

TEST_F(UtestMemTypeUtils, ToString_RtMemType_Success) {
  EXPECT_STRCASEEQ(MemTypeUtils::ToString(RT_MEMORY_TS).c_str(), "RT_MEMORY_TS");
  EXPECT_STRCASEEQ(MemTypeUtils::ToString(123456).c_str(), "123456");
}

TEST_F(UtestMemTypeUtils, ToString_ExternalMemType_Success) {
  EXPECT_STRCASEEQ(MemTypeUtils::ToString(MemoryType::MEMORY_TYPE_DEFAULT).c_str(), "MEMORY_TYPE_DEFAULT");
  EXPECT_STRCASEEQ(MemTypeUtils::ToString((MemoryType)123456).c_str(), "unknown type: 123456");
}

TEST_F(UtestMemTypeUtils, ExternalMemTypeToRtMemType_Success) {
  rtMemType_t rt_mem_type;
  EXPECT_EQ(MemTypeUtils::ExternalMemTypeToRtMemType(MemoryType::MEMORY_TYPE_DEFAULT, rt_mem_type), GRAPH_SUCCESS);
  EXPECT_EQ(rt_mem_type, RT_MEMORY_HBM);

  EXPECT_EQ(MemTypeUtils::ExternalMemTypeToRtMemType(MemoryType::MEMORY_TYPE_P2P, rt_mem_type), GRAPH_SUCCESS);
  EXPECT_EQ(rt_mem_type, RT_MEMORY_P2P_DDR);
}

TEST_F(UtestMemTypeUtils, ExternalMemTypeToRtMemType_Failed) {
  rtMemType_t rt_mem_type;
  EXPECT_NE(MemTypeUtils::ExternalMemTypeToRtMemType((MemoryType)2, rt_mem_type), GRAPH_SUCCESS);
}
}
