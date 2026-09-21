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


#define private public
#define protected public
#include "common/scope_allocator.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#undef private
#undef protected

using namespace ge;
using namespace fe;

class UTTEST_scope_allocator : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "fusion engine scope allocator UT SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "fusion engine scope allocator UT TearDown" << std::endl;
    }

    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {

    }
};

TEST_F(UTTEST_scope_allocator, alloc_scope_id) {
  bool case1 = ScopeAllocator::Instance().AllocateScopeId() > 0;
  EXPECT_EQ(case1, true);

  bool case2 = ScopeAllocator::Instance().AllocateNegScopeId() < 0;
  EXPECT_EQ(case2, true);

  bool case3 = ScopeAllocator::Instance().AllocateScopeId() < ScopeAllocator::Instance().AllocateScopeId();
  EXPECT_EQ(case3, true);

  bool case4 = ScopeAllocator::Instance().AllocateNegScopeId() > ScopeAllocator::Instance().AllocateNegScopeId();
  EXPECT_EQ(case4, true);

  bool case5 = ScopeAllocator::Instance().AllocateNegScopeId() == ScopeAllocator::Instance().GetCurrentNegScopeId();
  EXPECT_EQ(case5, true);

  bool case6 = ScopeAllocator::Instance().AllocateFixpipeScopeId() > 0;
  EXPECT_EQ(case6, true);

  ScopeAllocator::Instance().SetCurrentScopeId(5);
  int64_t scope_id = 0;
  scope_id = ScopeAllocator::Instance().AllocateScopeId();
  EXPECT_EQ(scope_id, 6);

  ScopeAllocator::Instance().SetCurrentNegScopeId(-6);
  int64_t neg_scope_id = 0;
  neg_scope_id = ScopeAllocator::Instance().AllocateNegScopeId();
  EXPECT_EQ(neg_scope_id, -7);

  ScopeAllocator::Instance().SetFixpipeCurrentScopeId(5);
  int64_t fixpipe_scope_id = 0;
  fixpipe_scope_id = ScopeAllocator::Instance().GetCurrentFixpipeScopeId();
  EXPECT_EQ(fixpipe_scope_id, 5);
}

TEST_F(UTTEST_scope_allocator, ub_scope_id)
{
  int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
  ge::OpDescPtr op_desc = nullptr;
  bool ret = ScopeAllocator::HasScopeAttr(op_desc);
  EXPECT_EQ(ret, false);
  ret = ScopeAllocator::SetScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, false);
  int64_t temp_scope_id = 0;
  ret = ScopeAllocator::GetScopeAttr(op_desc, temp_scope_id);
  EXPECT_EQ(ret, false);

  op_desc = std::make_shared<ge::OpDesc>("relu1", "Relu");
  ret = ScopeAllocator::HasScopeAttr(op_desc);
  EXPECT_EQ(ret, false);

  ret = ScopeAllocator::SetScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, true);

  ret = ScopeAllocator::GetScopeAttr(op_desc, temp_scope_id);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, temp_scope_id);
}

TEST_F(UTTEST_scope_allocator, l1_scope_id)
{
  int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
  ge::OpDescPtr op_desc = nullptr;
  bool ret = ScopeAllocator::HasL1ScopeAttr(op_desc);
  EXPECT_EQ(ret, false);
  ret = ScopeAllocator::SetL1ScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, false);
  int64_t temp_scope_id = 0;
  ret = ScopeAllocator::GetL1ScopeAttr(op_desc, temp_scope_id);
  EXPECT_EQ(ret, false);

  op_desc = std::make_shared<ge::OpDesc>("relu1", "Relu");
  ret = ScopeAllocator::HasL1ScopeAttr(op_desc);
  EXPECT_EQ(ret, false);

  ret = ScopeAllocator::SetL1ScopeAttr(op_desc, scope_id);
  EXPECT_EQ(ret, true);

  ret = ScopeAllocator::GetL1ScopeAttr(op_desc, temp_scope_id);
  EXPECT_EQ(ret, true);
  EXPECT_EQ(scope_id, temp_scope_id);
}
