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
#include <dlfcn.h>
#define protected public
#define private public

#include "ge/ge_api_types.h"
#undef private
#undef protected
#include "ge_graph_dsl/assert/check_utils.h"
#include "framework/fe_running_env/include/fe_running_env.h"

using namespace std;
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc,argv);


#ifdef DAVINCI_TINY
    rtError_t rtRet = tiny_rt_init();
    if (rtRet != ACL_RT_SUCCESS)
    {
        std::cout<< "rtSetDevice Error"<<std::endl;
    }
#endif
    ge::CheckUtils::init();

    int ret = RUN_ALL_TESTS();
    return ret;
}
