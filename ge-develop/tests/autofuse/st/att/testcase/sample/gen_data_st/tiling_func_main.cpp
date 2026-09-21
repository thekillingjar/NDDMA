/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**************************************************************
* 本文件提供FlashAttention算子的TilingFunc执行的测试用例，由工具自动生成
**************************************************************/
#include "functional"
#include "map"
#include "checker.h"
#include "OpTest_tiling_data.h"
#include "user_input_parser.h"
#include "kernel_context_holder_builder.h"
using namespace att;
namespace optiling {
extern ge::graphStatus GetCtxTiling(gert::TilingContext *context);
}
int main(int32_t argc, char *argv[]) {
  // TODO 用户可以根据实际情况，修改输入的shape
  KernelContextHolderBuilder builder;
  auto holder = builder
         .AddInput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ge::Format::FORMAT_ND, ge::DataType::DT_FLOAT16))
         .AddInput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ge::Format::FORMAT_ND, ge::DataType::DT_FLOAT16))
         .AddOutput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ge::Format::FORMAT_ND, ge::DataType::DT_FLOAT16))
         .SetTilingData(10240)
         .SetWorkSpace(1600)
         .SetCompileInfo(2)
         .SetPlatformInfo()
         .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
         .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<bool>(true)})
         .Build();
  gert::TilingContext *ctx = reinterpret_cast<gert::TilingContext *>(holder.context_);
  if (optiling::GetCtxTiling(ctx) == ge::GRAPH_SUCCESS) {
    ATTLOGI("Get tiling info successfully, you can print some data what you want to see.");
    // TODO 用户可根据期望获取的数据，打印不同输出的值
    GET_TILING_DATA(tmpTiling, ctx->GetRawTilingData()->GetData());
    PrintTilingData(tmpTiling);
    std::cout << "Tiling func execute success, tiling_key=" << ctx->GetTilingKey() << std::endl;
  } else {
    ATTLOGE(ge::FAILED, "Error:failed to get tiling!");
    return -1;
  }
  return 0;
}
