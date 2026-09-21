/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_impl/transdata/transdata_op_impl.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_data.h"

namespace gert {
namespace kernel {
ge::graphStatus InferShapeForTransData(InferShapeContext *context);
}
}
namespace gert_test {
class TransDataImplUT : public testing::Test {};
TEST_F(TransDataImplUT, TransDataInferShapeOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData"), nullptr);
  auto infer_shape_func = gert::kernel::InferShapeForTransData;

  gert::StorageShape input_shape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 8);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 224);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 224);
}
TEST_F(TransDataImplUT, TilingParseOk) {
  char *json = (char *)"{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": "
               "126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  char *node_type = (char *)"TransData";
  gert::kernel::TransDataCompileInfo compile_info;

  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs(std::vector<void *>({json, node_type}))
                    .Outputs(std::vector<void *>({&compile_info}))
                    .Build();


  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData"), nullptr);
  auto func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData")->tiling_parse;
  ASSERT_NE(func, nullptr);

  EXPECT_EQ(func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(compile_info.ub_size, 126464);
  EXPECT_EQ(compile_info.block_dim, 32);
  EXPECT_EQ(compile_info.group, 1);
  EXPECT_EQ(compile_info.vnc_fp32_flag, 0);
}

TEST_F(TransDataImplUT, TilingNd2Nz) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 256}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};

  // compile info
  gert::kernel::TransDataCompileInfo compile_info;
  compile_info.block_dim = 32;
  compile_info.ub_size = 128 * 1024;
  compile_info.group = 1;
  compile_info.vnc_fp32_flag = false;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info)
                    .TilingData(param.get())
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData"), nullptr);
  auto tiling_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}
TEST_F(TransDataImplUT, TilingNz2Nd) {
  gert::StorageShape in_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  gert::StorageShape out_shape = {{1, 16, 256}, {1, 16, 256}};

  // compile info
  gert::kernel::TransDataCompileInfo compile_info;
  compile_info.block_dim = 32;
  compile_info.ub_size = 128 * 1024;
  compile_info.group = 1;
  compile_info.vnc_fp32_flag = false;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .CompileInfo(&compile_info)
                    .TilingData(param.get())
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData"), nullptr);
  auto tiling_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("TransData")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}
}  // namespace gert_test