/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/model_converter.h"
#include "graph/utils/graph_utils.h"
#include "runtime/model_v2_executor.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "faker/fake_value.h"
#include "runtime/gert_api.h"

namespace  gert {
ge::graphStatus lstsmpGraphExec() {
  rtSetDevice(0);
  ge::graphStatus error_code;
  auto model_executor = gert::LoadExecutorFromFile("./lstmp_input_256_nocast.om", error_code);
  GE_CHK_GRAPH_STATUS_RET(model_executor->Load(),"load model executor failed") ;
  auto allocator = memory::CachingMemAllocator::GetAllocator();
  const uint32_t buffer_size = 8 * 512 *1024;
  auto out_tensor_buffer = allocator->Malloc(buffer_size);
  auto outputs = FakeTensors({2}, 3, out_tensor_buffer);
  auto input_tensor_buffer = allocator->Malloc(buffer_size);

  auto i0 = FakeValue<Tensor>(
      Tensor{{{8, 1, 256}, { 8, 1, 256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, input_tensor_buffer});
  auto i1 = FakeValue<Tensor>(
      Tensor{{{8}, {8}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_INT32, input_tensor_buffer});
  auto i2 = FakeValue<Tensor>(
      Tensor{{{1, 8, 512}, {1, 8, 512}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, input_tensor_buffer});
  auto i3 = FakeValue<Tensor>(
      Tensor{{{1, 8, 512}, {1, 8, 512}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, input_tensor_buffer});

  rtStream_t stream;
  rtStreamCreate(&stream ,static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT));

  auto i4 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  GE_CHK_GRAPH_STATUS_RET(model_executor->Execute(std::vector<void *>({i0.value, i1.value, i2.value, i3.value, i4.value}).data(), 5,
                          outputs.GetAddrList(), outputs.size()), "model executor init execute failed");

  GE_CHK_GRAPH_STATUS_RET(model_executor->Execute(std::vector<void *>({i0.value, i1.value, i2.value, i3.value, i4.value}).data(), 5,
                          outputs.GetAddrList(), outputs.size()), "model executor execute failed");
  GE_CHK_GRAPH_STATUS_RET(model_executor->UnLoad(), "model executor unload failed");
  return ge::GRAPH_SUCCESS;
}
}

int main(){
  gert::lstsmpGraphExec();
  return 0;
}

