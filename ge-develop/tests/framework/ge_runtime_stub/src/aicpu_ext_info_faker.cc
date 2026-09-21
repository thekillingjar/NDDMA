/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/aicpu_ext_info_faker.h"
#include <cinttypes>
#include <securec.h>
#include "faker/task_def_faker.h"
#include "aicpu_engine_struct.h"
#include "aicpu_task_struct.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"

namespace gert {
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
namespace {
void AppendShape(aicpu::FWKAdapter::FWKTaskExtInfoType type, size_t shape_num, std::string &out) {
  size_t len = sizeof(AicpuShapeAndType) * shape_num + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = type;
  aicpu_ext_info->infoLen = sizeof(AicpuShapeAndType) * shape_num;
  AicpuShapeAndType input_shape_and_types[shape_num];
  for (auto m = 0; m < shape_num; m++) {
    input_shape_and_types[m].dims[0] = 5;
  }
  memcpy_s(aicpu_ext_info->infoMsg, sizeof(AicpuShapeAndType) * shape_num,
           reinterpret_cast<void *>(input_shape_and_types), sizeof(AicpuShapeAndType) * shape_num);

  std::string s(vec.data(), len);
  out.append(s);
}

void AppendSessionInfo(std::string &out) {
  size_t len = sizeof(AicpuSessionInfo) + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
  aicpu_ext_info->infoLen = sizeof(AicpuSessionInfo);
  AicpuSessionInfo session;
  *(ge::PtrToPtr<char, AicpuSessionInfo>(aicpu_ext_info->infoMsg)) = session;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendBitMap(std::string &out) {
  size_t len = sizeof(uint64_t) + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP;
  aicpu_ext_info->infoLen = sizeof(uint64_t);
  *(ge::PtrToPtr<char, uint64_t>(aicpu_ext_info->infoMsg)) = 1;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendUpdateAddr(std::string &out) {
  size_t len = sizeof(uint32_t) + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_UPDATE_ADDR;
  aicpu_ext_info->infoLen = sizeof(uint32_t);
  *(ge::PtrToPtr<char, uint32_t>(aicpu_ext_info->infoMsg)) = 1;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendTopicType(std::string &out) {
  size_t len = sizeof(int32_t) + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE;
  aicpu_ext_info->infoLen = sizeof(int32_t);
  *(ge::PtrToPtr<char, int32_t>(aicpu_ext_info->infoMsg)) = 1;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendAsyncWait(std::string &out) {
  size_t len = sizeof(AsyncWaitInfo) + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  AsyncWaitInfo info;
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
  aicpu_ext_info->infoLen = sizeof(AsyncWaitInfo);
  *(ge::PtrToPtr<char, AsyncWaitInfo>(aicpu_ext_info->infoMsg)) = info;
  std::string s(vec.data(), len);
  out.append(s);
}

void AppendWorkSpaceInfo(std::string &out) {
  size_t len = sizeof(WorkSpaceInfo) + sizeof(AicpuExtInfo);
  vector<char> vec(len);
  AicpuExtInfo *aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(vec.data());
  WorkSpaceInfo info;
  info.size = 1024UL;
  info.addr = 1024UL;
  aicpu_ext_info->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_WORKSPACE_INFO;
  aicpu_ext_info->infoLen = sizeof(WorkSpaceInfo);
  *(ge::PtrToPtr<char, WorkSpaceInfo>(aicpu_ext_info->infoMsg)) = info;
  std::string s(vec.data(), len);
  out.append(s);
}
} // namespace

std::string GetFakeExtInfo() {
  std::string ext_info;
  AppendShape(aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE, 2, ext_info);
  AppendShape(aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE, 1, ext_info);
  AppendSessionInfo(ext_info);
  AppendBitMap(ext_info);
  AppendUpdateAddr(ext_info);
  AppendTopicType(ext_info);
  AppendAsyncWait(ext_info);
  AppendWorkSpaceInfo(ext_info);
  return ext_info;
}
} // gert
