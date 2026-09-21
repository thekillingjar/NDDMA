/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_AICPU_EXT_INFO_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_AICPU_EXT_INFO_H_

#include "fwk_adpt_struct.h"
#include "ge/ge_api_error_codes.h"
#include "aicpu_engine_struct.h"
#include "graph/op_desc.h"
#include "exe_graph/runtime/shape.h"
#include "runtime/rt.h"

namespace gert {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AsyncWaitInfo = aicpu::FWKAdapter::AsyncWait;
using WorkSpaceInfo = aicpu::FWKAdapter::WorkSpaceInfo;
using AicpuSessionInfo = SessionInfo;

class AicpuExtInfoHandler {
 public:
  AicpuExtInfoHandler(const std::string &node_name, const uint32_t input_num, const uint32_t output_num,
                      const ge::UnknowShapeOpType unknown_type)
      : node_name_(node_name), input_num_(input_num), output_num_(output_num), unknown_type_(unknown_type) {
  }

  ~AicpuExtInfoHandler() = default;

  uint8_t *GetExtInfo() const {
    return ext_info_;
  }
  size_t GetExtInfoLen() const {
    return ext_info_len_;
  }
  void *GetDeviceAddr() const {
    return device_ext_info_;
  }
  uint32_t GetOutputNum() const {
    return output_num_;
  }
  WorkSpaceInfo* GetWorkSpaceInfo() const {
    return workspace_info_;
  }

  ge::Status Parse(const std::string &ext_info, uint8_t *host_addr, void *device_addr);

  ge::Status UpdateInputShape(const uint32_t input_index, const Shape &input_shape);
  ge::Status UpdateOutputShape(const uint32_t output_index, const Shape &output_shape);

  ge::Status UpdateSessionInfo(const uint64_t session_id, const uint64_t kernel_id, const bool sess_flag) const;
  ge::Status UpdateSessionInfoId(const uint64_t session_id) const;
  ge::Status UpdateWorkSpaceInfo(const uint64_t workspace_size, const uint64_t workspace_addr) const;

  ge::Status UpdateExecuteMode(const bool flag);
  ge::Status UpdateEventId(const uint32_t event_id);

  ge::Status GetOutputShapeAndType(const uint32_t output_index, Shape &shape, ge::DataType &data_type) const;
  ge::Status CopyH2D(const rtStream_t stream);
  ge::Status CopyOutputShapeForThirdOp();

  int32_t GetDeployTypeFlag() const { return deploy_type_flag_; }
  uint32_t GeQosLevelFlag() const { return qos_level_flag_; }

 private:
  ge::Status ParseExtShapeType(AicpuExtInfo &aicpu_ext_info) const;
  ge::Status ParseExtInputShape(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtOutputShape(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtSessionInfo(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtBitMap(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtUpdateAddr(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtTopicType(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtAsyncWait(AicpuExtInfo &aicpu_ext_info);
  ge::Status ParseExtWorkSpaceInfo(AicpuExtInfo &aicpu_ext_info);

  static ge::Status UpdateShape(const Shape &shape, AicpuShapeAndType *const shape_and_type);

  static void GetShapeAndType(const AicpuShapeAndType &shape_and_type, Shape &shape, ge::DataType &data_type);

  static int32_t TopicTypeToRtsFlag(const int32_t topic_type);

  // base info
  const std::string node_name_;
  const uint32_t input_num_;
  const uint32_t output_num_;
  const ge::UnknowShapeOpType unknown_type_;

  // host and device info
  uint8_t *ext_info_ = nullptr;
  void *device_ext_info_ = nullptr;
  size_t ext_info_len_ = 0U;

  // some ext_info
  AicpuSessionInfo *session_info_ = nullptr;
  AsyncWaitInfo *async_wait_ = nullptr;
  uint64_t *bit_map_ = nullptr;
  uint32_t *update_addr_ = nullptr;
  int32_t deploy_type_flag_ = 0;  // default is device only
  uint32_t qos_level_flag_ = 0U;
  WorkSpaceInfo *workspace_info_ = nullptr;

  std::vector<AicpuShapeAndType *> input_shape_and_type_;
  std::vector<AicpuShapeAndType *> output_shape_and_type_;

  // for 3th op update output
  std::vector<AicpuShapeAndType> output_shape_;
  size_t output_shape_offset_ = 0U;
  size_t output_shape_len_ = 0U;
};
}  // namespace gert
#endif // AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_AICPU_EXT_INFO_H_
