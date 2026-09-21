/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_AICPU_KERNEL_AICPU_ARGS_HANDLER_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_AICPU_KERNEL_AICPU_ARGS_HANDLER_H_
#include <string>
#include <memory>
#include "graph/def_types.h"
#include "runtime/kernel.h"
#include "ge/ge_api_error_codes.h"
#include "graph/ge_error_codes.h"

namespace gert {
class AicpuArgsHandler {
 public:
  AicpuArgsHandler(const std::string &node_name, const uint32_t io_num, const bool need_device_ext) :
      node_name_(node_name), io_num_(io_num), need_device_ext_(need_device_ext), args_({}) {
    io_sizes_.resize(io_num_);
  }

  uint8_t *GetIoAddr() const {
    return host_buffer_.get() + io_addr_offset_;
  }

  void SetIoSize(size_t index, size_t size) {
    if (index >= io_num_) {
      return;
    }
    io_sizes_[index] = size;
  }

  size_t GetIoSize(size_t index) {
    if (index >= io_num_) {
      return 0U;
    }
    return io_sizes_[index];
  }

  uint8_t *GetExtInfoAddr() const {
    return host_buffer_.get() + ext_info_offset_;
  }

  uint8_t *GetArgs() const {
    return host_buffer_.get();
  }

  void *GetExtInfoDeivceAddr() const {
    return ext_info_device_buffer_;
  }

  const rtAicpuArgsEx_t &GetArgsEx() const {
    return args_;
  }

  const std::string &GetNodeName() const {
    return node_name_;
  }

  size_t GetIoNum() const {
    return io_num_;
  }

  size_t GetHostInputSize() const {
    return host_input_size_;
  }

  size_t GetInputAddrAlignBytes() const {
    return align_bytes_;
  }

  void SetBlockOp(const bool is_block_op) {
    is_block_op_ = is_block_op;
  }

  bool IsBlockOp() const {
    return is_block_op_;
  }

  void SetRtEvent(const rtEvent_t rt_event) {
    rtEvent_ = rt_event;
  }

  rtEvent_t GetRtEvent() const {
    return rtEvent_;
  }

  void SetAsyncTimeout(const uint32_t async_timeout) {
    async_timeout_ = async_timeout;
  }

  uint32_t GetAsyncTimeout() const {
    return async_timeout_;
  }

  const inline std::vector<rtHostInputInfo_t> &GetKernelOffset() const {
    return kernel_offset_info_;
  }

  const inline std::vector<rtHostInputInfo_t> &GetHostInputOffset() const {
    return host_input_info_;
  }

  ge::graphStatus MallocMem();

  void ResetHostInputInfo();

  // align_size is for op in device, actual alloced src_size may smaller than align_size.
  ge::graphStatus AddHostInput(const size_t idx, void *data, const size_t src_size, const size_t align_size);

  ~AicpuArgsHandler();

 protected:
  ge::graphStatus SetLaunchArgs(const size_t arg_size);

  const std::string node_name_;
  const size_t io_num_;
  const bool need_device_ext_;
  bool is_block_op_ = false;
  rtEvent_t rtEvent_ = nullptr;
  uint32_t async_timeout_ = 0xFFFFFFFF;

  // for rtKernelLaunch
  rtAicpuArgsEx_t args_;
  std::vector<rtHostInputInfo_t> kernel_offset_info_;
  std::vector<rtHostInputInfo_t> host_input_info_;

  // args
  std::unique_ptr<uint8_t[]> host_buffer_;
  void *ext_info_device_buffer_ = nullptr;
  std::vector<size_t> io_sizes_;

  // offset
  size_t io_addr_offset_ = 0U;
  size_t ext_info_offset_ = 0U;
  size_t host_mem_offset_ = 0U;
  size_t so_name_offset_ = 0U;
  size_t kernel_name_offset_ = 0U;
  size_t task_info_offset_ = 0;

  // size
  size_t ext_info_size_ = 0U;
  size_t buffer_size_ = 0U;
  size_t host_input_size_ = 0U;
  size_t align_bytes_ = 4U;
};

/* 自研host_buffer排布
 *  |args, 包括AicpuParamHead|
 *  |io_addr|
 *  |kernel_name|
 *  |so_name|
 *  |extInfo|
 *  |host_input|
 */
class AicpuCCArgsHandler : public AicpuArgsHandler {
 public:
  AicpuCCArgsHandler(const std::string &node_name, const uint32_t io_num, const bool need_device_ext) :
      AicpuArgsHandler(node_name, io_num, need_device_ext) {}

  ge::graphStatus BuildCCArgs(const std::string &arg_data, const std::string &kernel_name, const std::string &so_name,
                              const size_t ext_info_size);
  ge::graphStatus BuildHostCCArgs(const std::string &arg_data, const size_t ext_info_size);

 private:
  ge::graphStatus SetHostArgs(const std::string &arg_data, const size_t ext_info_size, const uint64_t ext_info_addr);
  ge::graphStatus SetOffsetArgs();
};

/* STR_FWK_OP_KERNEL 组成
 *  |workspaceBaseAddr|
 *  |extInfoAddr|
 *  |extInfoLen|
 *  |inputOutputAddr|
 *  |kernelID|
 *  |sessionID|
 */

/* host_buffer 排布
 *  |STR_FWK_OP_KERNEL|
 *  |task_info|
 *  |io_addr|
 *  |ext_info|
 *  |host_input|
 *  |so_name| 占位
 *  |kernel_name| 占位
 */
class AicpuTfArgsHandler : public AicpuArgsHandler {
 public:
  AicpuTfArgsHandler(const std::string &node_name, const uint32_t io_num, const bool need_device_ext) :
      AicpuArgsHandler(node_name, io_num, need_device_ext) {}

  ge::graphStatus BuildTfArgs(const std::string &arg_data, const std::string &task_info, const size_t ext_info_size,
                              const uint64_t session_id, const void *step_id);

 private:
  ge::graphStatus SetOffsetArgs();
};
} // gert
#endif // AIR_CXX_RUNTIME_V2_ENGINE_AICPU_KERNEL_AICPU_ARGS_HANDLER_H_
