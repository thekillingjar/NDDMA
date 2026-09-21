/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _RTS_ENGINE_OP_OP_H_
#define _RTS_ENGINE_OP_OP_H_

#include <string>
#include <vector>
#include <climits>
#include <unordered_map>
#include "common/util/log.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/node.h"
#include "proto/task.pb.h"
#include "common/optimizer/graph_optimizer.h"

using namespace std;
using namespace ge;

namespace cce {
namespace runtime {
using domi::TaskDef;
/**
 * The base class for all op.
 */
class Op {
 public:
  Op(const ge::Node &node, ge::RunContext &runContext);

  virtual ~Op() = default;

  /**
   *  @brief init param for run
   *  @return SUCCESS:success
   *          Other:failed
   */
  virtual ge::Status Init() = 0;

  virtual ge::Status Run(vector<TaskDef> &tasks) = 0;
  virtual ge::Status UpdateTaskDef(vector<TaskDef> &tasks) {
    const uint32_t streamId = op_desc_->GetStreamId();
    for (auto &taskDef : tasks) {
      taskDef.set_stream_id(streamId);
    }
    RTS_LOGI("update taskDefSize:%zu, streamId:%u.", tasks.size(), streamId);
    return ge::SUCCESS;
  }
  virtual ge::Status GenerateCtxDef(const ge::Node &node) {
    (void)node;
    return ge::SUCCESS;
  }

 protected:
  ge::Status UpdateOutDescFromInDesc(const void *inputAddr, const void *outputAddr, vector<TaskDef> &tasks);

 protected:
  const ge::RunContext &run_context_;
  const ge::Node &node_;
  ge::ConstOpDescPtr op_desc_;
  std::string name_;
  std::string type_;

  // input
  size_t input_num_;
  vector<void *> v_input_data_addr_;
  vector<int64_t> v_input_size_;
  std::unordered_map<uintptr_t, uintptr_t> inputDescAddrs_;

  // output
  size_t output_num_;
  vector<void *> v_output_data_addr_;
  vector<int64_t> v_output_size_;
  std::unordered_map<uintptr_t, uintptr_t> outputDescAddrs_;
  bool dynamic_flag_{false};

 private:
  /**
   * calculate mem address.
   * @param mem_base mem base
   * @param mem_size mem size
   * @param offset mem offset, if < 0, no need check.
   * @param data_size data size
   * @param memAddr out addr
   * @return SUCCESS:success
   *         Other:failed
   */
  ge::Status CalcAddr(uintptr_t mem_base, uint64_t mem_size, int64_t offset, int64_t data_size, uintptr_t &memAddr);

  ge::Status CheckOffsetAndSize(uint64_t offset, uint64_t space_size, uint64_t total_size);

  ge::Status GetOpInputMemData(uint8_t *&memBase, uint64_t &memSize, int64_t &inputOffset,
                               ge::ConstGeTensorDescPtr tensorDescPtr);

  ge::Status GetOpOutputMemData(uint8_t *&memBase, uint64_t &memSize, int64_t &outputOffset,
                                ge::ConstGeTensorDescPtr tensorDescPtr);

  ge::Status InitInput();

  ge::Status InitOutput();

  ge::Status GetOpMemType(const std::string &str, int64_t &memType) const;

  static bool CheckUint64Overflow(uint64_t a, uint64_t b) {
    // overflow
    return b > (ULLONG_MAX - a);
  }

  ge::Status InitDesc(const uintptr_t memBase, const uint64_t memSize, const ge::ConstGeTensorDescPtr &tensorDesc,
                      const uintptr_t dataAddr, std::unordered_map<uintptr_t, uintptr_t> &descAddrs);
};
}  // namespace runtime
}  // namespace cce

#endif
