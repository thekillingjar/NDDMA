/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_TOOLCHAIN_DUMP_UDF_DUMP_TASK_DEVICE_H
#define UDF_TOOLCHAIN_DUMP_UDF_DUMP_TASK_DEVICE_H

#include <memory>
#include <string>
#include <vector>
#include "udf_dump_task.h"
#include "udf_dump_common_utils.h"
#include "flow_func/flow_func_defines.h"
#include "ff_dump_data.pb.h"

namespace FlowFunc {
using IDE_SESSION = void *;

class UdfDumpTaskDevice : public UdfDumpTask {
public:
    UdfDumpTaskDevice(const std::string &flow_func_info, const int32_t host_pid, const uint32_t device_id,
                      const int32_t aicpu_pid = 0, const uint32_t logic_dev_id = 0);
    ~UdfDumpTaskDevice() override = default;

    int32_t DumpOpInfo(const std::string &dump_file_path, const uint32_t step_id) override;

    int32_t PreProcessOutput(uint32_t index, const MbufTensor *tensor) override;

    int32_t PreProcessInput(const std::vector<MbufTensor *> &tensors) override;

private:

    int32_t ProcessDumpTensor(const std::string &dump_file_path);

    int32_t ProcessInputDump(const dumpNS::DumpData &dump_data);

    int32_t ProcessOutputDump(const dumpNS::DumpData &dump_data);

    std::string DumpPath(const uint64_t now_time, const uint32_t step_id);

    int32_t DoDumpTensor(const std::string &dump_file_path);

    int32_t ProcessDumpData(uint64_t len, uint64_t data_addr, const std::string &path,
                            uint64_t &dumped_size, const IDE_SESSION ide_session);
    
    int32_t SendDumpMessageToAicpuSd() const;

    dumpNS::DumpData base_dump_data_;
    std::string base_dump_path_;
    std::string op_name_;
    uint64_t input_total_size_;
    uint64_t output_total_size_;

    std::unique_ptr<char[]> buff_;
    uint64_t buff_size_;
    uint64_t offset_;
    int32_t host_pid_;
    uint32_t device_id_;
    int32_t aicpu_pid_ = 0;
    uint32_t logic_dev_id_ = 0;
};  // class UdfDumpTaskDevice
}  // namespace FlowFunc
#endif // UDF_TOOLCHAIN_DUMP_UDF_DUMP_TASK_DEVICE_H
