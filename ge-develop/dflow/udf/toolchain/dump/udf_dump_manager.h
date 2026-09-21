/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_TOOLCHAIN_DUMP_UDF_DUMP_MANAGER_H
#define UDF_TOOLCHAIN_DUMP_UDF_DUMP_MANAGER_H

#include <string>
#include <set>
#include <vector>
#include "flow_func/flow_func_defines.h"
#include "flow_func/flow_func_dumper.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY UdfDumpManager {
public:
    static UdfDumpManager &Instance();

    void ClearDumpInfo();

    void SetDumpPath(const std::string &path);

    const std::string &GetDumpPath() const;

    void SetDumpStep(const std::string &step);

    const std::string &GetDumpStep() const;

    bool IsInDumpStep(const uint32_t step_id) const;

    void SetDumpMode(const std::string &mode);

    const std::string &GetDumpMode() const;

    void EnableDump();

    bool IsEnableDump() const;

    void SetDeviceId(uint32_t dev_id);

    uint32_t GetDeviceId() const;

    void SetHostPid(int32_t host_pid);

    int32_t GetHostPid() const;

    int32_t GetAicpuPid() const;

    int32_t Init();

    int32_t InitAicpuDump();

    void SetLogicDeviceId(uint32_t dev_id);

    uint32_t GetLogicDeviceId() const;
private:

    UdfDumpManager() = default;

    ~UdfDumpManager() = default;

    bool enable_dump_ = false;
    std::string dump_path_;
    std::string dump_step_;
    std::string dump_mode_;
    int32_t host_pid_ = 0;
    uint32_t device_id_ = 0;
    std::shared_ptr<FlowFuncDumper> dumper_;
    std::vector<std::pair<uint32_t, uint32_t>> step_range_;
    std::set<uint32_t> step_set_;
    int32_t aicpu_pid_ = 0;
    uint32_t logic_dev_id_ = 0;
};
}

#endif // UDF_TOOLCHAIN_DUMP_UDF_DUMP_MANAGER_H
