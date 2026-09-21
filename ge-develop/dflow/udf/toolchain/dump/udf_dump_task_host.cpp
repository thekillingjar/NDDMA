/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_dump_task_host.h"
#include <cstring>
#include <securec.h>
#include "common/udf_log.h"
#include "common/scope_guard.h"
#include "flow_func/flow_func_timer.h"
#include "mmpa/mmpa_api.h"
#include "flow_func/flow_func_config_manager.h"
#include "udf_dump_common_utils.h"

namespace FlowFunc {
namespace {
constexpr char kOsSplitChar = '/';
constexpr uint32_t kDefaultPathMode = 0700;
constexpr size_t kMaxErrstrLen = 128U;

std::string GetFileDir(const std::string &path) {
    std::string dir;
    size_t pos = path.find_last_of(kOsSplitChar);
    if (pos != std::string::npos) {
        dir = path.substr(0, pos);
        if (dir.empty()) {
            dir += "/";
        }
        return dir;
    }
    return dir;
}

bool IsFileExist(const std::string &path) {
    if (path.empty()) {
        return false;
    }

    if (mmAccess(path.c_str()) == EN_OK) {
        return true;
    }

    return false;
}

static int32_t CreateDir(const std::string &path) {
    if (path.empty()) {
        UDF_LOG_INFO("create dir input empty");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    if (IsFileExist(path)) {
        return FLOW_FUNC_SUCCESS;
    }
    std::string dir = GetFileDir(path);
    if (dir.empty()) {
        UDF_LOG_ERROR("Get file dir %s failed", path.c_str());
        return FLOW_FUNC_FAILED;
    }

    auto ret = CreateDir(dir);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Create dir failed, ret: %d", ret);
        return ret;
    }

    if (!IsFileExist(path)) {
        if ((mmMkdir(path.c_str(), static_cast<mmMode_t>(kDefaultPathMode)) != EN_OK) && (errno != EEXIST)) {
            UDF_LOG_ERROR("mkdir %s failed", path.c_str());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    }

    return FLOW_FUNC_SUCCESS;
}

static int32_t WriteFile(const std::string &file_name, void *const data, uint64_t len) {
    if (file_name.empty()) {
        UDF_LOG_ERROR("file name nullptr");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (data == nullptr) {
        UDF_LOG_ERROR("data is nullptr");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    int fd = mmOpen2(file_name.c_str(), O_APPEND | M_RDWR | M_CREAT, M_IREAD | M_IWRITE);
    if (fd < 0) {
        UDF_LOG_ERROR("Open file %s failed", file_name.c_str());
        return FLOW_FUNC_FAILED;
    }

    int32_t ret = FLOW_FUNC_SUCCESS;
    const int32_t mmpa_ret = static_cast<int32_t>(mmWrite(fd, data, static_cast<uint32_t>(len)));
    // mmWrite return -1:Failed to write data to file；return -2:Invalid parameter
    if ((mmpa_ret == -1) || (mmpa_ret == -2)) {
        UDF_LOG_ERROR("[WriteData]Failed, errno:%d", mmpa_ret);
        ret = FLOW_FUNC_FAILED;
    }

    if (mmClose(fd) != EN_OK) {
        UDF_LOG_ERROR("[CloseFile]Failed, file %s", file_name.c_str());
        ret = FLOW_FUNC_FAILED;
    }
    return ret;
}
}  // namespace

UdfDumpTaskHost::UdfDumpTaskHost(const std::string &flow_func_info, const int32_t host_pid, const uint32_t device_id)
    : UdfDumpTask(),
      op_name_(flow_func_info),
      host_pid_(host_pid),
      device_id_(device_id) { }

int32_t UdfDumpTaskHost::PreProcessInput(const std::vector<MbufTensor *> &tensors) {
    dumpNS::DumpData dump_data;
    UDF_LOG_INFO("PreProcessInput op name[%s], input size %zu on host", op_name_.c_str(), tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &item = tensors.at(i);
        dumpNS::OpInput * const op_input = dump_data.add_input();
        if (op_input == nullptr) {
            UDF_LOG_ERROR("op name[%s], call protobuf function to add input failed on host", op_name_.c_str());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }

        if (item == nullptr) {
            op_input->set_offset(0);
            op_input->set_size(0);
            op_input->set_address(0);
            continue;
        }

        op_input->set_data_type(UdfDumpCommonUtils::GetDumpDataType(item->GetDataType()));
        op_input->set_format(dumpNS::FORMAT_ND);
        const auto shape = item->GetShape();
        dumpNS::Shape * const in_shape = op_input->mutable_shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            in_shape->add_dim(static_cast<uint64_t>(shape[j]));
        }
        op_input->set_offset(0);
        op_input->set_size(item->GetDataSize());
        const uint64_t addr = reinterpret_cast<uintptr_t>(item->GetData());
        op_input->set_address(addr);
    }
    dump_data.set_version("2.0");
    dump_data.set_op_name(op_name_);
    base_dump_data_ = std::move(dump_data);
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskHost::PreProcessOutput(uint32_t index, const MbufTensor *tensor) {
    dumpNS::DumpData dump_data;
    UDF_LOG_INFO("PreProcessOutput op name[%s], output index %u on host", op_name_.c_str(), index);
    for (uint32_t i = 0; i <= index; ++i) {
        dumpNS::OpOutput * const op_output = dump_data.add_output();
        if (op_output == nullptr) {
            UDF_LOG_ERROR("op name[%s], call protobuf function to add output failed on host", op_name_.c_str());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }

        if (i != index) {
            op_output->set_offset(0);
            op_output->set_address(0);
            op_output->set_size(0);
            continue;
        }

        op_output->set_data_type(UdfDumpCommonUtils::GetDumpDataType(tensor->GetDataType()));
        op_output->set_format(dumpNS::FORMAT_ND);
        const auto shape = tensor->GetShape();
        dumpNS::Shape * const out_shape = op_output->mutable_shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            out_shape->add_dim(static_cast<uint64_t>(shape[j]));
        }
        op_output->set_offset(0);
        op_output->set_size(tensor->GetDataSize());
        const uint64_t addr = reinterpret_cast<uintptr_t>(tensor->GetData());
        op_output->set_address(addr);
    }
    dump_data.set_version("2.0");
    dump_data.set_op_name(op_name_);
    base_dump_data_ = std::move(dump_data);
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskHost::DumpInputData(const dumpNS::DumpData &dump_data,
                                       const std::string &dump_file) const {
    for (int64_t i = 0U; i < dump_data.input_size(); i++) {
        auto &input = dump_data.input(i);
        uint64_t len = input.size();
        if (len == 0U) {
            UDF_LOG_INFO("op name[%s], input[%ld] data size is zero", op_name_.c_str(), i);
            continue;
        }

        const uint64_t base_addr = input.address();
        uint64_t data_addr = base_addr;
        UDF_LOG_INFO("op name[%s], input[%ld] size[%llu].", op_name_.c_str(), i, len);
        if (data_addr == 0U) {
            UDF_LOG_WARN("op name[%s], input[%ld] addr is null", op_name_.c_str(), i);
            continue;
        }

        if (WriteFile(dump_file, reinterpret_cast<void *>(static_cast<uintptr_t>(data_addr)),
                      len) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("[DumpInputData] Dump the %ld input data of op [%s] failed",
                          i, op_name_.c_str());
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskHost::DumpOutputData(const dumpNS::DumpData &dump_data,
                                        const std::string &dump_file) const {
    for (int64_t i = 0U; i < dump_data.output_size(); i++) {
        auto &output = dump_data.output(i);
        uint64_t len = output.size();
        if (len == 0U) {
            UDF_LOG_INFO("op name[%s], output[%ld] data size is zero", op_name_.c_str(), i);
            continue;
        }

        const uint64_t base_addr = output.address();
        uint64_t data_addr = base_addr;
        UDF_LOG_INFO("op name[%s], output[%ld] size[%llu].", op_name_.c_str(), i, len);
        if (data_addr == 0U) {
            UDF_LOG_WARN("op name[%s], output[%ld] addr is null", op_name_.c_str(), i);
            continue;
        }

        if (WriteFile(dump_file, reinterpret_cast<void *>(static_cast<uintptr_t>(data_addr)),
                      len) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("[DumpOutputData] Dump the %ld output data of op [%s] failed",
                          i, op_name_.c_str());
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskHost::DoDumpTensorHost(const std::string &dump_file_path) {
    UDF_LOG_INFO("op name[%s], start to dump on host, path[%s]", op_name_.c_str(), dump_file_path.c_str());
    if (CreateDir(dump_path_) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("op name[%s], create dir [%s]failed.", op_name_.c_str(), dump_path_.c_str());
        return FLOW_FUNC_FAILED;
    }

    uint64_t proto_size = base_dump_data_.ByteSizeLong();
    const std::unique_ptr<char[]> proto_msg(new(std::nothrow) char[proto_size]);
    if (proto_msg == nullptr) {
        UDF_LOG_ERROR("DoDumpTensorHost proto msg null");
        return FLOW_FUNC_FAILED;
    }
    const bool ret = base_dump_data_.SerializeToArray(proto_msg.get(), static_cast<int32_t>(proto_size));
    if ((!ret) || (proto_size == 0U)) {
        UDF_LOG_ERROR("DoDumpTensorHost Dump data proto serialize failed");
        return FLOW_FUNC_FAILED;
    }

    uint64_t dump_size = static_cast<uint64_t>(sizeof(uint64_t));
    if (WriteFile(dump_file_path, &proto_size, dump_size) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to dump proto size");
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_INFO("Dump_size: %lu", proto_size);
    if (WriteFile(dump_file_path, proto_msg.get(), proto_size) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to dump proto msg");
        return FLOW_FUNC_FAILED;
    }
    if (DumpInputData(base_dump_data_, dump_file_path) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to dump input");
        return FLOW_FUNC_FAILED;
    }
    if (DumpOutputData(base_dump_data_, dump_file_path) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to dump output");
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_INFO("op name[%s], dump data success, path[%s]", op_name_.c_str(), dump_file_path.c_str());
    return FLOW_FUNC_SUCCESS;
}

std::string UdfDumpTaskHost::DumpPath(const uint64_t now_time, const uint32_t step_id) {
    std::ostringstream oss;
    oss << base_dump_path_;
    std::string op_name = op_name_;
    UdfDumpCommonUtils::ReplaceStringElem(op_name);
    oss << device_id_ << "/" << op_name << "/" << 0 << "/" << step_id << "/";
    dump_path_ = oss.str();
    UDF_LOG_INFO("full dump path is [%s]", dump_path_.c_str());
    oss << op_name << "." << now_time;

    return oss.str();
}

int32_t UdfDumpTaskHost::DumpOpInfo(const std::string &dump_file_path, const uint32_t step_id) {
    base_dump_path_ = dump_file_path;
    UDF_LOG_INFO("DumpOpInfo, baseDumpPath[%s].", base_dump_path_.c_str());

    if ((base_dump_data_.ByteSizeLong() > static_cast<uint64_t>(INT_MAX)) || (base_dump_data_.ByteSizeLong() == 0U)) {
        UDF_LOG_ERROR("op name[%s], dump data size[%zuB] should be in [1B, 2GB)].",
            op_name_.c_str(), base_dump_data_.ByteSizeLong());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    UDF_LOG_INFO("op name[%s], proto buffer total bytes[%llu]", op_name_.c_str(), base_dump_data_.ByteSizeLong());

    const uint64_t now_time = FlowFuncTimer::Instance().GetCurrentTimestamp();
    base_dump_data_.set_dump_time(now_time);
    // dump file path name
    const std::string dump_file_name = DumpPath(now_time, step_id);
    return DoDumpTensorHost(dump_file_name);
}
}  // namespace FlowFunc
