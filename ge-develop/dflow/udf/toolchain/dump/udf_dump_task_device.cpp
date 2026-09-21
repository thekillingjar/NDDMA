/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_dump_task_device.h"
#include <securec.h>
#include "common/udf_log.h"
#include "common/scope_guard.h"
#include "flow_func/flow_func_timer.h"
#include "flow_func/flow_func_config_manager.h"
#include "udf_dump_common_utils.h"
#include "ascend_hal.h"
#include "aicpu/aicpu_schedule/aicpusd_info.h"

namespace FlowFunc {
namespace {
const uint64_t kDumpSliceSize = 128UL << 20U; // 分片大小
// DATA_DUMP_GRUOP_ID， EVENT_CCPU_CTRL_MSG， AICPU_SUB_EVENT_REPORT_UDF_DUMPDATA固定，与aicpu保持一致
constexpr uint32_t kDataDumpGroupId = 31U;
#define EVENT_CCPU_CTRL_MSG 19
#define AICPU_SUB_EVENT_REPORT_UDF_DUMPDATA 15
constexpr uint32_t  kDataDumpTimeoutInterval = 3600000U;
}  // namespace

UdfDumpTaskDevice::UdfDumpTaskDevice(const std::string &flow_func_info, const int32_t host_pid, const uint32_t device_id,
                                     const int32_t aicpu_pid, const uint32_t logic_dev_id)
    : UdfDumpTask(),
      op_name_(flow_func_info),
      input_total_size_(0U),
      output_total_size_(0U),
      buff_(nullptr),
      buff_size_(0U),
      offset_(0U),
      host_pid_(host_pid),
      device_id_(device_id),
      aicpu_pid_(aicpu_pid),
      logic_dev_id_(logic_dev_id) {}

int32_t UdfDumpTaskDevice::PreProcessInput(const std::vector<MbufTensor *> &tensors) {
    dumpNS::DumpData dump_data;
    UDF_LOG_INFO("PreProcessInput op name[%s], input size %zu on device", op_name_.c_str(), tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &item = tensors.at(i);
        dumpNS::OpInput * const op_input = dump_data.add_input();
        if (op_input == nullptr) {
            UDF_LOG_ERROR("op name[%s], call protobuf function to add input failed on device", op_name_.c_str());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }

        if (item == nullptr) {
            op_input->set_offset(0);
            op_input->set_size(0);
            input_total_size_ += 0;
            op_input->set_address(0);
            continue;
        }

        op_input->set_data_type(UdfDumpCommonUtils::GetDumpDataType(item->GetDataType()));
        op_input->set_format(dumpNS::FORMAT_ND);
        op_input->set_offset(0);
        const auto shape = item->GetShape();
        dumpNS::Shape * const inShape = op_input->mutable_shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            inShape->add_dim(static_cast<uint64_t>(shape[j]));
        }

        op_input->set_size(item->GetDataSize());
        input_total_size_ += item->GetDataSize();
        const uint64_t addr = reinterpret_cast<uintptr_t>(item->GetData());
        op_input->set_address(addr);
    }
    dump_data.set_version("2.0");
    dump_data.set_op_name(op_name_);
    base_dump_data_ = std::move(dump_data);
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskDevice::PreProcessOutput(uint32_t index, const MbufTensor *tensor) {
    dumpNS::DumpData dump_data;
    UDF_LOG_INFO("PreProcessOutput op name[%s], output index %u on device", op_name_.c_str(), index);
    for (uint32_t i = 0; i <= index; ++i) {
        dumpNS::OpOutput * const op_output = dump_data.add_output();
        if (op_output == nullptr) {
            UDF_LOG_ERROR("op name[%s], call protobuf function to add output failed on device", op_name_.c_str());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }

        if (i != index) {
            op_output->set_offset(0);
            op_output->set_address(0);
            output_total_size_ += 0;
            op_output->set_size(0);
            continue;
        }

        op_output->set_data_type(UdfDumpCommonUtils::GetDumpDataType(tensor->GetDataType()));
        op_output->set_format(dumpNS::FORMAT_ND);
        op_output->set_offset(0);

        const auto shape = tensor->GetShape();
        dumpNS::Shape * const outShape = op_output->mutable_shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            outShape->add_dim(static_cast<uint64_t>(shape[j]));
        }

        op_output->set_size(tensor->GetDataSize());
        output_total_size_ += tensor->GetDataSize();
        const uint64_t addr = reinterpret_cast<uintptr_t>(tensor->GetData());
        op_output->set_address(addr);
    }
    dump_data.set_version("2.0");
    dump_data.set_op_name(op_name_);
    base_dump_data_ = std::move(dump_data);
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskDevice::SendDumpMessageToAicpuSd() const {
    uint64_t flag = (static_cast<uint64_t>(logic_dev_id_) << 32UL) | static_cast<uint64_t>(BUFF_SP_HUGEPAGE_PRIOR) |
                    static_cast<uint64_t>(BUFF_SP_DVPP);
    int32_t mem_group_id = FlowFuncConfigManager::GetConfig()->GetMemGroupId();
    Mbuf *mbuf = nullptr;
    auto drv_ret = halMbufAllocEx(buff_size_, 64U, flag, mem_group_id, &mbuf);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halMbufAllocEx failed, drv_ret=%d, dataSize=%lu, flag=%lu, groupId=%d.", drv_ret,
            buff_size_, flag, mem_group_id);
        return FLOW_FUNC_FAILED;
    }
    auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
    std::unique_ptr<Mbuf, decltype(mbuf_deleter)> mbuf_guard(mbuf, mbuf_deleter);

    void *dst_data_buf = nullptr;
    drv_ret = halMbufGetBuffAddr(mbuf, &dst_data_buf);
    if ((drv_ret != static_cast<int32_t>(DRV_ERROR_NONE)) || (dst_data_buf == nullptr)) {
        UDF_LOG_ERROR("Fail to get buff addr for mbuf, ret=[%d].", drv_ret);
        return FLOW_FUNC_FAILED;
    }
    errno_t e_ret = memcpy_s(dst_data_buf, buff_size_, reinterpret_cast<void *>(buff_.get()), offset_);
    if (e_ret != EOK) {
        UDF_LOG_ERROR("Fail to memcpy, ret=[%d].", e_ret);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    event_summary drv_event = {};
    AICPUDumpUdfInfo msg = {};
    msg.udfInfo = reinterpret_cast<uintptr_t>(dst_data_buf);
    msg.length = buff_size_;
    msg.udfPid = getpid();
    drv_event.dst_engine = CCPU_DEVICE;
    drv_event.policy = ONLY;
    drv_event.pid = aicpu_pid_;
    drv_event.grp_id = kDataDumpGroupId;
    drv_event.event_id = static_cast<EVENT_ID>(EVENT_CCPU_CTRL_MSG);
    drv_event.subevent_id = static_cast<uint32_t>(AICPU_SUB_EVENT_REPORT_UDF_DUMPDATA);
    drv_event.msg_len = static_cast<uint32_t>(sizeof(struct event_sync_msg) + sizeof(msg));
    auto event_msg = std::make_unique<char[]>(drv_event.msg_len);
    e_ret = memcpy_s(reinterpret_cast<void *>(event_msg.get() + sizeof(struct event_sync_msg)), sizeof(msg),
                    reinterpret_cast<void *>(&msg), sizeof(msg));
    if (e_ret != EOK) {
        UDF_LOG_ERROR("Fail to memcpy, ret=[%d].", e_ret);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    drv_event.msg = event_msg.get();
    struct event_reply reply = {};
    struct event_proc_result result = {};
    reply.buf = reinterpret_cast<char *>(&result);
    reply.buf_len = sizeof(struct event_proc_result);
    drv_ret = halEschedSubmitEventSync(logic_dev_id_, &drv_event, kDataDumpTimeoutInterval, &reply);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halEschedSubmitEventSync failed, ret:%d, devid:%u", drv_ret, device_id_);
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskDevice::ProcessInputDump(const dumpNS::DumpData &dump_data) {
    if (dump_data.input_size() == 0) {
        return FLOW_FUNC_SUCCESS;
    }
    return SendDumpMessageToAicpuSd();
}


int32_t UdfDumpTaskDevice::ProcessOutputDump(const dumpNS::DumpData &dump_data) {
    if (dump_data.output_size() == 0) {
        return FLOW_FUNC_SUCCESS;
    }
    return SendDumpMessageToAicpuSd();
}

int32_t UdfDumpTaskDevice::DoDumpTensor(const std::string &dump_file_path) {
    int32_t ret = ProcessInputDump(base_dump_data_);
    if (ret != FLOW_FUNC_SUCCESS) {
        return ret;
    }

    ret = ProcessOutputDump(base_dump_data_);
    if (ret != FLOW_FUNC_SUCCESS) {
        return ret;
    }

    UDF_LOG_INFO("op name[%s], dump data success, path[%s]", op_name_.c_str(), dump_file_path.c_str());
    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDumpTaskDevice::ProcessDumpTensor(const std::string &dump_file_path) {
    // data size + data + path size + path
    buff_size_ = sizeof(uint64_t) + base_dump_data_.ByteSizeLong() + sizeof(uint64_t) + dump_file_path.length();
    buff_.reset(new (std::nothrow) char[buff_size_]);
    if (buff_ == nullptr) {
        UDF_LOG_ERROR("op name[%s], malloc buffer for data dump faild, size[%llu]", op_name_.c_str(), buff_size_);
        return FLOW_FUNC_FAILED;
    }
    // for memory statistic
    UDF_LOG_INFO("op name[%s], MallocMemory, func=new, size=%llu", op_name_.c_str(), buff_size_);
    uint64_t * const data_size_ptr = reinterpret_cast<uint64_t *>(buff_.get());
    *data_size_ptr = base_dump_data_.ByteSizeLong();
    offset_ = sizeof(uint64_t);
    if (!base_dump_data_.SerializeToArray(buff_.get() + offset_,
        static_cast<int32_t>(base_dump_data_.ByteSizeLong()))) {
        UDF_LOG_ERROR("op name[%s], serialize dump data to string failed, data size[%zuB].",
            op_name_.c_str(), base_dump_data_.ByteSizeLong());
        return FLOW_FUNC_FAILED;
    }
    offset_ += base_dump_data_.ByteSizeLong();

    const uint64_t dump_file_path_length = dump_file_path.length();
    // can not change to uint64_t ptr to assign value, as (buff_.get() + offset_) maybe not 8 byte alignment.
    errno_t ret =
        memcpy_s(buff_.get() + offset_, buff_size_ - offset_, &dump_file_path_length, sizeof(dump_file_path_length));
    if (ret != EOK) {
      UDF_LOG_ERROR("op name[%s], memcpy dump path length failed, buff_size_=%lu, offset_=%lu", op_name_.c_str(),
                    buff_size_, offset_);
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    offset_ += sizeof(uint64_t);
    ret = memcpy_s(buff_.get() + offset_, buff_size_ - offset_, dump_file_path.c_str(), dump_file_path_length);
    if (ret != EOK) {
        UDF_LOG_ERROR("op name[%s], memcpy dump path failed, path[%s]", op_name_.c_str(), dump_file_path.c_str());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    offset_ += dump_file_path_length;

    return DoDumpTensor(dump_file_path);
}

std::string UdfDumpTaskDevice::DumpPath(const uint64_t now_time, const uint32_t step_id) {
    std::ostringstream oss;
    oss << base_dump_path_;
    std::string opName = op_name_;
    UdfDumpCommonUtils::ReplaceStringElem(opName);
    oss << device_id_ << "/" << opName << "/" << 0 << "/" << step_id << "/" << opName << "." << now_time;
    return oss.str();
}

int32_t UdfDumpTaskDevice::DumpOpInfo(const std::string &dump_file_path, const uint32_t step_id) {
    base_dump_path_ = dump_file_path;
    UDF_LOG_INFO("DumpOpInfo, baseDumpPath[%s].", base_dump_path_.c_str());

    const uint64_t now_time = FlowFuncTimer::Instance().GetCurrentTimestamp();
    base_dump_data_.set_dump_time(now_time);

    if ((base_dump_data_.ByteSizeLong() > static_cast<uint64_t>(INT_MAX)) || (base_dump_data_.ByteSizeLong() == 0U)) {
        UDF_LOG_ERROR("op name[%s], dump data size[%zuB] should be in [1B, 2GB)].",
            op_name_.c_str(), base_dump_data_.ByteSizeLong());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    UDF_LOG_INFO("op name[%s], proto buffer total bytes[%llu]", op_name_.c_str(), base_dump_data_.ByteSizeLong());

    // dump file path name
    const std::string dump_file_name = DumpPath(now_time, step_id);
    return ProcessDumpTensor(dump_file_name);
}
}  // namespace FlowFunc
