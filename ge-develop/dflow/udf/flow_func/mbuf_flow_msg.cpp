/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mbuf_flow_msg.h"
#include <iomanip>
#include "securec.h"
#include "common/udf_log.h"
#include "common/util.h"
#include "common/data_utils.h"
#include "flow_func_config_manager.h"

namespace FlowFunc {
uint64_t FlowMsg::GetTransactionId() const {
    UDF_LOG_ERROR("Not supported, please implement the GetTransactionId method.");
    return UINT64_MAX;
}

void FlowMsg::SetTransactionId(uint64_t transactionId) {
    (void)transactionId;
    UDF_LOG_ERROR("Not supported, please implement the SetTransactionId method.");
}

const std::vector<Tensor *> FlowMsg::GetTensorList() const {
    UDF_LOG_ERROR("Not supported, please implement the GetTensorList method.");
    return {};
}

int32_t FlowMsg::GetRawData(void *&dataPtr, uint64_t &dataSize) const {
    (void)dataPtr;
    (void)dataSize;
    UDF_LOG_ERROR("Not supported, please implement the GetRawData method.");
    return FLOW_FUNC_ERR_NOT_SUPPORT;
}

void FlowMsg::SetMsgType(MsgType msgType) {
    (void)msgType;
    UDF_LOG_ERROR("Not supported, please implement the SetMsgType method.");
    return;
}

int32_t Tensor::Reshape(const std::vector<int64_t> &shape) {
    (void)shape;
    UDF_LOG_ERROR("Not supported, please implement the Reshape method.");
    return FLOW_FUNC_ERR_NOT_SUPPORT;
}

uint64_t Tensor::GetDataBufferSize() const {
    UDF_LOG_ERROR("Not supported, please implement the GetDataBuffSize method.");
    return 0UL;
}

std::string MbufHeadMsg::DebugString() const {
    std::ostringstream debug_info;
    debug_info << "trans_id:" << trans_id << ", version:" << version <<
              ", msg_type:" << msg_type << ", ret_code:" << ret_code <<
              ", start_time:" << start_time << ", end_time:" << end_time << ", flags:" << flags <<
              ", data_flag:" << static_cast<uint32_t>(data_flag) << ", data_label=" << data_label <<
              ", route_label=" << route_label;
    return debug_info.str();
}

std::string RuntimeTensorDesc::DebugString() const {
    std::ostringstream debug_info;
    debug_info << "dataAddr:" << dataAddr << ", dataOffsetSize:" << dataOffsetSize << ", dtype:" << dtype <<
              ", shape size:" << shape[0] << ", shape[";
    int64_t shape_size = std::min(shape[0], static_cast<int64_t>(kMaxDimSize));
    for (int64_t dim = 0; dim < shape_size; dim++) {
        if (dim != 0) {
            debug_info << ",";
        }
        debug_info << shape[dim + 1];
    }
    debug_info << "], format:" << format << ", data_size=" << data_size;
    return debug_info.str();
}

std::shared_ptr<Tensor> FlowBufferFactory::AllocTensor(const std::vector<int64_t> &shape,
    TensorDataType dataType, uint32_t align) {
    auto device_id = FlowFuncConfigManager::GetConfig()->GetDeviceId();
    MbufHead head = {};
    std::shared_ptr<Tensor> tensor = nullptr;
    auto tensor_msg = MbufFlowMsg::AllocTensorMsg(shape, dataType, device_id, head, align);
    if (tensor_msg != nullptr) {
        tensor = tensor_msg->GetSharedTensor();
    }
    return tensor;
}

MbufTensor::MbufTensor(std::shared_ptr<Mbuf> mbuf, void *mbuf_data, uint64_t mbuf_data_size)
    : Tensor(), mbuf_(mbuf), mbuf_data_(mbuf_data), mbuf_data_size_(mbuf_data_size) {
    if (mbuf_data_size_ >= sizeof(RuntimeTensorDesc)) {
        tensor_desc_ = reinterpret_cast<RuntimeTensorDesc *>(mbuf_data_);
        data_ = static_cast<uint8_t *>(mbuf_data_) + sizeof(RuntimeTensorDesc);
        data_size_ = mbuf_data_size_ - sizeof(RuntimeTensorDesc);
    }
}

MbufTensor::MbufTensor(std::shared_ptr<Mbuf> mbuf, void *mbuf_data, uint64_t mbuf_data_size, uint64_t data_size)
    : Tensor(), mbuf_(mbuf), mbuf_data_(mbuf_data), mbuf_data_size_(mbuf_data_size), data_size_(data_size) {
    if ((mbuf_data_size_ >= sizeof(RuntimeTensorDesc))) {
        tensor_desc_ = reinterpret_cast<RuntimeTensorDesc *>(mbuf_data_);
        data_ = static_cast<uint8_t *>(mbuf_data_) + sizeof(RuntimeTensorDesc);
    }
}

MbufTensor::~MbufTensor() noexcept {
    mbuf_.reset();
    mbuf_data_ = nullptr;
    mbuf_data_size_ = 0UL;
    tensor_desc_ = nullptr;
    data_ = nullptr;
    data_size_ = 0UL;
}

int32_t MbufTensor::InitRuntimeTensorDesc(const std::vector<int64_t> &shape, TensorDataType data_type) {
    shape_ = shape;
    if ((mbuf_data_size_ < sizeof(RuntimeTensorDesc)) || (data_size_ > mbuf_data_size_ - sizeof(RuntimeTensorDesc))) {
        UDF_LOG_ERROR("Data buffer size=%lu or data size=%lu is invalid.", mbuf_data_size_, data_size_);
        return FLOW_FUNC_FAILED;
    }
    return InitRuntimeTensorDesc(tensor_desc_, shape, data_type, mbuf_data_size_ - sizeof(RuntimeTensorDesc));
}

int32_t MbufTensor::InitRuntimeTensorDesc(RuntimeTensorDesc *tensor_desc, const std::vector<int64_t> &shape,
    TensorDataType data_type, uint64_t data_size) {
    if (tensor_desc == nullptr) {
        UDF_LOG_ERROR("tensor desc is null.");
        return FLOW_FUNC_FAILED;
    }

    (void)memset_s(tensor_desc, sizeof(RuntimeTensorDesc), '\0', sizeof(RuntimeTensorDesc));
    tensor_desc->dtype = static_cast<int64_t>(data_type);
    tensor_desc->format = 3; // ge FORMAT_ND is 3
    tensor_desc->shape[0] = static_cast<int64_t>(shape.size());
    for (size_t i = 0UL; i < shape.size(); ++i) {
        tensor_desc->shape[i + 1] = shape[i];
    }
    tensor_desc->data_size = data_size;
    return FLOW_FUNC_SUCCESS;
}

int32_t MbufTensor::ParseRuntimeTensorDesc() {
    if (tensor_desc_ == nullptr) {
        UDF_LOG_ERROR("tensor desc is null, mbuf_data_size_=%lu.", mbuf_data_size_);
        return FLOW_FUNC_FAILED;
    }
    int64_t dim_num = tensor_desc_->shape[0];
    if (dim_num < 0 || dim_num > static_cast<int64_t>(kMaxDimSize)) {
        UDF_LOG_ERROR("tensor desc shape is invalid, dim_num=%ld, range=[0, %u].", dim_num, kMaxDimSize);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    shape_.reserve(static_cast<size_t>(dim_num));
    for (int64_t i = 1; i <= dim_num; ++i) {
        shape_.emplace_back(tensor_desc_->shape[i]);
    }

    int64_t calc_size = CalcDataSize(shape_, static_cast<TensorDataType>(tensor_desc_->dtype));
    if (calc_size < 0) {
        UDF_LOG_ERROR("CalcDataSize failed, calc_size=%ld.", calc_size);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (static_cast<uint64_t>(calc_size) > data_size_) {
        UDF_LOG_ERROR(
            "data size is less than shape and data type need size, data_type=%ld, calc_size=%ld, data_size=%lu.",
            tensor_desc_->dtype, calc_size, data_size_);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (data_size_ > static_cast<uint64_t>(calc_size)) {
        UDF_LOG_WARN("data size is over than shape and data type need size, so fix data_size=%lu to calc_size=%ld.",
            data_size_, calc_size);
        data_size_ = static_cast<uint64_t>(calc_size);
    }
    return FLOW_FUNC_SUCCESS;
}

int64_t MbufTensor::GetElementCnt() const {
    return CalcElementCnt(shape_);
}

uint64_t MbufTensor::GetDataBufferSize() const {
    return mbuf_data_size_ > sizeof(RuntimeTensorDesc) ? (mbuf_data_size_ - sizeof(RuntimeTensorDesc)) : 0UL;
}

int32_t MbufTensor::Reshape(const std::vector<int64_t> &shape) {
    if (tensor_desc_ == nullptr) {
        UDF_LOG_ERROR("tensor desc is null.");
        return FLOW_FUNC_FAILED;
    }
    if (shape.size() > kMaxDimSize) {
        UDF_LOG_ERROR("Limit max dim size is %u, but shape size=%zu, shape=%s", kMaxDimSize, shape.size(),
            ToString(shape).c_str());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    int64_t origin_element_cnt = CalcElementCnt(shape_);
    int64_t element_cnt = CalcElementCnt(shape);
    if (origin_element_cnt != element_cnt) {
        UDF_LOG_ERROR("Reshape must be same as origin element count, origin shape=%s, reshape shape=%s",
            ToString(shape_).c_str(), ToString(shape).c_str());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    tensor_desc_->shape[0] = static_cast<int64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        tensor_desc_->shape[i + 1] = shape[i];
    }
    UDF_LOG_INFO("Reshape tensor shape from %s to %s", ToString(shape_).c_str(), ToString(shape).c_str());
    shape_ = shape;
    return FLOW_FUNC_SUCCESS;
}

void MbufTensor::PrintBytesHex(const uint8_t *bytes_data, uint64_t size, std::ostringstream &ss) {
    for (uint64_t i = 0UL; i < size; ++i) {
        // byte hex is 2
        ss << std::setfill('0') << std::setw(2) << std::hex << static_cast<uint32_t>(bytes_data[i]) << ' ';
    }
}

std::string MbufTensor::DebugString() const {
    std::ostringstream debug_info;
    debug_info << "mbuf_data_size_:" << mbuf_data_size_ << ", tensor_desc:";
    if (tensor_desc_ == nullptr) {
        debug_info << "NULL";
    } else {
        debug_info << '{' << tensor_desc_->DebugString() << '}';
    }

    debug_info << ", data_size:" << data_size_;
    if (data_ == nullptr) {
        debug_info << ", data:NULL";
    } else {
        // dump first 16 bytes and last 16 bytes
        constexpr uint64_t dumpSize = 16UL;
        debug_info << ", hex data:[";
        if (data_size_ <= dumpSize * 2U) {
            PrintBytesHex(static_cast<uint8_t *>(data_), data_size_, debug_info);
        } else {
            PrintBytesHex(static_cast<uint8_t *>(data_), dumpSize, debug_info);
            debug_info << "... ";
            PrintBytesHex(static_cast<uint8_t *>(data_) + (data_size_ - dumpSize), dumpSize, debug_info);
        }
        debug_info << ']';
    }
    return debug_info.str();
}

MbufFlowMsg::MbufFlowMsg(std::shared_ptr<Mbuf> mbuf) : FlowMsg(), mbuf_(mbuf) {
}

MbufFlowMsg::~MbufFlowMsg() {
    mbuf_.reset();
}

int32_t MbufFlowMsg::SetMbufHead(Mbuf *mbuf, const MbufHead &mbuf_head) {
    void *head_buf = nullptr;
    uint32_t head_buf_len = 0U;
    auto drv_ret = halMbufGetPrivInfo(mbuf, &head_buf, &head_buf_len);
    if (drv_ret != DRV_ERROR_NONE || head_buf == nullptr) {
        UDF_LOG_ERROR("Failed to get mbuf priv info, ret[%d].", drv_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    if (mbuf_head.head_buf_len == 0U) {
        const auto ret = memset_s(head_buf, static_cast<size_t>(head_buf_len), 0, static_cast<size_t>(head_buf_len));
        if (ret != EOK) {
            UDF_LOG_ERROR("Failed to memset mbuf priv info, ret[%d].", ret);
            return FLOW_FUNC_FAILED;
        }
    } else if (mbuf_head.head_buf_len != head_buf_len) {
        UDF_LOG_ERROR("Failed to alloc mbuf, input mbuf head len[%u], but current mbuf head len[%u].",
            mbuf_head.head_buf_len, head_buf_len);
        return FLOW_FUNC_FAILED;
    } else {
        const auto ret = memcpy_s(head_buf, static_cast<size_t>(head_buf_len), mbuf_head.head_buf,
            static_cast<size_t>(mbuf_head.head_buf_len));
        if (ret != EOK) {
            UDF_LOG_ERROR("Failed to memcpy mbuf priv info, ret[%d].", ret);
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

Mbuf *MbufFlowMsg::AllocMbuf(uint64_t alloc_size, uint32_t device_id, const MbufHead &mbuf_head, uint32_t align) {
    // udf may alloc memory fo dvpp, so need add dvpp flag.
    uint64_t flag = (static_cast<uint64_t>(device_id) << 32UL) | static_cast<uint64_t>(BUFF_SP_HUGEPAGE_PRIOR) |
                    static_cast<uint64_t>(BUFF_SP_DVPP);
    int32_t mem_group_id = FlowFuncConfigManager::GetConfig()->GetMemGroupId();
    Mbuf *mbuf = nullptr;
    // alignSize use 64
    auto drv_ret = halMbufAllocEx(alloc_size, align, flag, mem_group_id, &mbuf);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halMbufAllocEx failed, drv_ret=%d, data_size=%lu, align=%u, flag=%lu, groupId=%d.", drv_ret,
            alloc_size, align, flag, mem_group_id);
        return nullptr;
    }

    auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
    std::unique_ptr<Mbuf, decltype(mbuf_deleter)> mbufGuard(mbuf, mbuf_deleter);

    drv_ret = halMbufSetDataLen(mbuf, alloc_size);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halMbufSetDataLen failed, drv_ret=%d, data_size=%lu.", drv_ret, alloc_size);
        return nullptr;
    }

    auto set_ret = SetMbufHead(mbuf, mbuf_head);
    if (set_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to set mbuf head, ret=%d.", set_ret);
        return nullptr;
    }

    // mbuf_flow_msg take ownership of mbuf
    (void)mbufGuard.release();

    UDF_LOG_DEBUG("Alloc mbuf success, data_size=%lu.", alloc_size);
    return mbuf;
}

int32_t MbufFlowMsg::InitMbufTensor(const std::vector<int64_t> &shape,
    TensorDataType data_type, void *mbuf_addr, uint64_t mbuf_len, uint64_t data_size) {
    std::shared_ptr<MbufTensor> mbuf_tensor;
    if (data_size == 0UL) {
        mbuf_tensor.reset(new(std::nothrow) MbufTensor(mbuf_, mbuf_addr, mbuf_len));
    } else {
        mbuf_tensor.reset(new(std::nothrow) MbufTensor(mbuf_, mbuf_addr, mbuf_len, data_size));
    }
    if (mbuf_tensor == nullptr) {
        UDF_LOG_ERROR("alloc MbufTensor failed.");
        return FLOW_FUNC_FAILED;
    }
    mbuf_tensor_list_.emplace_back(mbuf_tensor);
    tensor_list_.emplace_back(mbuf_tensor.get());
    return mbuf_tensor->InitRuntimeTensorDesc(shape, data_type);
}

int32_t MbufFlowMsg::InitMbufTensor(const std::vector<int64_t> &shape, TensorDataType data_type) {
    return InitMbufTensor(shape, data_type, mbuf_info_.mbuf_addr, mbuf_info_.mbuf_len);
}

int32_t MbufFlowMsg::InitMbufTensorList(const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<TensorDataType> &data_types, const std::vector<uint64_t> &data_orig_size,
    const std::vector<uint64_t> &data_align_size) {
    // data_orig_size is not include RuntimeTensorDesc. data_align_size is include RunTimeTensorDesc size.
    if ((shapes.size() != data_types.size()) || (shapes.size() != data_align_size.size()) ||
        (shapes.size() != data_orig_size.size())) {
        UDF_LOG_ERROR("Shape size=%zu datatype size=%zu data_orig_size size=%zu data_orig_size size=%zu should be same",
                      shapes.size(), data_types.size(), data_align_size.size(), data_orig_size.size());
        return FLOW_FUNC_FAILED;
    }
    auto ret = FLOW_FUNC_SUCCESS;
    uint64_t offset = 0UL;
    for (size_t i = 0UL; i < shapes.size(); ++i) {
        void *mbuf_addr = static_cast<void *>(static_cast<uint8_t *>(mbuf_info_.mbuf_addr) + offset);
        ret = InitMbufTensor(shapes[i], data_types[i], mbuf_addr, data_align_size[i], data_orig_size[i]);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Init mbuf tensor for index[%zu] failed.", i);
            return FLOW_FUNC_FAILED;
        }
        offset += data_align_size[i];
    }
    return FLOW_FUNC_SUCCESS;
}

std::shared_ptr<MbufFlowMsg> MbufFlowMsg::CreateFlowMsg(uint64_t size, uint32_t device_id, const MbufHead &mbuf_head,
    uint32_t align) {
    Mbuf *mbuf = AllocMbuf(size, device_id, mbuf_head, align);
    if (mbuf == nullptr) {
        UDF_LOG_ERROR("alloc mbuf failed, size=%lu.", size);
        return nullptr;
    }
    auto mbuf_deleter = [](Mbuf *buf) { (void)halMbufFree(buf); };
    // mbuf_flow_msg take ownership of mbuf
    std::shared_ptr<Mbuf> mbuf_ptr(mbuf, mbuf_deleter);

    std::shared_ptr<MbufFlowMsg> mbuf_flow_msg;
    try {
        mbuf_flow_msg = std::make_shared<MbufFlowMsg>(mbuf_ptr);
    } catch (std::exception &e) {
        UDF_LOG_ERROR("alloc MbufFlowMsg exception, %s.", e.what());
        return nullptr;
    }

    auto parse_ret = mbuf_flow_msg->ParseMbuf();
    if (parse_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("parse mbuf failed, parse_ret=%d.", parse_ret);
        return nullptr;
    }
    return mbuf_flow_msg;
}

std::shared_ptr<MbufFlowMsg> MbufFlowMsg::CreateFlowMsg(std::shared_ptr<Mbuf> mbuf, const MbufHead &mbuf_head) {
    auto set_ret = SetMbufHead(mbuf.get(), mbuf_head);
    if (set_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to set mbuf head, ret=%d.", set_ret);
        return nullptr;
    }

    std::shared_ptr<MbufFlowMsg> mbuf_flow_msg;
    try {
        mbuf_flow_msg = std::make_shared<MbufFlowMsg>(mbuf);
    } catch (std::exception &e) {
        UDF_LOG_ERROR("alloc MbufFlowMsg exception, %s.", e.what());
        return nullptr;
    }

    auto parse_ret = mbuf_flow_msg->ParseMbuf();
    if (parse_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("parse mbuf failed, parse_ret=%d.", parse_ret);
        return nullptr;
    }
    return mbuf_flow_msg;
}

std::shared_ptr<MbufFlowMsg> MbufFlowMsg::AllocEmptyTensorMsg(uint32_t device_id, const MbufHead &mbuf_head) {
    uint64_t alloc_size = sizeof(RuntimeTensorDesc);
    auto mbuf_flow_msg = CreateFlowMsg(alloc_size, device_id, mbuf_head);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("create flow msg failed. size=%ld.", alloc_size);
        return nullptr;
    }

    MbufHeadMsg &head_msg_ref = *(mbuf_flow_msg->head_msg_);
    head_msg_ref.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA);
    head_msg_ref.ret_code = 0;
    // 0 bit is null data flag, 1 is null data, 0 is not null data
    head_msg_ref.data_flag |= 1U;
    auto *tensor_desc = reinterpret_cast<RuntimeTensorDesc *>(mbuf_flow_msg->mbuf_info_.mbuf_addr);
    // empty tensor can't make MbufTensor,just make sure tensor desc valid.
    (void)MbufTensor::InitRuntimeTensorDesc(tensor_desc, {0}, TensorDataType::DT_FLOAT, 0);
    UDF_LOG_DEBUG("Success to alloc empty tensor msg.");
    return mbuf_flow_msg;
}

std::shared_ptr<MbufFlowMsg> MbufFlowMsg::AllocTensorMsg(const std::vector<int64_t> &shape,
    TensorDataType data_type, uint32_t device_id, const MbufHead &mbuf_head, uint32_t align) {
    if (shape.size() > kMaxDimSize) {
        UDF_LOG_ERROR("shape.size=%zu is over max dimSize=%u.", shape.size(), kMaxDimSize);
        return nullptr;
    }
    int64_t data_size = CalcDataSize(shape, data_type);
    if (data_size < 0) {
        UDF_LOG_ERROR("CalcDataSize failed. calc result=%ld.", data_size);
        return nullptr;
    }

    uint64_t alloc_size = static_cast<uint64_t>(data_size) + sizeof(RuntimeTensorDesc);
    auto mbuf_flow_msg = CreateFlowMsg(alloc_size, device_id, mbuf_head, align);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("create flow msg failed. size=%ld.", alloc_size);
        return nullptr;
    }

    MbufHeadMsg &head_msg_ref = *(mbuf_flow_msg->head_msg_);
    head_msg_ref.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA);
    head_msg_ref.ret_code = 0;
    head_msg_ref.data_flag &= ~(1U);
    auto flowRet = mbuf_flow_msg->InitMbufTensor(shape, data_type);
    if (flowRet != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("alloc MbufTensor failed, alloc_size=%lu.", alloc_size);
        return nullptr;
    }
    UDF_LOG_INFO("alloc tensor msg success. alloc_size=%ld.", alloc_size);
    return mbuf_flow_msg;
}

std::shared_ptr<MbufFlowMsg> MbufFlowMsg::ToTensorMsg(const std::vector<int64_t> &shape,
    TensorDataType data_type, std::shared_ptr<Mbuf> mbuf, const MbufHead &mbuf_head) {
    auto mbuf_flow_msg = CreateFlowMsg(mbuf, mbuf_head);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("create flow msg failed.");
        return nullptr;
    }

    MbufHeadMsg &head_msg_ref = *(mbuf_flow_msg->head_msg_);
    head_msg_ref.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA);
    head_msg_ref.ret_code = 0;
    head_msg_ref.data_flag &= ~(1U);
    auto flowRet = mbuf_flow_msg->InitMbufTensor(shape, data_type);
    if (flowRet != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("alloc MbufTensor failed.");
        return nullptr;
    }
    UDF_LOG_INFO("alloc tensor msg success.");
    return mbuf_flow_msg;
}


std::shared_ptr<MbufFlowMsg> MbufFlowMsg::AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<TensorDataType> &data_types, uint32_t device_id, const MbufHead &mbuf_head, uint32_t align) {
    if (shapes.size() != data_types.size()) {
        UDF_LOG_ERROR("shapes size=%zu should be same as data_types size=%zu .", shapes.size(), data_types.size());
        return nullptr;     
    }
    uint64_t alloc_size = 0;
    std::vector<uint64_t> data_orig_size;
    std::vector<uint64_t> data_align_size;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto &shape = shapes[i];
        const auto &data_type = data_types[i];
        if (shape.size() > kMaxDimSize) {
            UDF_LOG_ERROR("shape.size=%zu is over max dimSize=%u.", shape.size(), kMaxDimSize);
            return nullptr;
        }
        int64_t data_size = CalcDataSize(shape, data_type);
        if (data_size < 0) {
            UDF_LOG_ERROR("CalcDataSize failed. calc result=%ld.", data_size);
            return nullptr;
        }
        uint64_t align_data_size = (static_cast<uint64_t>(data_size) + align - 1) & ~(align - 1);
        // algin data size must be greater than data_size , so just check align data size
        if (CheckAddOverflowUint64(alloc_size, sizeof(RuntimeTensorDesc)) ||
            CheckAddOverflowUint64(alloc_size + sizeof(RuntimeTensorDesc), align_data_size)) {
            UDF_LOG_ERROR("Alloc size=%lu add new alignDatasize=%lu and RuntimeTensorDesc is overflow.",
                          alloc_size, align_data_size);
            return nullptr;
        }
        // it is safe result of align data size + RuntimeTensorDesc size + alloc is not overflow.
        data_orig_size.emplace_back(data_size);
        data_align_size.emplace_back(align_data_size + sizeof(RuntimeTensorDesc));
        alloc_size += align_data_size + sizeof(RuntimeTensorDesc);
    }

    auto mbuf_flow_msg = CreateFlowMsg(alloc_size, device_id, mbuf_head, align);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("create flow msg failed. size=%ld.", alloc_size);
        return nullptr;
    }

    MbufHeadMsg &head_msg_ref = *(mbuf_flow_msg->head_msg_);
    head_msg_ref.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST);
    head_msg_ref.ret_code = 0;
    head_msg_ref.data_flag &= ~(1U);
    auto flowRet = mbuf_flow_msg->InitMbufTensorList(shapes, data_types, data_orig_size, data_align_size);
    if (flowRet != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("alloc MbufTensorList failed, alloc_size=%lu.", alloc_size);
        return nullptr;
    }
    UDF_LOG_INFO("alloc tensor msg success. alloc_size=%ld.", alloc_size);
    return mbuf_flow_msg;
}

std::shared_ptr<MbufFlowMsg> MbufFlowMsg::AllocRawDataMsg(int64_t size, uint32_t device_id,
    const MbufHead &mbuf_head, uint32_t align) {
    if (size < 0) {
        UDF_LOG_ERROR("alloc raw data msg param invalid. size=%ld.", size);
        return nullptr;
    }
    auto mbuf_flow_msg = CreateFlowMsg(static_cast<uint64_t>(size), device_id, mbuf_head, align);
    if (mbuf_flow_msg == nullptr) {
        UDF_LOG_ERROR("create flow msg failed. size=%ld.", size);
        return nullptr;
    }
    MbufHeadMsg &head_msg_ref = *(mbuf_flow_msg->head_msg_);
    head_msg_ref.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_RAW_MSG);
    head_msg_ref.data_flag &= ~(1U);
    UDF_LOG_INFO("alloc raw data msg success. size=%ld.", size);
    return mbuf_flow_msg;
}

int32_t MbufFlowMsg::InitMbufTensor() {
    std::shared_ptr<MbufTensor> mbuf_tensor;
    mbuf_tensor.reset(new(std::nothrow) MbufTensor(mbuf_, mbuf_info_.mbuf_addr, mbuf_info_.mbuf_len));
    if (mbuf_tensor == nullptr) {
        UDF_LOG_ERROR("alloc MbufTensor failed.");
        return FLOW_FUNC_FAILED;
    }

    mbuf_tensor_list_.emplace_back(mbuf_tensor);
    tensor_list_.emplace_back(mbuf_tensor.get());
    return mbuf_tensor->ParseRuntimeTensorDesc();
}

int32_t MbufFlowMsg::GetRawData(void *&dataPtr, uint64_t &dataSize) const {
    if (head_msg_ == nullptr) {
        UDF_LOG_ERROR("Head msg is nullptr.");
        return FLOW_FUNC_FAILED;
    }
    dataPtr = mbuf_info_.mbuf_addr;
    dataSize = mbuf_info_.mbuf_len;
    return FLOW_FUNC_SUCCESS;
}

int32_t MbufFlowMsg::InitMbufTensorList() {
    if (mbuf_info_.mbuf_addr == nullptr) {
        UDF_LOG_ERROR("MBuff addr is nullptr.");
        return FLOW_FUNC_FAILED;
    }
    uint64_t total_len = 0UL;
    uint8_t *start_ptr = static_cast<uint8_t *>(mbuf_info_.mbuf_addr);
    while (total_len < mbuf_info_.mbuf_len) {
        // check overflow
        if (CheckAddOverflowUint64(total_len, sizeof(RuntimeTensorDesc))) {
            UDF_LOG_ERROR("Total length=%lu add RuntimeTensor=%zu is overflow.",
                          total_len, sizeof(RuntimeTensorDesc));
            return FLOW_FUNC_FAILED;
        }
        // make sure ptr is not out of range
        if ((total_len + sizeof(RuntimeTensorDesc) > mbuf_info_.mbuf_len)) {
            UDF_LOG_ERROR("Tensor list data is invalid. total len=%lu add RuntimeDesc=%zu"
                          "is bigger than mbuf length=%lu.", total_len, sizeof(RuntimeTensorDesc), mbuf_info_.mbuf_len);
            return FLOW_FUNC_FAILED;
        }
        auto *rtd = reinterpret_cast<RuntimeTensorDesc *>(start_ptr + total_len);
        if (CheckAddOverflowUint64(total_len + sizeof(RuntimeTensorDesc), rtd->data_size)) {
            UDF_LOG_ERROR("Total length=%lu add data_size=%lu and RuntimeTensor=%zu is overflow.",
                          total_len, rtd->data_size, sizeof(RuntimeTensorDesc));
            return FLOW_FUNC_FAILED;
        }
        if ((total_len + sizeof(RuntimeTensorDesc) + rtd->data_size > mbuf_info_.mbuf_len)) {
            UDF_LOG_ERROR("Tensor list data is invalid. total len=%lu add RuntimeDesc=%zu data size=%lu "
                          "is bigger than mbuf length=%lu.", total_len, sizeof(RuntimeTensorDesc),
                          rtd->data_size, mbuf_info_.mbuf_len);
            return FLOW_FUNC_FAILED;
        }
        total_len += sizeof(RuntimeTensorDesc) + rtd->data_size;
        std::shared_ptr<MbufTensor> mbuf_tensor;
        mbuf_tensor.reset(new(std::nothrow) MbufTensor(mbuf_, static_cast<void *>(rtd),
            sizeof(RuntimeTensorDesc) + rtd->data_size));
        if (mbuf_tensor == nullptr) {
            UDF_LOG_ERROR("alloc MbufTensor failed.");
            return FLOW_FUNC_FAILED;
        }
 
        mbuf_tensor_list_.emplace_back(mbuf_tensor);
        tensor_list_.emplace_back(mbuf_tensor.get());
        int32_t ret = mbuf_tensor->ParseRuntimeTensorDesc();
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Parse runtime tensor desc failed.");
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t MbufFlowMsg::Init() {
    int32_t parse_ret = ParseMbuf();
    if (parse_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("parse mbuf failed, parse_ret=%d.", parse_ret);
        return parse_ret;
    }

    if ((head_msg_->msg_type == static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA)) ||
        (head_msg_->msg_type == static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_LIST))) {
        // 0 bit is null data flag, 1 is null data, 0 is not null data
        if ((head_msg_->data_flag & 1U) == 1U) {
            UDF_LOG_DEBUG("Current data is null data.");
            return FLOW_FUNC_SUCCESS;
        }
        if (head_msg_->ret_code != 0) {
            UDF_LOG_WARN("mbuf head msg ret_code=%d, trans_id=%lu.", head_msg_->ret_code, head_msg_->trans_id);
        }
        int32_t init_ret = FLOW_FUNC_SUCCESS;
        if (head_msg_->msg_type == static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA)) {
            init_ret = InitMbufTensor();
        } else {
            init_ret = InitMbufTensorList();
        }
        if (init_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("init mbuf tensor failed, trans_id=%lu, ret_code=%d.", head_msg_->trans_id,
                head_msg_->ret_code);
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t MbufFlowMsg::ParseMbuf() {
    auto drv_ret = halMbufGetPrivInfo(mbuf_.get(), &mbuf_info_.head_buf, &mbuf_info_.head_buf_len);
    if (drv_ret != DRV_ERROR_NONE || mbuf_info_.head_buf == nullptr) {
        UDF_LOG_ERROR("Failed to get mbuf priv info, ret[%d].", drv_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    if (mbuf_info_.head_buf_len < sizeof(MbufHeadMsg)) {
        UDF_LOG_ERROR("mbuf priv info len=%u can't be less than to sizeof(MbufHeadMsg)=%zu.",
            mbuf_info_.head_buf_len, sizeof(MbufHeadMsg));
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    head_msg_ = reinterpret_cast<MbufHeadMsg *>(static_cast<uint8_t *>(mbuf_info_.head_buf) +
                                               (mbuf_info_.head_buf_len - sizeof(MbufHeadMsg)));

    drv_ret = halMbufGetBuffAddr(mbuf_.get(), &mbuf_info_.mbuf_addr);
    if (drv_ret != DRV_ERROR_NONE || mbuf_info_.mbuf_addr == nullptr) {
        UDF_LOG_ERROR("get mbuf addr failed, ret=%d.", drv_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }

    drv_ret = halMbufGetDataLen(mbuf_.get(), &mbuf_info_.mbuf_len);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("get mbuf data len failed, ret=%d.", drv_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

std::string MbufFlowMsg::DebugString() const {
    std::ostringstream debug_info;
    debug_info << "MbufFlowMsg head len:" << mbuf_info_.head_buf_len << ", mbuf len:" << mbuf_info_.mbuf_len
              << ", headMsg:";
    if (head_msg_ == nullptr) {
        debug_info << "NULL";
    } else {
        debug_info << '{' << head_msg_->DebugString() << '}';
    }

    if (!mbuf_tensor_list_.empty()) {
        debug_info << ", mbuf_tensor:";
        for (const auto &mbuf_tensor : mbuf_tensor_list_) {
            debug_info << "{" << mbuf_tensor->DebugString() << '}';
        }
    }
    return debug_info.str();
}

void MbufFlowMsg::SetRetCode(int32_t ret_code) {
    if ((ret_code < FLOW_FUNC_ERR_USER_DEFINE_START) || (ret_code > FLOW_FUNC_ERR_USER_DEFINE_END)) {
        UDF_LOG_WARN("The ret code should in [%d, %d], but got ret code[%d].",
            FLOW_FUNC_ERR_USER_DEFINE_START, FLOW_FUNC_ERR_USER_DEFINE_END, ret_code);
    }
    SetRetCodeInner(ret_code);
}

void MbufFlowMsg::SetRetCodeInner(int32_t ret_code) const
{
    if (head_msg_ != nullptr) {
        head_msg_->ret_code = ret_code;
    }
}
}