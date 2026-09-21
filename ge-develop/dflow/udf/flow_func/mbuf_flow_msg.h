/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_MBUF_FLOW_MSG_H
#define FLOW_FUNC_MBUF_FLOW_MSG_H

#include <memory>
#include <string>
#include <sstream>
#include "common/common_define.h"
#include "flow_func/flow_msg.h"
#include "ascend_hal.h"

namespace FlowFunc {
#pragma pack(push, 1)

struct MbufHeadMsg {
    uint64_t trans_id;
    uint16_t version;
    uint16_t msg_type;
    int32_t ret_code;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t flags;
    uint8_t data_flag;
    uint8_t rsv0[3];
    int32_t worked_id;       // EES使用，占位
    uint32_t step_id;
    uint8_t rsv[8];
    uint32_t data_label;     // use for data align
    uint32_t route_label;    // use for data route

    std::string DebugString() const;
};

constexpr uint32_t kMaxDimSize = 32U;
constexpr uint8_t kCustomTransIdFlagBit = 1U << 1;

struct RuntimeTensorDesc {
    uint64_t dataAddr;
    int64_t dataOffsetSize;
    int64_t dtype;
    int64_t shape[kMaxDimSize + 1U]; // shape:Dim_Num|DIM0|DIM1|...|DIM31
    int64_t originalShape[kMaxDimSize + 1U]; // original_shape:Dim_Num|DIM0|DIM1|...|DIM31
    int64_t format;
    int64_t subFormat;
    uint64_t data_size;
    uint8_t reserved[448]; // padding to 1024 bytes

    std::string DebugString() const;
};

#pragma pack(pop)

class FLOW_FUNC_VISIBILITY MbufTensor : public Tensor {
public:
    /**
     * @brief construct MbufTensor from mbuf data.
     * attention:Mbuf must contain tensor desc at first 1024 bytes.
     * @param mbuf
     */
    MbufTensor(std::shared_ptr<Mbuf> mbuf, void *mbuf_data, uint64_t mbuf_data_size);
    MbufTensor(std::shared_ptr<Mbuf> mbuf, void *mbuf_data, uint64_t mbuf_data_size, uint64_t data_size);

    ~MbufTensor() noexcept override;

    int32_t ParseRuntimeTensorDesc();

    int32_t InitRuntimeTensorDesc(const std::vector<int64_t> &shape, TensorDataType data_type);

    const std::vector<int64_t> &GetShape() const override {
        return shape_;
    }

    TensorDataType GetDataType() const override {
        if (tensor_desc_ == nullptr) {
            return TensorDataType::DT_UNDEFINED;
        }
        return static_cast<TensorDataType>(tensor_desc_->dtype);
    }

    void *GetData() const override {
        return data_;
    }

    uint64_t GetDataSize() const override {
        return data_size_;
    }

    uint64_t GetDataBufferSize() const override;

    int64_t GetElementCnt() const override;

    int32_t Reshape(const std::vector<int64_t> &shape) override;

    std::string DebugString() const;

    static int32_t InitRuntimeTensorDesc(RuntimeTensorDesc *tensor_desc, const std::vector<int64_t> &shape,
        TensorDataType data_type, uint64_t data_size);

    std::shared_ptr<Mbuf> SharedMbuf() {
        return mbuf_;
    }

private:
    static void PrintBytesHex(const uint8_t *bytes_data, uint64_t size, std::ostringstream &ss);

    std::shared_ptr<Mbuf> mbuf_ = nullptr;
    void *mbuf_data_ = nullptr;
    uint64_t mbuf_data_size_ = 0UL;
    std::vector<int64_t> shape_;

    // point to mem in mbuf_data_
    RuntimeTensorDesc *tensor_desc_ = nullptr;

    // point to mem in mbuf_data_
    void *data_ = nullptr;
    uint64_t data_size_ = 0UL;
};

struct MbufInfo {
    void *head_buf = nullptr;
    uint32_t head_buf_len = 0;
    uint32_t reserve = 0;   // just for bit align
    void *mbuf_addr = nullptr;
    uint64_t mbuf_len = 0;
};

constexpr uint32_t kMaxMbufHeadLen = 256U;
struct MbufHead {
    uint8_t head_buf[kMaxMbufHeadLen] = {};
    uint32_t head_buf_len = 0U;
};

class FLOW_FUNC_VISIBILITY MbufFlowMsg : public FlowMsg {
public:
    /**
     * @brief construct MbufTensor from mbuf.
     * attention:Mbuf must contain tensor desc at first 1024 bytes.
     * @param mbuf
     */
    explicit MbufFlowMsg(std::shared_ptr<Mbuf> mbuf);

    ~MbufFlowMsg() override;

    static std::shared_ptr<MbufFlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type,
        uint32_t device_id, const MbufHead &mbuf_head, uint32_t align = Common::kDefaultMbufAlignSize);

    static std::shared_ptr<MbufFlowMsg> AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes,
        const std::vector<TensorDataType> &data_types, uint32_t device_id, const MbufHead &mbuf_head,
        uint32_t align);

    static std::shared_ptr<MbufFlowMsg> AllocEmptyTensorMsg(uint32_t device_id, const MbufHead &mbuf_head);

    static std::shared_ptr<MbufFlowMsg> AllocRawDataMsg(int64_t size, uint32_t device_id,
        const MbufHead &mbuf_head = {}, uint32_t align = Common::kDefaultMbufAlignSize);

    static std::shared_ptr<MbufFlowMsg> ToTensorMsg(const std::vector<int64_t> &shape,
        TensorDataType data_type, std::shared_ptr<Mbuf> mbuf, const MbufHead &mbuf_head);

    int32_t Init();

    MsgType GetMsgType() const override {
        if (head_msg_ != nullptr) {
            return static_cast<MsgType>(head_msg_->msg_type);
        }
        return MsgType::MSG_TYPE_TENSOR_DATA;
    }

    void SetMsgType(MsgType msgType) override {
        if (head_msg_ != nullptr) {
            head_msg_->msg_type = static_cast<uint16_t>(msgType);
        }
    }

    Tensor *GetTensor() const override {
        if (tensor_list_.empty()) {
            return nullptr;
        }
        return *tensor_list_.cbegin();
    }

    std::shared_ptr<Tensor> GetSharedTensor() const {
        if (mbuf_tensor_list_.empty()) {
            return std::shared_ptr<Tensor>(nullptr);
        }
        return *mbuf_tensor_list_.cbegin();
    }

    const std::vector<Tensor *> GetTensorList() const override {
        return tensor_list_;
    }

    int32_t GetRetCode() const override {
        if (head_msg_ == nullptr) {
            return 0;
        }
        return head_msg_->ret_code;
    }

    void SetRetCode(int32_t ret_code) override;

    int32_t GetRawData(void *&dataPtr, uint64_t &dataSize) const override;

    /**
     * @brief SetRetCode is external interface, need check ret_code,
     * this method will not check ret_code.
     * @param ret_code ret code.
     */
    void SetRetCodeInner(int32_t ret_code) const;

    void SetStartTime(uint64_t start_time) override {
        if (head_msg_ != nullptr) {
            head_msg_->start_time = start_time;
        }
    }

    uint64_t GetStartTime() const override {
        if (head_msg_ != nullptr) {
            return head_msg_->start_time;
        }
        return 0UL;
    }

    void SetEndTime(uint64_t end_time) override {
        if (head_msg_ != nullptr) {
            head_msg_->end_time = end_time;
        }
    }

    uint64_t GetEndTime() const override {
        if (head_msg_ != nullptr) {
            return head_msg_->end_time;
        }
        return 0UL;
    }

    void SetFlowFlags(uint32_t flags) override {
        if (head_msg_ != nullptr) {
            head_msg_->flags = flags;
        }
    }

    uint32_t GetFlowFlags() const override {
        if (head_msg_ != nullptr) {
            return head_msg_->flags;
        }
        // default not flag.
        return 0U;
    }

    void SetRouteLabel(uint32_t route_label) override {
        if (head_msg_ != nullptr) {
            head_msg_->route_label = route_label;
        }
    }

    uint64_t GetTransactionId() const override {
        if (head_msg_ != nullptr) {
            return head_msg_->trans_id;
        }
        return UINT64_MAX;
    }

    void SetTransactionId(uint64_t transactionId) override {
        if (head_msg_ != nullptr) {
            if (transactionId == 0UL) {
                // clear custom trans id flag
                head_msg_->data_flag &= (~kCustomTransIdFlagBit);
            } else {
                head_msg_->data_flag |= kCustomTransIdFlagBit;
            }
            head_msg_->trans_id = transactionId;
        }
    }

    void SetTransactionIdInner(uint64_t transaction_id) const {
        if (head_msg_ != nullptr) {
            head_msg_->trans_id = transaction_id;
        }
    }

    void SetDataLabel(uint32_t data_label) const {
        if (head_msg_ != nullptr) {
            head_msg_->data_label = data_label;
        }
    }

    void SetStepId(uint32_t step_id) const {
        if (head_msg_ != nullptr) {
            head_msg_->step_id = step_id;
        }
    }

    uint32_t GetStepId() const {
        if (head_msg_ != nullptr) {
            return head_msg_->step_id;
        }
        // default step 0.
        return 0U;
    }

    Mbuf *GetMbuf() {
        return mbuf_.get();
    }

    const MbufInfo &GetMbufInfo() {
        return mbuf_info_;
    }

    std::string DebugString() const;
private:
    static Mbuf *AllocMbuf(uint64_t alloc_size, uint32_t device_id, const MbufHead &mbuf_head, uint32_t align);

    static std::shared_ptr<MbufFlowMsg> CreateFlowMsg(uint64_t size, uint32_t device_id, const MbufHead &mbuf_head = {},
        uint32_t align = Common::kDefaultMbufAlignSize);

    static std::shared_ptr<MbufFlowMsg> CreateFlowMsg(std::shared_ptr<Mbuf> mbuf, const MbufHead &mbuf_head);

    int32_t InitMbufTensor(const std::vector<int64_t> &shape, TensorDataType data_type);

    int32_t InitMbufTensorList(const std::vector<std::vector<int64_t>> &shapes,
                               const std::vector<TensorDataType> &data_types,
                               const std::vector<uint64_t> &data_orig_size,
                               const std::vector<uint64_t> &data_align_size);

    int32_t InitMbufTensor(const std::vector<int64_t> &shape, TensorDataType data_type,
                           void *mbuf_addr, uint64_t mbuf_len, uint64_t data_size = 0UL);
    int32_t InitMbufTensor();
    int32_t InitMbufTensorList();

    int32_t ParseMbuf();

    static int32_t SetMbufHead(Mbuf *mbuf, const MbufHead &mbuf_head);

    std::shared_ptr<Mbuf> mbuf_ = nullptr;
    MbufInfo mbuf_info_;
    std::vector<std::shared_ptr<MbufTensor>> mbuf_tensor_list_;
    std::vector<Tensor *> tensor_list_;
    /**
     * @brief point to Mbuf head in mbuf.
     */
    MbufHeadMsg *head_msg_ = nullptr;
};
}

#endif // FLOW_FUNC_MBUF_FLOW_MSG_H
