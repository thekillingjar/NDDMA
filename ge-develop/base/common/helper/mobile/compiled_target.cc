/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compiled_target.h"

#include <iostream>
#include <array>
#include <climits>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/buffer.h"

namespace {

constexpr uint32_t NPU_COMPILED_MAGIC_NUM = 0xABCDABCDU;
constexpr uint32_t VERSION = 0;

struct Tlv {
    uint32_t type;
    uint32_t len;
};

constexpr uint32_t RESV_MAX = 4;

struct Head {
    uint32_t magic_num;
    uint32_t ver;
    uint32_t total_len;
    uint32_t optional_num;
    std::array<uint32_t, RESV_MAX> resv;
};

std::size_t GetModelTaskSize(std::shared_ptr<ge::mobile::proto::ModelTaskDef> model_task)
{
    std::size_t size = 0;
    if (model_task == nullptr) {
        GELOGI("[Mobile] model_task == nullptr");
        return 0;
    }
    size += sizeof(Tlv);
#if GOOGLE_PROTOBUF_VERSION < 3013000
    size += static_cast<size_t>(static_cast<uint32_t>(model_task->ByteSize()));
#else
    GE_ASSERT_TRUE(model_task->ByteSizeLong() <= UINT32_MAX, "[Mobile] overflow, failed.");
    size += static_cast<size_t>(static_cast<uint32_t>(model_task->ByteSizeLong()));
#endif
    return size;
}

void FillHead(Head* head, uint32_t total_len, uint32_t opt_num)
{
    (void)memset_s(head, sizeof(Head), 0, sizeof(Head));
    head->ver = VERSION;
    head->total_len = total_len;
    head->magic_num = NPU_COMPILED_MAGIC_NUM;
    head->optional_num = opt_num;
    GELOGI("[Mobile] total len: %d", head->total_len);
}

} // namespace

namespace ge {

std::size_t CompiledTarget::GetSize() const
{
    constexpr size_t max_section_size = 3;
    std::size_t size = sizeof(Head);
    GE_ASSERT_TRUE(sections_.size() <= max_section_size, "[Mobile] section num = %u should be <= 3", sections_.size());
    for (const SectionHolder& item : sections_) {
        size += sizeof(Tlv);
        size += item.size;
    }
    size += GetModelTaskSize(model_task_);
    return size == sizeof(Head) ? 0UL : size;
}

uint32_t CompiledTarget::SerializeModelTask(uint8_t* addr, uint32_t space_size, uint32_t section_type)
{
    uint32_t size;
    std::shared_ptr<mobile::proto::ModelTaskDef> model_task = model_task_;
    GE_ASSERT_NOTNULL(model_task, "[Mobile] model_task is empty.");

#if GOOGLE_PROTOBUF_VERSION < 3013000
    GE_ASSERT_TRUE(model_task->ByteSize() <= UINT32_MAX, "[Mobile] overflow, failed.");
    size = static_cast<uint32_t>(model_task->ByteSize());
#else
    GE_ASSERT_TRUE(model_task->ByteSizeLong() <= UINT32_MAX, "[Mobile] overflow, failed.");
    size = static_cast<uint32_t>(model_task->ByteSizeLong());
#endif
    GE_ASSERT_TRUE(sizeof(Tlv) <= UINT32_MAX, "[Mobile] overflow, failed.");
    const uint32_t len = size + static_cast<uint32_t>(sizeof(Tlv));
    if (len > space_size) {
        GELOGE(ge::FAILED, "[Mobile] spaceSize[need :%d, real = %d] is not enough", len, space_size);
        return 0;
    }

    GE_ASSERT_NOTNULL(addr, "[Mobile] addr is empty.");
    Tlv* tlvHead = reinterpret_cast<Tlv*>(addr);
    tlvHead->type = section_type;
    tlvHead->len = len;
    (void)model_task->SerializePartialToArray(&addr[sizeof(Tlv)], static_cast<int32_t>(size));
    return len;
}

uint32_t CompiledTarget::Serialize(ge::BaseBuffer& buffer)
{
    GELOGI("[Mobile] MobileModel CompiledTarget Serialize.");
    GE_ASSERT_TRUE(GetSize() <= UINT32_MAX, "[Mobile] overflow, failed.");
    const uint32_t size = static_cast<uint32_t>(GetSize());
    uint8_t *base_addr = buffer.GetData();
    GE_ASSERT_TRUE((size <= buffer.GetSize()) && (static_cast<std::size_t>(size) >= sizeof(Head)),
        "[Mobile] buffer is no enough");
    GE_ASSERT_NOTNULL(base_addr, "[Mobile] GetData is null.");

    Head* head = reinterpret_cast<Head*>(base_addr);
    GE_ASSERT_TRUE(sections_.size() <= UINT32_MAX, "[Mobile] overflow, failed.");
    FillHead(head, size, static_cast<uint32_t>(sections_.size()));
    uint32_t offset = static_cast<uint32_t>(sizeof(Head));

    // add modelTaskDef
    GE_ASSERT_TRUE(offset < buffer.GetSize(), "[Mobile] overflow, failed.");
    offset += SerializeModelTask(&base_addr[offset], size - offset, static_cast<uint32_t>(SectionType::SECTION_TYPE_MODELDEF));

    Tlv* tlvHead = nullptr;
    for (const SectionHolder& section : sections_) {
        GE_ASSERT_TRUE(sizeof(Tlv) <= UINT32_MAX, "[Mobile] overflow, failed.");
        if ((offset + sizeof(Tlv) + section.size) <= size) {
            GE_ASSERT_TRUE(offset < buffer.GetSize(), "[Mobile] overflow, failed.");
            tlvHead = reinterpret_cast<Tlv *>(&base_addr[offset]);
            tlvHead->type = section.type;
            tlvHead->len = section.size + static_cast<uint32_t>(sizeof(Tlv));
            offset += static_cast<uint32_t>(sizeof(Tlv));
            GE_ASSERT_TRUE(offset < buffer.GetSize(), "[Mobile] overflow, failed.");
            const auto ret = memcpy_s(&base_addr[offset], static_cast<size_t>(size - offset),
                section.data.get(), static_cast<size_t>(section.size));
            if (ret != EOK) {
                GELOGE(ge::FAILED, "[Mobile] memcpy_s failed.");
                return ge::FAILED;
            }
            offset += section.size;
        } else {
            GELOGE(ge::FAILED, "[Mobile] buffer is not enough, calc size is wrong.");
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

void CompiledTarget::AddSection(SectionHolder& section)
{
    sections_.push_back(std::move(section));
}

void CompiledTarget::AddModelTaskDef(std::shared_ptr<mobile::proto::ModelTaskDef> task_def)
{
    model_task_ = task_def;
}

void CompiledTarget::UpdateGraphOpTaskSize(
    ge::mobile::proto::ModelDef& mobile_model_def) const
{
    bool is_update = false;
    auto update_func = [&is_update, this](ge::mobile::proto::OpDef* op) {
        const std::string GRAPH_OP_TYPE = "GraphOp";
        if (op->type() != GRAPH_OP_TYPE) {
            return;
        }
        auto* attr = op->mutable_attr();
        const std::string graphop_task_size_str = "graphop_task_size";
        (*attr)[graphop_task_size_str].set_i(static_cast<int64_t>(GetSize()));
        GELOGI("[Mobile] base_addr:  0x%llx", model_task_->base_addr());
        GELOGI("[Mobile] weight_addr: 0x%llx", model_task_->weight_addr());
        is_update = true;
    };

    int graph_size = mobile_model_def.graph_size();
    for (int i = 0; i < graph_size; i++) {
        auto* g = mobile_model_def.mutable_graph(i);
        GELOGI("[Mobile] subgraph name: %s", g->name().c_str());
        if (g->name() == "ge_default") {
            for (int j = 0; (j < g->op_size()) && (!is_update); j++) {
                auto* op = g->mutable_op(j);
                update_func(op);
            }
        } else {
            GELOGD("[Mobile] skip.");
        }
    }
}

ge::Status CompiledTarget::UpdateModelModelTaskDef(ge::mobile::proto::ModelDef& mobile_model_def) const
{
    // save subgraph data and netoutput
    for (const auto& g: mobile_model_def.graph()) {
        GELOGI("[Mobile] subgraph name: %s", g.name().c_str());
        if (g.name() != "ge_default/SubGraph_0") {
            continue;
        }
        GELOGI("[Mobile] use subgraph name: %s, update model task op.", g.name().c_str());
        for (const auto& op: g.op()) {
            if ((op.type() != "Data") && (op.type() != "NetOutput")) {
                continue;
            }
            GELOGI("[Mobile] op name: %s type: %s", op.name().c_str(), op.type().c_str());
#if GOOGLE_PROTOBUF_VERSION < 3013000
            GE_ASSERT_TRUE(op.ByteSize() <= UINT32_MAX, "[Mobile] overflow, failed.");
            const uint32_t buffer_size = static_cast<uint32_t>(op.ByteSize());
#else
            GE_ASSERT_TRUE(op.ByteSizeLong() <= UINT32_MAX, "[Mobile] overflow, failed.");
            uint32_t buffer_size = static_cast<uint32_t>(op.ByteSizeLong());
#endif
            for (const auto& output_offset: op.output_i()) {
                GELOGI("[Mobile]   output_offset: %d", output_offset);
            }

            ge::Buffer buffer(static_cast<size_t>(buffer_size));
            GE_ASSERT_TRUE(buffer.GetSize() <= INT32_MAX, "[Mobile] overflow, failed.");
            google::protobuf::io::ArrayOutputStream array_stream(
                buffer.GetData(), static_cast<int32_t>(buffer.GetSize()));
            google::protobuf::io::CodedOutputStream output_stream(&array_stream);
            output_stream.SetSerializationDeterministic(true);
            (void)op.SerializeToCodedStream(&output_stream);
            model_task_->add_op(reinterpret_cast<const void*>(buffer.GetData()), buffer.GetSize());
            GELOGI("[Mobile] op buffer size: %d", buffer.GetSize());
        }
        GELOGI("[Mobile] model task op size: %d", model_task_->op_size());
        break;
    }
    // update graph op task size
    UpdateGraphOpTaskSize(mobile_model_def);
    return ge::SUCCESS;
}

} // namespace ge
