/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_buffer_helper.h"

#include <iostream>
#include <array>
#include <vector>
#include <climits>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "graph/model.h"
#include "graph/buffer.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "securec.h"
#include "common/scope_guard.h"
#include "large_memory_ops.h"
#include "model.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"

namespace {

constexpr uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49U;
constexpr uint32_t MODEL_FILE_HEAD_LEN = 256;
constexpr uint32_t MODEL_VERSION = 0x10000000U;
constexpr uint32_t MODEL_FILE_CHECKSUM_LENGTH = 64;

enum class ModelEncryptType : uint8_t {
    UNENCRYPTED,
    ENCRYPTED
};

enum class ModelCheckType : uint8_t {
    CHECK,
    UNCHECK
};

constexpr uint32_t MODEL_NAME_LENGTH = 32;
constexpr uint32_t USER_DEFINE_INFO_LENGTH = 32;
constexpr uint32_t PLATFORM_VERSION_LEN = 20;
constexpr uint32_t CUSTOM_LIB_INFO_LENGTH = 16;
constexpr uint32_t MODEL_FILE_RESERVED_LENGTH = 52;

struct ModelFileHeader {
    uint32_t magic = MODEL_FILE_MAGIC_NUM;
    uint32_t head_size = MODEL_FILE_HEAD_LEN;
    uint32_t version = MODEL_VERSION;
    std::array<uint8_t, MODEL_FILE_CHECKSUM_LENGTH> checksum{};
    uint32_t length = 0;
    uint8_t is_encrypt = static_cast<uint8_t>(ModelEncryptType::UNENCRYPTED);
    uint8_t is_checksum = static_cast<uint8_t>(ModelCheckType::CHECK);
    uint8_t model_type = 0;
    uint8_t gen_mode = 0;
    std::array<uint8_t, MODEL_NAME_LENGTH> name{};
    uint32_t ops = 0;
    std::array<uint8_t, USER_DEFINE_INFO_LENGTH> user_define_info{};
    uint32_t om_ir_version = 0;
    std::array<uint8_t, PLATFORM_VERSION_LEN> platform_version{};
    uint8_t platform_type = {0};
    uint8_t heter_tuning_flag = {0};
    uint32_t custom_data_len = 0;
    uint32_t multi_shape_gears = 0;
    std::array<uint8_t, CUSTOM_LIB_INFO_LENGTH> custom_lib_info{};
    std::array<uint8_t, MODEL_FILE_RESERVED_LENGTH> reserved{};
};

enum class ModelType : uint32_t {
    IR_GRAPH_MODEL = 0,
    OM_STANDARD_MODEL,
    IGS_MODEL,
    STANDARD_IR_GRAPH_MODEL,
    HCS_PARTITION_MODEL,
    ISPNN_MODEL,
    SPECIAL_3RD_MODEL,
    IR_API_GRAPH_MODEL,
    UTINY_MODEL,
    MULTI_SHAPE_MODEL,
    ENUM_SHAPE_MODEL,
    SINGLE_IP_MODEL,
    MULTI_SHAPE_SINGLE_IP_MODEL,
};

enum class ModelPartitionType : uint32_t {
    MODEL_DEF = 0,
    WEIGHTS_DATA,
    TASK_INFO,
    TASK_GRAPH,
    SIGNATURE_INFO,
    MODEL_CONFIG,
    AIPP_CUSTOM_INFO,
    ENCRYPT_INFO,
    WEIGHT_INFO
};

struct ModelPartitionMemInfo {
    ModelPartitionType type;
    uint32_t memOffset;
    uint32_t memSize;
};

struct ModelPartitionTable {
    uint32_t num;
    ModelPartitionMemInfo partition[0];
};

uint64_t SizeOfModelPartitionTable(const ModelPartitionTable table)
{
    return sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * (table.num);
}

struct ModelPartition {
    ModelPartitionType type;
    uint8_t* data {nullptr};
    uint32_t size {0};
};

struct OmFileContext {
    std::vector<ModelPartition> partition_datas;
    std::vector<char> partition_table;
    uint64_t model_data_len = 0;
};

struct ModelBasicInfo {
    uint32_t model_type = 0;
    uint32_t version = 0;
    string name;
    string framework_version = "";
    uint64_t graph_hash = 0;
    int64_t timestamp = 0;
};

ge::Status SetPlatformVersion(ModelFileHeader& model_header, const std::string& platform_version)
{
    GE_ASSERT_TRUE(platform_version.size() <= UINT32_MAX, "[Mobile] overflow, failed.");
    uint32_t version_size = static_cast<uint32_t>(platform_version.size());
    version_size = version_size > static_cast<uint32_t>(PLATFORM_VERSION_LEN) - 1U ?
        static_cast<uint32_t>(PLATFORM_VERSION_LEN) - 1U : version_size;
    const errno_t ret = memcpy_s(model_header.platform_version.data(), static_cast<size_t>(PLATFORM_VERSION_LEN), platform_version.c_str(), static_cast<size_t>(version_size));
    if (ret != EOK) {
        GELOGE(ge::FAILED, "[Mobile] memcpy_s failed!, platform_version save:%s", model_header.platform_version);
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status SetModelName(ModelFileHeader& model_header, const std::string& model_name)
{
    GE_ASSERT_TRUE(model_name.size() <= UINT32_MAX, "[Mobile] overflow, failed.");
    uint32_t name_size = static_cast<uint32_t>(model_name.size());
    name_size = name_size > static_cast<uint32_t>(MODEL_NAME_LENGTH) - 1U ?
        static_cast<uint32_t>(MODEL_NAME_LENGTH) - 1U : name_size;
    const errno_t ret = memcpy_s(model_header.name.data(), static_cast<size_t>(MODEL_NAME_LENGTH), model_name.c_str(), static_cast<size_t>(name_size));
    if (ret != EOK) {
        GELOGE(ge::FAILED, "[Mobile] copy model name %s failed!", model_header.name);
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

ge::Status SetModelLen(ModelFileHeader& model_header, const ModelPartitionTable* partition_table, uint64_t data_len)
{
    GE_ASSERT_TRUE(data_len > static_cast<uint64_t>(0),
        "[Mobile] get model_data_len is <= 0.");
    GE_ASSERT_NOTNULL(partition_table, "[Mobile] partition_table is nullptr.");
    const uint64_t table_size = SizeOfModelPartitionTable(*partition_table);
    const uint64_t total_size = table_size + data_len;
    if (total_size > UINT32_MAX) {
        GELOGE(ge::FAILED, "[Mobile] table_size(%lu) + data_len(%lu) > UINT32_MAX:(%u).", table_size, data_len, UINT32_MAX);
        return ge::FAILED;
    }
    model_header.length = static_cast<uint32_t>(total_size);
    return ge::SUCCESS;
}

class OmFileSaveHelper {
public:
    const ModelFileHeader& GetModelFileHeader()
    {
        return model_header_;
    }

    ge::Status UpdateModelFileHeader(
        const ModelBasicInfo& model_info)
    {
        model_header_.model_type = static_cast<uint8_t>(model_info.model_type);
        model_header_.om_ir_version = model_info.version;
        auto ret = SetPlatformVersion(model_header_, model_info.framework_version);
        GE_ASSERT_TRUE(ret == ge::SUCCESS,
            "[Mobile] set platform version failed.");
        ret = SetModelName(model_header_, model_info.name);
        GE_ASSERT_TRUE(ret == ge::SUCCESS,
            "[Mobile] set model name failed.");
        ret = SetModelLen(model_header_, GetPartitionTable(), GetModelDataSize());
        GE_ASSERT_TRUE(ret == ge::SUCCESS,
            "[Mobile] set model len failed.");
        return ge::SUCCESS;
    }

    uint64_t GetModelDataSize()
    {
        return context_.model_data_len;
    }

    void AddPartition(const ModelPartition& partition)
    {
        context_.partition_datas.push_back(partition);
        if (context_.model_data_len > (UINT64_MAX - partition.size)) {
            GELOGW("[Mobile] overflow failed.");
            return;
        }
        context_.model_data_len += partition.size;
    }

    ModelPartitionTable* GetPartitionTable()
    {
        const uint64_t partition_size = context_.partition_datas.size();
        context_.partition_table.clear();
        context_.partition_table.resize(
            sizeof(ModelPartitionTable) + sizeof(ModelPartitionMemInfo) * partition_size, static_cast<char>(0));
        ModelPartitionTable* partition_table = reinterpret_cast<ModelPartitionTable*>(context_.partition_table.data());
        if (partition_table == nullptr) {
            GELOGE(ge::FAILED, "[Mobile] partition_table is nullptr.");
            return nullptr;
        }
        GE_ASSERT_TRUE(partition_size <= UINT32_MAX, "[Mobile] overflow, failed.");
        partition_table->num = static_cast<uint32_t>(partition_size);

        uint64_t mem_offset = 0;
        for (uint64_t i = 0; i < partition_size; i++) {
            ModelPartition partition = context_.partition_datas[i];
            if (mem_offset > UINT32_MAX) {
                GELOGE(ge::FAILED, "[Mobile] mem_offset large than UINT32_MAX failed.");
                return nullptr;
            }
            partition_table->partition[i] = {partition.type, static_cast<uint32_t>(mem_offset), partition.size};
            mem_offset += partition.size;
        }
        return partition_table;
    }

    std::vector<ModelPartition>& GetModelPartitions()
    {
        return context_.partition_datas;
    }

private:
    ModelFileHeader model_header_;
    OmFileContext context_;
};

void GetModelBasicInfo(
    const ge::GeModelPtr& ge_model,
    ModelType model_type,
    ModelBasicInfo& model_info)
{
    model_info.model_type = static_cast<uint32_t>(model_type);

    constexpr uint32_t OM_PROTO_VERSION = 2;
    model_info.version = OM_PROTO_VERSION;

    model_info.name = ge_model->GetName();
    model_info.framework_version = ge_model->GetPlatformVersion();

    model_info.graph_hash = static_cast<uint64_t>(0);
    model_info.timestamp = 0;
}

void SavePartition(
    OmFileSaveHelper& om_file_save_helper,
    ge::Buffer& buffer,
    uint64_t size,
    ModelPartitionType type)
{
    ModelPartition partition;
    partition.data = buffer.GetData();
    partition.size = static_cast<uint32_t>(size);
    partition.type = type;
    om_file_save_helper.AddPartition(partition);
}

ge::Status SaveModelPartition(
    OmFileSaveHelper& om_file_save_helper,
    ge::Buffer& model_buffer,
    const ge::GeModelPtr& ge_model,
    const ge::mobile::proto::ModelDef& mobile_model_def,
    const ModelBasicInfo& model_info,
    uint32_t& align_offset)
{
    (void)model_info;
    (void)ge_model;

// serialize graph
#if GOOGLE_PROTOBUF_VERSION < 3013000
    GE_ASSERT_TRUE(mobile_model_def.ByteSize() <= UINT32_MAX, "[Mobile] overflow, failed.");
    const uint32_t model_buffer_size = static_cast<uint32_t>(mobile_model_def.ByteSize());
#else
    GE_ASSERT_TRUE(mobile_model_def.ByteSizeLong() <= UINT32_MAX, "[Mobile] overflow, failed.");
    const uint32_t model_buffer_size = static_cast<uint32_t>(mobile_model_def.ByteSizeLong());
#endif
    model_buffer = ge::Buffer(static_cast<size_t>(model_buffer_size));
    google::protobuf::io::ArrayOutputStream array_stream(
        model_buffer.GetData(), static_cast<int32_t>(model_buffer.GetSize()));
    google::protobuf::io::CodedOutputStream output_stream(&array_stream);
    output_stream.SetSerializationDeterministic(true);
    GE_ASSERT_TRUE(mobile_model_def.SerializeToCodedStream(&output_stream),
        "[Mobile] serial to coded stream failed.");
    
    GELOGI("[Mobile] save MODEL_DEF, model buffer size: %d", model_buffer.GetSize());
    SavePartition(om_file_save_helper, model_buffer, model_buffer.GetSize(), ModelPartitionType::MODEL_DEF);
    align_offset = static_cast<uint32_t>(0);
    return ge::SUCCESS;
}

ge::Status SaveCompiledPartion(
    OmFileSaveHelper& om_file_save_helper,
    ModelPartitionType type,
    const vector<ge::BaseBuffer>& compiled_buffers,
    uint32_t& align_offset)
{
    size_t buffer_size = 0;
    for (size_t i = 0; i < compiled_buffers.size(); i++) {
        buffer_size += compiled_buffers[i].GetSize();
    }
    GELOGI("[Mobile] save partition type: %d buffer size: %d", static_cast<uint32_t>(type), buffer_size);
    GE_ASSERT_TRUE((buffer_size <= UINT32_MAX),
        "[Mobile] buffer size large than UINT32_MAX failed.");
    ModelPartition partition;
    partition.data = nullptr;
    GE_ASSERT_TRUE(buffer_size <= UINT32_MAX, "[Mobile] overflow, failed.");
    partition.size = static_cast<uint32_t>(buffer_size) + align_offset;
    partition.type = type;
    om_file_save_helper.AddPartition(partition);
    align_offset = static_cast<uint32_t>(0);
    return ge::SUCCESS;
}

ge::Status SaveModelFileHeader(
    OmFileSaveHelper& om_file_save_helper,
    const ModelBasicInfo& model_info)
{
    return om_file_save_helper.UpdateModelFileHeader(model_info);
}

ge::Status CopyDataToBuffer(
    std::uint8_t* out_buffer_ptr,
    const size_t out_buffer_size,
    size_t& offset, const void* data, size_t size)
{
    GE_ASSERT_TRUE((out_buffer_size - offset >= size),
        "[Mobile] out_buffer_size - offset >= size, out_buffer_size: %d, offset: %d, size: %d.",
        out_buffer_size, offset, size);
    auto ret = ge::LargeMemoryOps::Copy(&out_buffer_ptr[offset], out_buffer_size - offset, data, size);
    GE_ASSERT_TRUE(ret, "[Mobile] copy failed.");
    offset += size;
    return ge::SUCCESS;
}

ge::Status SetModelHeader(
    std::uint8_t* base_ptr, size_t total_size, size_t& offset,
    const ModelFileHeader& model_header)
{
    GELOGI("[Mobile] SetModelHeader, base_ptr: %p total_size: %d offset: %d model_header size: %d",
        static_cast<void*>(base_ptr), total_size, offset, sizeof(model_header));
    const ge::Status ret = CopyDataToBuffer(base_ptr, total_size, offset, &model_header, sizeof(model_header));
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] copy failed.");
    return ge::SUCCESS;
}

ge::Status SetModelPartitionTable(
    const ModelPartitionTable* partition_table, std::uint8_t* base_ptr, size_t total_size, size_t& offset)
{
    GE_ASSERT_NOTNULL(partition_table, "[Mobile] partition table is nullptr.");
    const uint64_t table_len = SizeOfModelPartitionTable(*partition_table);
    GELOGI("[Mobile] SetModelPartitionTable, base_ptr: %p total_size: %d offset: %d partition_table size: %d",
        static_cast<void*>(base_ptr), total_size, offset, table_len);
    const ge::Status ret = CopyDataToBuffer(base_ptr, total_size, offset,
        static_cast<const void*>(partition_table), table_len);
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] copy failed.");
    return ge::SUCCESS;
}

ge::Status SetModelPartionData(
    OmFileSaveHelper& om_file_save_helper,
    uint8_t* base_ptr, size_t total_size, size_t& offset)
{
    std::vector<ModelPartition>& partition_datas = om_file_save_helper.GetModelPartitions();
    for (size_t i = 0; i < partition_datas.size(); i++) {
        if (partition_datas[i].type != ModelPartitionType::MODEL_DEF) {
            continue;
        }
        if (offset >= total_size || (total_size - offset < static_cast<size_t>(partition_datas[i].size))) {
            GELOGE(ge::FAILED, "[Mobile] save buffer error: type: %u, size: %u, offset: %u, total_size: %u.",
                partition_datas[i].type,
                static_cast<uint32_t>(partition_datas[i].size),
                static_cast<uint32_t>(offset),
                static_cast<uint32_t>(total_size));
            return ge::FAILED;
        }
        GELOGI("[Mobile] SetModelPartionData, base_ptr: %p total_size: %d offset: %d partition_data size: %d",
            static_cast<void*>(base_ptr), total_size, offset, partition_datas[i].size);
        const ge::Status ret = CopyDataToBuffer(base_ptr, total_size, offset,
            reinterpret_cast<void*>(partition_datas[i].data), partition_datas[i].size);
        GE_ASSERT_TRUE(ret == ge::SUCCESS,
            "[Mobile] copy failed.");
    }
    return ge::SUCCESS;
}

ge::Status SetCompiledPartionData(
    uint8_t* base_ptr, size_t total_size,
    const std::vector<ge::BaseBuffer>& compiled_buffers,
    size_t& offset, size_t& align_offset)
{
    for (size_t i = 0; i < compiled_buffers.size(); i++) {
        const std::size_t compiled_buffer_size = compiled_buffers[i].GetSize();
        if (compiled_buffer_size == 0) {
            GELOGI("[Mobile] compiled buffer size is 0!");
            continue;
        }
        if ((offset >= total_size) || (total_size - offset < compiled_buffer_size)) {
            GELOGE(ge::FAILED, "[Mobile] save buffer error: buffer size: %zu, offset: %zu, total_size: %zu.",
                compiled_buffer_size, offset, total_size);
            return ge::FAILED;
        }
        offset += align_offset;
        const ge::Status ret = CopyDataToBuffer(base_ptr, total_size, offset,
            reinterpret_cast<const void*>(compiled_buffers[i].GetData()), compiled_buffer_size);
        GE_ASSERT_TRUE(ret == ge::SUCCESS,
            "[Mobile] copy failed.");
        align_offset = static_cast<size_t>(0);
    }
    return ge::SUCCESS;
}

ge::Status CreateCompiledModelBuffer(
    OmFileSaveHelper& om_file_save_helper,
    ge::BaseBuffer& dst_buffer,
    const std::vector<ge::BaseBuffer>& weight_buffer,
    const std::vector<ge::BaseBuffer>& all_compiled_targets,
    const ge::BaseBuffer& weights_info_buffer)
{
    const size_t total_size = dst_buffer.GetSize();
    auto* base_ptr = dst_buffer.GetData();
    GE_ASSERT_NOTNULL(base_ptr, "[Mobile] base ptr is nullptr.");

    size_t offset = 0;
    const ModelFileHeader& model_header = om_file_save_helper.GetModelFileHeader();
    ge::Status ret = SetModelHeader(base_ptr, total_size, offset, model_header);
    GE_ASSERT_TRUE(ret == ge::SUCCESS,
        "[Mobile] set model header failed.");

    ret = SetModelPartitionTable(om_file_save_helper.GetPartitionTable(), base_ptr, total_size, offset);
    GE_ASSERT_TRUE(ret == ge::SUCCESS,
        "[Mobile] set model partition table failed.");

    size_t align_offset = 0;
    ret = SetModelPartionData(om_file_save_helper, base_ptr, total_size, offset);
    GE_ASSERT_TRUE(ret == ge::SUCCESS,
        "[Mobile] set model partition data failed.");

    ret = SetCompiledPartionData(base_ptr, total_size, weight_buffer, offset, align_offset);
    GE_ASSERT_TRUE(ret == ge::SUCCESS,
        "[Mobile] set compiled partition data (weight buffer) failed.");

    ret = SetCompiledPartionData(base_ptr, total_size, all_compiled_targets, offset, align_offset);
    GE_ASSERT_TRUE(ret == ge::SUCCESS,
        "[Mobile] set compiled partition data (all compiled targets) failed.");

    if (weights_info_buffer.GetSize() != static_cast<size_t>(0)) {
        ret = SetCompiledPartionData(base_ptr, total_size, {weights_info_buffer}, offset, align_offset);
        GE_ASSERT_TRUE(ret == ge::SUCCESS,
            "[Mobile] set compiled partition data (weights info buffer) failed.");
    } else {
        GELOGI("[Mobile] not support save weight info data.");
    }
    return ge::SUCCESS;
}

} // namespace

namespace ge {

Status ModelBufferSaver::SaveCompiledModelToBuffer(
    const ge::GeModelPtr& ge_model,
    const ge::mobile::proto::ModelDef& mobile_model_def,
    const std::vector<ge::BaseBuffer>& weights_buffer,
    const std::vector<ge::BaseBuffer>& all_compiled_targets,
    const ge::BaseBuffer& weights_info_buffer,
    ge::BaseBuffer& dst_buffer)
{
    GE_ASSERT_NOTNULL(ge_model, "[Mobile] ge_model is nullptr.");
    ModelBasicInfo model_info;
    GetModelBasicInfo(ge_model, ModelType::HCS_PARTITION_MODEL, model_info);

    OmFileSaveHelper om_file_save_helper;
    uint32_t align_offset = 0;
    ge::Buffer model_buffer;
    auto ret = SaveModelPartition(om_file_save_helper, model_buffer, ge_model, mobile_model_def, model_info, align_offset);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save model partition failed.");
    ret = SaveCompiledPartion(om_file_save_helper, ModelPartitionType::WEIGHTS_DATA, weights_buffer, align_offset);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save compiled partition WEIGHTS_DATA failed.");
    ret = SaveCompiledPartion(om_file_save_helper, ModelPartitionType::TASK_INFO, all_compiled_targets, align_offset);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save compiled partition TASK_INFO failed.");

    if (weights_info_buffer.GetSize() != static_cast<size_t>(0)) {
        ret = SaveCompiledPartion(
            om_file_save_helper, ModelPartitionType::WEIGHT_INFO, {weights_info_buffer}, align_offset);
        GE_ASSERT_TRUE(ret == SUCCESS,
            "[Mobile] save compiled partition TASK_INFO failed.");
    } else {
        GELOGI("[Mobile] not support save WEIGHT_INFO.");
    }

    ret = SaveModelFileHeader(om_file_save_helper, model_info);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save model file header failed.");

    const ModelFileHeader& model_header = om_file_save_helper.GetModelFileHeader();
    const size_t total_size = sizeof(ModelFileHeader) + model_header.length;
    GELOGI("[Mobile] total_size: %d", total_size);
    compiled_model_buffer_.resize(total_size);
    dst_buffer.SetData(compiled_model_buffer_.data());
    dst_buffer.SetSize(total_size);
    ret = CreateCompiledModelBuffer(
        om_file_save_helper,
        dst_buffer,
        weights_buffer,
        all_compiled_targets,
        weights_info_buffer);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] create compiled model buffer failed.");
    return SUCCESS;
}

} // namespace ge
