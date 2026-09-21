/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "framework/common/util.h"
#include "framework/common/fmk_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "runtime/rt.h"
#include "graph/def_types.h"

namespace ge {
namespace hybrid {
namespace {
// if dim count is not reach kMaxShapeDims(8), use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = std::numeric_limits<int64_t>::min();
const std::map<int32_t, int32_t> kTopicTypeToRtsFlagMap {
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY), 0},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST), RT_KERNEL_DEVICE_FIRST},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY), RT_KERNEL_HOST_ONLY},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST), RT_KERNEL_HOST_FIRST}
};

static std::atomic<std::uint64_t> g_kernel_id(0U);
}

Status AicpuExtInfoHandler::Parse(const std::string &ext_info) {
  GELOGI("Node[%s] parse ext info start.", node_name_.c_str());
  if (ext_info.empty()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param:ext_info]Node[%s] parse ext info failed as ext info is empty.", node_name_.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext info failed as ext info is empty.", node_name_.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  ext_info_len_ = ext_info.size();
  ext_info_ = MakeUnique<uint8_t[]>(ext_info_len_);
  GE_CHECK_NOTNULL(ext_info_);

  if (memcpy_s(ext_info_.get(), ext_info_len_, ext_info.c_str(), ext_info.size()) != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][ext_info_][%s] Failed to copy ext info", node_name_.c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s] Failed to copy ext info.", node_name_.c_str());
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  input_shape_and_type_.clear();
  output_shape_and_type_.clear();

  const auto ext_info_data = ext_info_.get();
  size_t offset = 0UL;
  while ((offset + sizeof(AicpuExtInfo)) <= ext_info_len_) {
    auto tmp_ext_info_data = PtrAdd(ext_info_data, ext_info_len_, offset);
    GE_CHECK_NOTNULL(tmp_ext_info_data);
    auto &aicpu_ext_info = *(PtrToPtr<uint8_t, AicpuExtInfo>(tmp_ext_info_data));
    GELOGD("Ext infoType=%d, infoLen=%u.", aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
    switch (aicpu_ext_info.infoType) {
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        GE_CHK_STATUS_RET(ParseExtShapeType(aicpu_ext_info), "[Parse][ExtShapeType] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtInputShape(aicpu_ext_info), "[Parse][ExtInputShape] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtOutputShape(aicpu_ext_info), "[Parse][ExtOutputShape] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        GE_CHK_STATUS_RET(ParseExtSessionInfo(aicpu_ext_info), "[Parse][ExtSessionInfo] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP:
        GE_CHK_STATUS_RET(ParseExtBitMap(aicpu_ext_info), "[Parse][ExtBitMap] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_UPDATE_ADDR:
        GE_CHK_STATUS_RET(ParseExtUpdateAddr(aicpu_ext_info), "[Parse][ExtUpdateAddr] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE:
        GE_CHK_STATUS_RET(ParseExtTopicType(aicpu_ext_info), "[Parse][ExtTopicType] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT:
        GE_CHK_STATUS_RET(ParseExtAsyncWait(aicpu_ext_info), "[Parse][ExtAsyncWait] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_WORKSPACE_INFO:
        GE_CHK_STATUS_RET(ParseExtWorkSpaceInfo(aicpu_ext_info), "[Parse][ExtWorkSpaceInfo] failed.");
        break;
      default:
        GELOGD("Node[%s] ignore infoType=%d, infoLen=%u.",
               node_name_.c_str(), aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
        break;
    }
    offset += sizeof(AicpuExtInfo);
    offset += aicpu_ext_info.infoLen;
  }

  GE_IF_BOOL_EXEC(offset != ext_info_len_,
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] ext_info format error, parse not reach end,"
                                     "offset=%zu, ext_info_len=%zu.", node_name_.c_str(), offset, ext_info_len_);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node[%s] ext_info format error,"
                         "parse not reach end, offset=%zu, ext_info_len=%zu.",
                         node_name_.c_str(), offset, ext_info_len_);
                  return ACL_ERROR_GE_PARAM_INVALID;);
  GELOGI("Node[%s] parse ext info end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtWorkSpaceInfo(AicpuExtInfo &aicpu_ext_info) {
  if (aicpu_ext_info.infoLen != sizeof(WorkSpaceInfo)) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Node[%s] parse ext workspace info failed as infoLen must be %zu but %u.",
                       node_name_.c_str(), sizeof(WorkSpaceInfo), aicpu_ext_info.infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse ext workspace info failed as infoLen must be %zu but %u.",
           node_name_.c_str(), sizeof(WorkSpaceInfo), aicpu_ext_info.infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  workspace_info_ = PtrToPtr<char, WorkSpaceInfo>(aicpu_ext_info.infoMsg);
  UpdateWorkSpaceInfo(0U, 0U);
  GELOGI("Node[%s] parse work space info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateWorkSpaceInfo(const uint64_t workspace_length,
                                                const uint64_t workspace_addr) const{
  if (workspace_info_ == nullptr) {
    GELOGD("There is no workspace_info in ext_info, no need update.");
    return FAILED;
  }
  GELOGI("workspace_info size&addr=%lu %lu", workspace_length, workspace_addr);
  workspace_info_->size = workspace_length;
  workspace_info_->addr = workspace_addr;
  return ge::SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtAsyncWait(AicpuExtInfo &aicpu_ext_info) {
  if (aicpu_ext_info.infoLen != sizeof(AsyncWaitInfo)) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Node[%s] parse ext async wait info failed as infoLen must be %zu but %u.",
                       node_name_.c_str(), sizeof(AsyncWaitInfo), aicpu_ext_info.infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse ext async wait info failed as infoLen must be %zu but %u.",
           node_name_.c_str(), sizeof(AsyncWaitInfo), aicpu_ext_info.infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  async_wait_ = PtrToPtr<char, AsyncWaitInfo>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] parse async wait info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtShapeType(AicpuExtInfo &aicpu_ext_info) const {
  GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != sizeof(int32_t),
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext shape type failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(int32_t), aicpu_ext_info.infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Size]Node[%s] parse ext shape type failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(int32_t), aicpu_ext_info.infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  const auto type = PtrToPtr<char, const int32_t>(aicpu_ext_info.infoMsg);

  GE_IF_BOOL_EXEC(*type != unknown_type_,
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext shape type failed as need %d but %d.",
                                     node_name_.c_str(), unknown_type_, *type);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Type]Node[%s] parse ext shape type failed as need %d but %d.",
                         node_name_.c_str(), unknown_type_, *type);
                  return ACL_ERROR_GE_PARAM_INVALID;);
  GELOGI("Node[%s] parse ext shape type success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtInputShape(AicpuExtInfo &aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != (input_num_ * sizeof(AicpuShapeAndType)),
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext input shape failed as infoLen must be "
                                     "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                                     node_name_.c_str(), input_num_, sizeof(AicpuShapeAndType),
                                     aicpu_ext_info.infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse ext input shape failed as infoLen must be "
                         "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name_.c_str(), input_num_, sizeof(AicpuShapeAndType), aicpu_ext_info.infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  const auto input = PtrToPtr<char, AicpuShapeAndType>(aicpu_ext_info.infoMsg);

  for (uint32_t index = 0U; index < input_num_; ++index) {
    input_shape_and_type_.emplace_back(PtrAdd<AicpuShapeAndType>(input, static_cast<size_t>(input_num_),
                                                                 static_cast<size_t>(index)));
  }
  GELOGI("Node[%s] parse ext input shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtOutputShape(AicpuExtInfo &aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != (output_num_ * sizeof(AicpuShapeAndType)),
                  REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext output shape failed as infoLen must be "
                                     "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                                     node_name_.c_str(), output_num_, sizeof(AicpuShapeAndType),
                                     aicpu_ext_info.infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse ext output shape failed as infoLen must be "
                         "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name_.c_str(), output_num_, sizeof(AicpuShapeAndType), aicpu_ext_info.infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  const auto output = PtrToPtr<char, AicpuShapeAndType>(aicpu_ext_info.infoMsg);
  for (uint32_t index = 0U; index < output_num_; ++index) {
    output_shape_and_type_.emplace_back(PtrAdd<AicpuShapeAndType>(output, static_cast<size_t>(output_num_),
                                                                  static_cast<size_t>(index)));
  }
  GELOGI("Node[%s] parse ext output shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtSessionInfo(AicpuExtInfo &aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != sizeof(AicpuSessionInfo),
                  REPORT_INNER_ERR_MSG("E19999",
                                     "Node[%s] parse ext session info failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(SessionInfo), aicpu_ext_info.infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse ext session info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(SessionInfo), aicpu_ext_info.infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  session_info_ = PtrToPtr<char, AicpuSessionInfo>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] parse session info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtBitMap(AicpuExtInfo &aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != sizeof(uint64_t),
                  REPORT_INNER_ERR_MSG("E19999",
                                     "Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(uint64_t), aicpu_ext_info.infoLen);
                  GELOGE(PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(uint64_t), aicpu_ext_info.infoLen);
                  return PARAM_INVALID;);

  bit_map_ = PtrToPtr<char, uint64_t>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] bit_map info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtUpdateAddr(AicpuExtInfo &aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info.infoLen != sizeof(uint32_t),
                  REPORT_INNER_ERR_MSG("E19999",
                                     "Node[%s] parse update_addr info failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(uint32_t), aicpu_ext_info.infoLen);
                  GELOGE(PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse update_addr info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(uint32_t), aicpu_ext_info.infoLen);
                  return PARAM_INVALID;);

  update_addr_ = PtrToPtr<char, uint32_t>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] update_addr info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtTopicType(AicpuExtInfo &aicpu_ext_info) {
  if (aicpu_ext_info.infoLen != sizeof(int32_t)) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Node[%s] parse topic_type info failed as infoLen must be %zu but %u.",
                       node_name_.c_str(), sizeof(int32_t), aicpu_ext_info.infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse topic_type info failed as infoLen must be %zu but %u.",
           node_name_.c_str(), sizeof(int32_t), aicpu_ext_info.infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(aicpu_ext_info.infoMsg);
  const int32_t type = *(PtrToPtr<char, int32_t>(aicpu_ext_info.infoMsg));
  // Use the 5th and 6th bits of `type` indicate the value of topic_type.
  // xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
  // xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
  // xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
  // xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
  // Use the 9th-11th bits of `type` indicate the value of qos. 12th indicate qos on/off
  // xxxxxxxx xxxxxxxx xxxx0000 xxxxxxxx: qos off
  // xxxxxxxx xxxxxxxx xxxx1000 xxxxxxxx: qos on, level=0(min level)
  // xxxxxxxx xxxxxxxx xxxx1111 xxxxxxxx: qos on, level=7(max level)
  deploy_type_flag_ = TopicTypeToRtsFlag(type);
  if (deploy_type_flag_ == -1) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext topic type failed as need %d %d %d %d but %d.",
                       node_name_.c_str(),
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST,
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST,
                       type);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Type]Node[%s] parse ext shape type failed as need %d %d %d %d but %d.",
           node_name_.c_str(),
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST,
           type);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  qos_level_flag_ = (static_cast<uint32_t>(type) & 0xf00U);
  GELOGI("Node[%s] parse ext topic type info success infoLen=%u, topic_type=%d, rts_flag=%d, qos_flag=%u.",
         node_name_.c_str(), aicpu_ext_info.infoLen, type, deploy_type_flag_, qos_level_flag_);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateExecuteMode(const bool flag) {
  if (bit_map_ == nullptr) {
    GELOGD("There is no bit_map in ext_info, no need update.");
    return SUCCESS;
  }
  if (flag) {
    *(bit_map_) |= 1UL;
  } else {
    *(bit_map_) &= static_cast<uint64_t>(~(1ULL));
  }
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateSessionInfo(const uint64_t session_id, const uint64_t kernel_id,
                                              const bool sess_flag) const {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->kernelId = kernel_id;
  session_info_->sessFlag = sess_flag;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateEventId(const uint32_t event_id) const {
  if (async_wait_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "async_wait_ is nullptr.");
    GELOGE(FAILED, "[Check][Param] async_wait_ is nullptr.");
    return FAILED;
  }
  async_wait_->waitType = 1U;
  async_wait_->waitId = event_id;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateSessionInfoId(const uint64_t session_id) const {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->kernelId = GenerateKernelId();
  session_info_->sessFlag = true;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateInputShapeAndType(const uint32_t input_index, const GeTensorDesc &input_desc) {
  if (input_index >= input_num_) {
    REPORT_INNER_ERR_MSG("E19999", "[Update][Input] Node[%s] index out of range! index = %u, input_num = %u",
                       node_name_.c_str(), input_index, input_num_);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Update][Input] Node[%s] index out of range! index = %u, input_num = %u",
           node_name_.c_str(), input_index, input_num_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  const auto &shape = input_desc.GetShape();

  GE_CHK_STATUS_RET(UpdateShapeAndType(shape, input_desc.GetDataType(),
                                       input_shape_and_type_[static_cast<size_t>(input_index)]),
                    "[Update][ShapeAndType] failed, Node[%s] input[%u] .",
                    node_name_.c_str(), input_index);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateOutputShapeAndType(const uint32_t output_index, const GeTensorDesc &output_desc) {
  if (output_index >= output_num_) {
    REPORT_INNER_ERR_MSG("E19999", "[Update][Output] Node[%s] index out of range! index = %u, output_num = %u",
                       node_name_.c_str(), output_index, output_num_);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Update][Output] Node[%s] index out of range! index = %u, output_num = %u",
           node_name_.c_str(), output_index, output_num_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  auto shape = output_desc.GetShape();

  // shape range need use range update shape
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    std::vector<std::pair<int64_t, int64_t>> range;
    const auto range_ret = output_desc.GetShapeRange(range);
    GE_IF_BOOL_EXEC(range_ret != GRAPH_SUCCESS,
                    REPORT_INNER_ERR_MSG("E19999", "Node[%s] is shape range type but get GetShapeRange failed, ret=%u",
                                       node_name_.c_str(), range_ret);
                    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
                           "[Invoke][GetShapeRange]Node[%s] is shape range type but get GetShapeRange failed, ret=%u",
                           node_name_.c_str(), range_ret);
                    return ACL_ERROR_GE_INTERNAL_ERROR;);
    for (size_t k = 0U; k < range.size(); ++k) {
      if ((shape.GetDim(k) < 0) && (k < range.size())) {
        GELOGD("Node[%s] output[%u] update dim[%zu] from %ld to range max %ld.",
               node_name_.c_str(), output_index, k, shape.GetDim(k), range[k].second);
        (void)shape.SetDim(k, range[k].second);
      }
    }
  }

  return UpdateShapeAndType(shape, output_desc.GetDataType(),
                            output_shape_and_type_[static_cast<size_t>(output_index)]);
}

Status AicpuExtInfoHandler::UpdateInputShapeAndType(const size_t input_index, const GeTensorDesc &input_desc,
                                                    const std::vector<int64_t> &input_dims) {
  if (input_index >= input_num_) {
    REPORT_INNER_ERR_MSG("E19999", "[Update][Input] Node[%s] index out of range! index = %zu, input_num = %u",
                       node_name_.c_str(), input_index, input_num_);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Update][Input] Node[%s] index out of range! index = %zu, input_num = %u",
           node_name_.c_str(), input_index, input_num_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  return UpdateShapeAndType(input_dims, input_desc.GetDataType(), *input_shape_and_type_[input_index]);
}

Status AicpuExtInfoHandler::UpdateOutputShapeAndType(const size_t output_index, const GeTensorDesc &output_desc,
                                                     const std::vector<int64_t> &output_dims) {
  if (output_index >= output_num_) {
    REPORT_INNER_ERR_MSG("E19999", "[Update][Output] Node[%s] index out of range! index = %zu, output_num = %u",
                       node_name_.c_str(), output_index, output_num_);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Update][Output] Node[%s] index out of range! index = %zu, output_num = %u",
           node_name_.c_str(), output_index, output_num_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  return UpdateShapeAndType(output_dims, output_desc.GetDataType(), *output_shape_and_type_[output_index]);
}

Status AicpuExtInfoHandler::UpdateShapeAndType(const std::vector<int64_t> &dims, const DataType data_type,
                                               AicpuShapeAndType &shape_and_type) const {
  const auto dim_num = dims.size();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DimNum]Node[%s] Update shape and type failed, as dim_num %zu is over max shape dims %u.",
           node_name_.c_str(), dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] Update shape and type failed, as dim_num %zu is over max shape dims %u.",
                       node_name_.c_str(), dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  size_t index = 0U;
  for (; index < dim_num; ++index) {
    shape_and_type.dims[index] = dims[index];
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type.dims[index] = kDimEndFlag;
  }

  shape_and_type.type = static_cast<int32_t>(data_type);
  return SUCCESS;
}

Status AicpuExtInfoHandler::GetOutputShapeAndType(const uint32_t output_index, GeShape &shape, DataType &data_type) {
  if (output_index >= output_num_) {
    REPORT_INNER_ERR_MSG("E19999", "[Get][Output] Node[%s] index out of range! index = %u, output_num = %u",
                       node_name_.c_str(), output_index, output_num_);
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][Output] Node[%s] index out of range! index = %u, output_num = %u",
           node_name_.c_str(), output_index, output_num_);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  GetShapeAndType(output_shape_and_type_[static_cast<size_t>(output_index)], shape, data_type);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateShapeAndType(const GeShape &shape, const DataType data_type,
                                               AicpuShapeAndType *const shape_and_type) {
  static_cast<void>(data_type);
  const auto dim_num = shape.GetDimNum();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DimNum]Update shape and type failed, as dim_num %zu is over max shape dims %u.",
           dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    REPORT_INNER_ERR_MSG("E19999", "Update shape and type failed, as dim_num %zu is over max shape dims %u.",
                       dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t index = 0U;
  for (; index < dim_num; ++index) {
    shape_and_type->dims[index] = shape.GetDim(index);
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  // now only support update shape, type is not support
  return SUCCESS;
}

void AicpuExtInfoHandler::GetShapeAndType(const AicpuShapeAndType *const shape_and_type, GeShape &shape,
                                          DataType &data_type) {
  std::vector<int64_t> dims;
  for (uint32_t i = 0U; i < aicpu::FWKAdapter::kMaxShapeDims; ++i) {
    if (shape_and_type->dims[i] == kDimEndFlag) {
      break;
    }
    dims.emplace_back(shape_and_type->dims[i]);
  }
  data_type = static_cast<DataType>(shape_and_type->type);
  shape = GeShape(dims);
}

int32_t AicpuExtInfoHandler::TopicTypeToRtsFlag(const int32_t topic_type) {
  const auto it = kTopicTypeToRtsFlagMap.find(static_cast<int32_t>(((static_cast<uint32_t>(topic_type)) & 0x30U) >> 4));
  if (it != kTopicTypeToRtsFlagMap.end()) {
    return it->second;
  }

  return -1;
}

uint64_t AicpuExtInfoHandler::GenerateKernelId() {
  return g_kernel_id++;
}
}  // namespace hybrid
}  // namespace ge
