/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ext_info_handle.h"
#include "framework/common/util.h"
#include "framework/common/fmk_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "exe_graph/runtime/kernel_context.h"
#include "graph/node.h"
#include "graph/def_types.h"
#include "graph/debug/ge_attr_define.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "framework/common/taskdown_common.h"
#include "aicpu_args_handler.h"
#include "graph/def_types.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "framework/common/ge_types.h"

namespace gert {
namespace {
enum class ExtInfoHandler {
  kData,
  kLen,
  kNodeName,
  kInputNum,
  kOutputNum,
  kUnknownShapeType,
  kHostAddr,
  kDeviceAddr,
  kSessionId,
  kIsBlockOp,
  kEventId
};

enum class UpdateExtInfo {
  kExtInfoHandler,
  kInputNum,
  kOutputNum,
  kStream,
  kCommonEnd
};

enum class GetExtInfo {
  kExtInfoHandler,
  kEngineName,
  kCommonEnd
};

void ShapeToStringStream(std::stringstream &ss, const Shape &shape) {
  ss << " [";
  for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
    ss << shape.GetDim(j);
    if (j + 1U < shape.GetDimNum()) {
      ss << ", ";
    }
  }
  ss << "] ";
}

std::vector<std::string> PrintUpdateExtShape(const KernelContext *context) {
  auto input_num = context->GetInputValue<size_t>(static_cast<size_t>(UpdateExtInfo::kInputNum));
  auto output_num = context->GetInputValue<size_t>(static_cast<size_t>(UpdateExtInfo::kOutputNum));

  std::stringstream ss;
  ss << "Input shape: ";
  const size_t idx_start = static_cast<size_t>(UpdateExtInfo::kCommonEnd);
  for (size_t i = 0U; i < input_num; ++i) {
    auto storage_shape = context->GetInputPointer<StorageShape>(static_cast<size_t>(idx_start + i));
    ShapeToStringStream(ss, storage_shape->GetStorageShape());
  }
  ss << "Output shape: ";
  for (size_t i = 0U; i < output_num; ++i) {
    auto storage_shape = context->GetInputPointer<StorageShape>(static_cast<size_t>(idx_start + input_num + i));
    ShapeToStringStream(ss, storage_shape->GetStorageShape());
  }
  return {ss.str()};
}

std::vector<std::string> PrintExtOutputShapes(const KernelContext *context) {
  std::stringstream original_ss;
  std::stringstream storage_ss;
  original_ss << "output original shapes : ";
  storage_ss << "output storage shapes  : ";
  for (size_t index = 0U; index < context->GetOutputNum(); index++) {
    auto shape = context->GetOutputPointer<StorageShape>(index);
    ShapeToStringStream(original_ss, shape->GetOriginShape());
    ShapeToStringStream(storage_ss, shape->GetStorageShape());
  }
  original_ss << ", " << storage_ss.str();
  return {original_ss.str()};
}

// if dim count is not reach kMaxShapeDims(8), use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = std::numeric_limits<int64_t>::min();
const std::map<int32_t, int32_t> kTopicTypeToRtsFlagMap {
  {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY), 0},
  {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST), RT_KERNEL_DEVICE_FIRST},
  {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY), RT_KERNEL_HOST_ONLY},
  {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST), RT_KERNEL_HOST_FIRST}
};
} // namespace

ge::Status AicpuExtInfoHandler::Parse(const std::string &ext_info, uint8_t *host_addr, void *device_addr) {
  GELOGI("Node[%s] parse ext info start.", node_name_.c_str());
  GE_ASSERT_TRUE(!ext_info.empty());
  ext_info_len_ = ext_info.size();
  ext_info_ = host_addr;
  device_ext_info_ = device_addr;

  if (memcpy_s(ext_info_, ext_info_len_, ext_info.c_str(), ext_info_len_) != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][ext_info_][%s] Failed to copy ext info", node_name_.c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s] Failed to copy ext info.", node_name_.c_str());
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  input_shape_and_type_.clear();
  output_shape_and_type_.clear();

  const auto ext_info_data = ext_info_;
  size_t offset = 0UL;
  while ((offset + sizeof(AicpuExtInfo)) <= ext_info_len_) {
    auto tmp_ext_info_data = ge::PtrAdd(ext_info_data, ext_info_len_, offset);
    GE_ASSERT_NOTNULL(tmp_ext_info_data);
    auto &aicpu_ext_info = *(ge::PtrToPtr<uint8_t, AicpuExtInfo>(tmp_ext_info_data));
    GELOGD("Ext infoType=%d, infoLen=%u.", aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
    switch (aicpu_ext_info.infoType) {
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        GE_CHK_STATUS_RET(ParseExtShapeType(aicpu_ext_info), "[Parse][ExtShapeType] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtInputShape(aicpu_ext_info), "[Parse][ExtInputShape] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        output_shape_offset_ = offset + sizeof(AicpuExtInfo);
        output_shape_len_ = aicpu_ext_info.infoLen;
        output_shape_.resize(output_num_);
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

  GE_ASSERT_TRUE(offset == ext_info_len_);
  GELOGI("Node[%s] parse ext info end.", node_name_.c_str());
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtAsyncWait(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(AsyncWaitInfo));
  async_wait_ = ge::PtrToPtr<char, AsyncWaitInfo>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] parse async wait info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtWorkSpaceInfo(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(WorkSpaceInfo));
  workspace_info_ = ge::PtrToPtr<char, WorkSpaceInfo>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] parse workspace info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtShapeType(AicpuExtInfo &aicpu_ext_info) const {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(int32_t));
  const auto type = ge::PtrToPtr<char, const int32_t>(aicpu_ext_info.infoMsg);
  GE_ASSERT_TRUE(*type == unknown_type_);
  GELOGI("Node[%s] parse ext shape type success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtInputShape(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == (input_num_ * sizeof(AicpuShapeAndType)));
  const auto input = ge::PtrToPtr<char, AicpuShapeAndType>(aicpu_ext_info.infoMsg);

  for (uint32_t index = 0U; index < input_num_; ++index) {
    input_shape_and_type_.emplace_back(ge::PtrAdd<AicpuShapeAndType>(input, static_cast<size_t>(input_num_),
                                                                     static_cast<size_t>(index)));
  }
  GELOGI("Node[%s] parse ext input shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtOutputShape(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == (output_num_ * sizeof(AicpuShapeAndType)));
  const auto output = ge::PtrToPtr<char, AicpuShapeAndType>(aicpu_ext_info.infoMsg);

  for (uint32_t index = 0U; index < output_num_; ++index) {
    output_shape_and_type_.emplace_back(ge::PtrAdd<AicpuShapeAndType>(output, static_cast<size_t>(output_num_),
                                                                      static_cast<size_t>(index)));
  }
  GELOGI("Node[%s] parse ext output shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtSessionInfo(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(AicpuSessionInfo));
  session_info_ = ge::PtrToPtr<char, AicpuSessionInfo>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] parse session info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtBitMap(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(uint64_t));
  bit_map_ = ge::PtrToPtr<char, uint64_t>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] bit_map info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtUpdateAddr(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(uint32_t));
  update_addr_ = ge::PtrToPtr<char, uint32_t>(aicpu_ext_info.infoMsg);
  GELOGI("Node[%s] update_addr info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info.infoLen);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::ParseExtTopicType(AicpuExtInfo &aicpu_ext_info) {
  GE_ASSERT_TRUE(aicpu_ext_info.infoLen == sizeof(int32_t));
  GE_ASSERT_NOTNULL(aicpu_ext_info.infoMsg);
  const int32_t type = *(ge::PtrToPtr<char, int32_t>(aicpu_ext_info.infoMsg));
  /*
  * Use the 5th and 6th bits of `type` indicate the value of topic_type.
  * xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
  * xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
  * xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
  * xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
  * Use the 9th-11th bits of `type` indicate the value of qos. 12th indicate qos on/off
  * xxxxxxxx xxxxxxxx xxxx0000 xxxxxxxx: qos off
  * xxxxxxxx xxxxxxxx xxxx1000 xxxxxxxx: qos on, level=0(min level)
  * xxxxxxxx xxxxxxxx xxxx1111 xxxxxxxx: qos on, level=7(max level)
  */
  deploy_type_flag_ = TopicTypeToRtsFlag(type);
  GE_ASSERT_TRUE(deploy_type_flag_ != -1);
  qos_level_flag_ = (static_cast<uint32_t>(type) & 0xf00U);
  GELOGI("Node[%s] of rt2 parse ext topic type info success infoLen=%u, topic_type=%d, rts_flag=%d, qos_flag=%u.",
         node_name_.c_str(), aicpu_ext_info.infoLen, type, deploy_type_flag_, qos_level_flag_);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::UpdateEventId(const uint32_t event_id) {
  if (async_wait_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "async_wait_ is nullptr.");
    GELOGE(ge::FAILED, "[Check][Param] async_wait_ is nullptr.");
    return ge::FAILED;
  }
  async_wait_->waitType = 1U;
  async_wait_->waitId = event_id;
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::UpdateExecuteMode(const bool flag) {
  if (bit_map_ == nullptr) {
    GELOGD("There is no bit_map in ext_info, no need update.");
    return ge::SUCCESS;
  }
  if (flag) {
    *(bit_map_) |= 1UL;
  } else {
    *(bit_map_) &= static_cast<uint64_t>(~(1ULL));
  }
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::UpdateSessionInfo(const uint64_t session_id, const uint64_t kernel_id,
                                                  const bool sess_flag) const {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return ge::SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->kernelId = kernel_id;
  session_info_->sessFlag = sess_flag;
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::UpdateSessionInfoId(const uint64_t session_id) const {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return ge::SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->kernelId = ::ge::hybrid::AicpuExtInfoHandler::GenerateKernelId();
  session_info_->sessFlag = true;
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::UpdateWorkSpaceInfo(const uint64_t workspace_size,
                                                    const uint64_t workspace_addr) const {
  if (workspace_info_ == nullptr) {
    GELOGD("There is no workspace info in ext_info, no need update.");
    return ge::SUCCESS;
  }
  workspace_info_->size = workspace_size;
  workspace_info_->addr = workspace_addr;
  GELOGD("After UpdateWorkSpaceInfo, workspace info size is %lu.", workspace_info_->size);
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::UpdateInputShape(const uint32_t input_index, const Shape &input_shape) {
  GE_ASSERT_TRUE(input_index < input_num_);
  return UpdateShape(input_shape, input_shape_and_type_[static_cast<size_t>(input_index)]);
}

ge::Status AicpuExtInfoHandler::UpdateOutputShape(const uint32_t output_index, const Shape &output_shape) {
  GE_ASSERT_TRUE(output_index < output_num_);
  return UpdateShape(output_shape, output_shape_and_type_[static_cast<size_t>(output_index)]);
}

ge::Status AicpuExtInfoHandler::UpdateShape(const Shape &shape, AicpuShapeAndType *const shape_and_type) {
  const auto dim_num = shape.GetDimNum();
  size_t index = 0U;
  for (; index < dim_num; ++index) {
    shape_and_type->dims[index] = shape.GetDim(index);
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::CopyH2D(const rtStream_t stream) {
  if ((unknown_type_ == ge::DEPEND_SHAPE_RANGE) && (ext_info_len_ > 0) && (device_ext_info_ != nullptr)) {
    GE_ASSERT_NOTNULL(ext_info_);
    GE_ASSERT_RT_OK(rtMemcpyAsync(device_ext_info_, ext_info_len_, ext_info_, ext_info_len_,
                                  RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  }
  return ge::SUCCESS;
}

ge::Status AicpuExtInfoHandler::CopyOutputShapeForThirdOp() {
  // tf/aicpu
  if ((output_shape_len_ > 0) && (device_ext_info_ != nullptr)) {
    auto output_shape_device_addr = ge::ValueToPtr(ge::PtrToValue(device_ext_info_) + output_shape_offset_);
    const size_t host_len = output_shape_.size() * sizeof(AicpuShapeAndType);
    GE_ASSERT_RT_OK(rtMemcpy(output_shape_.data(), host_len, output_shape_device_addr, output_shape_len_,
                             RT_MEMCPY_DEVICE_TO_HOST));
    return ge::SUCCESS;
  }

  // host cpu
  if ((output_shape_len_ > 0) && (ext_info_ != nullptr)) {
    auto output_shape_host_addr = ge::ValueToPtr(ge::PtrToValue(ext_info_) + output_shape_offset_);
    const size_t host_len = output_shape_.size() * sizeof(AicpuShapeAndType);
    GE_ASSERT_EOK(memcpy_s(output_shape_.data(), host_len, output_shape_host_addr, output_shape_len_));
    return ge::SUCCESS;
  }

  GELOGE(ge::FAILED, "CopyOutputShapeForThirdOp fail, output_shape_len_:%zu.", output_shape_len_);
  return ge::FAILED;
}

ge::Status AicpuExtInfoHandler::GetOutputShapeAndType(const uint32_t output_index, Shape &shape,
                                                      ge::DataType &data_type) const {
  GE_ASSERT_TRUE(output_index < output_num_);
  GetShapeAndType(output_shape_[static_cast<size_t>(output_index)], shape, data_type);
  return ge::SUCCESS;
}

void AicpuExtInfoHandler::GetShapeAndType(const AicpuShapeAndType &shape_and_type, Shape &shape,
                                          ge::DataType &data_type) {
  shape.SetDimNum(0U);
  for (uint32_t i = 0U; i < aicpu::FWKAdapter::kMaxShapeDims; ++i) {
    if (shape_and_type.dims[i] == kDimEndFlag) {
      break;
    }
    shape.AppendDim(shape_and_type.dims[i]);
  }
  data_type = static_cast<ge::DataType>(shape_and_type.type);
}

int32_t AicpuExtInfoHandler::TopicTypeToRtsFlag(const int32_t topic_type) {
  const auto it = kTopicTypeToRtsFlagMap.find(
      static_cast<int32_t>(((static_cast<uint32_t>(topic_type)) & 0x30U) >> 4));
  if (it != kTopicTypeToRtsFlagMap.end()) {
    return it->second;
  }

  return -1;
}

ge::graphStatus BuildExtInfoHandle(KernelContext *context) {
  auto ext_data = context->GetInputValue<char *>(static_cast<size_t>(ExtInfoHandler::kData));
  auto ext_len = context->GetInputValue<size_t>(static_cast<size_t>(ExtInfoHandler::kLen));
  auto host_addr = context->GetInputValue<uint8_t *>(static_cast<size_t>(ExtInfoHandler::kHostAddr));
  auto device_addr = context->GetInputValue<void *>(static_cast<size_t>(ExtInfoHandler::kDeviceAddr));
  auto session_id = context->GetInputPointer<uint64_t>(static_cast<size_t>(ExtInfoHandler::kSessionId));
  auto is_block_op = context->GetInputValue<bool>(static_cast<size_t>(ExtInfoHandler::kIsBlockOp));
  auto ext_handle = context->GetOutputPointer<AicpuExtInfoHandler>(0U);
  GE_ASSERT_NOTNULL(ext_data);
  GE_ASSERT_NOTNULL(host_addr);
  GE_ASSERT_NOTNULL(ext_handle);
  GE_ASSERT_NOTNULL(session_id);

  auto ext_str = std::string(ext_data, ext_len);
  GE_ASSERT_SUCCESS(ext_handle->Parse(ext_str, host_addr, device_addr));
  GE_ASSERT_SUCCESS(ext_handle->UpdateSessionInfoId(*session_id));
  if (is_block_op) {
    auto event_id = context->GetInputPointer<uint32_t>(static_cast<size_t>(ExtInfoHandler::kEventId));
    GE_ASSERT_NOTNULL(event_id);
    GE_ASSERT_SUCCESS(ext_handle->UpdateEventId(*event_id));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOutputsForExtHandle(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto node_name = context->GetInputValue<char *>(static_cast<size_t>(ExtInfoHandler::kNodeName));
  auto input_num = context->GetInputValue<size_t>(static_cast<size_t>(ExtInfoHandler::kInputNum));
  auto output_num = context->GetInputValue<size_t>(static_cast<size_t>(ExtInfoHandler::kOutputNum));
  auto unknown_shape_type_val = context->GetInputValue<int32_t>(static_cast<size_t>(ExtInfoHandler::kUnknownShapeType));
  GE_ASSERT_NOTNULL(node_name);
  const auto unknown_type = static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val);
  auto ext_info = ge::MakeUnique<AicpuExtInfoHandler>(node_name, input_num, output_num, unknown_type);
  GE_ASSERT_NOTNULL(ext_info);

  auto av_holder = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(av_holder);
  av_holder->SetWithDefaultDeleter<AicpuExtInfoHandler>(ext_info.release());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(BuildExtInfoHandle).RunFunc(BuildExtInfoHandle).OutputsCreator(CreateOutputsForExtHandle);

ge::graphStatus UpdateExtShape(KernelContext *context) {
  auto ext_handle =
      context->MutableInputPointer<AicpuExtInfoHandler>(static_cast<size_t>(UpdateExtInfo::kExtInfoHandler));
  auto input_num = context->GetInputValue<size_t>(static_cast<size_t>(UpdateExtInfo::kInputNum));
  auto output_num = context->GetInputValue<size_t>(static_cast<size_t>(UpdateExtInfo::kOutputNum));
  // stream can be nullptr
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(UpdateExtInfo::kStream));
  GE_ASSERT_NOTNULL(ext_handle);

  size_t idx_start = static_cast<size_t>(UpdateExtInfo::kCommonEnd);
  for (size_t i = 0U; i < input_num; ++i) {
    auto storage_shape = context->GetInputPointer<StorageShape>(static_cast<size_t>(idx_start + i));
    GE_ASSERT_NOTNULL(storage_shape);
    ext_handle->UpdateInputShape(i, storage_shape->GetStorageShape());
  }
  for (size_t i = 0U; i < output_num; ++i) {
    auto storage_shape = context->GetInputPointer<StorageShape>(static_cast<size_t>(idx_start + input_num + i));
    GE_ASSERT_NOTNULL(storage_shape);
    ext_handle->UpdateOutputShape(i, storage_shape->GetStorageShape());
  }
  GE_ASSERT_SUCCESS(ext_handle->CopyH2D(stream));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(UpdateExtShape).RunFunc(UpdateExtShape).TracePrinter(PrintUpdateExtShape);

ge::graphStatus CreateOutputsForExtOutputShapes(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  auto engine_name =
      context->GetInputValue<char_t *>(static_cast<size_t>(GetExtInfo::kEngineName));
  GE_ASSERT_NOTNULL(engine_name);
  GELOGI("Get engine name: %s", engine_name);
  for (size_t index = 0U; index < context->GetOutputNum(); index++) {
    auto chain = context->GetOutput(index);
    GE_ASSERT_NOTNULL(chain);
    Tensor *shape_tensor = nullptr;
    if (engine_name == ge::kEngineNameHostCpu) {
      // hostcpu引擎创建此kernel传入的是输入shape
      auto input_desc = extend_context->GetInputDesc(index);
      GE_ASSERT_NOTNULL(input_desc);
      shape_tensor = new (std::nothrow) Tensor(StorageShape(),
          input_desc->GetFormat(), input_desc->GetDataType());
    } else {
      // cpu和tf引擎创建此kernel传入的是输出shape
      auto output_desc = extend_context->GetOutputDesc(index);
      GE_ASSERT_NOTNULL(output_desc);
      shape_tensor = new (std::nothrow) Tensor(StorageShape(),
          output_desc->GetFormat(), output_desc->GetDataType());
    }
    GE_ASSERT_NOTNULL(shape_tensor);
    chain->SetWithDefaultDeleter(shape_tensor);
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus UpdateExtWorkSpaceInfo(KernelContext *context) {
  auto workspace_size = context->GetInputPointer<uint64_t>(0U);
  GE_ASSERT_NOTNULL(workspace_size);
  auto tensor_data = context->GetInputValue<gert::GertTensorData *>(1U);
  GE_ASSERT_NOTNULL(tensor_data);
  uint64_t workspace_addr = 0UL;
  if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
    workspace_addr = ge::PtrToValue(tensor_data->GetAddr());
  }

  auto ext_handle = context->GetInputValue<gert::AicpuExtInfoHandler *>(2U); // 2 is for the third input
  GE_ASSERT_NOTNULL(ext_handle);
  ext_handle->UpdateWorkSpaceInfo(*workspace_size, workspace_addr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceInfo(const KernelContext *context, uintptr_t &addr, int64_t &bytes) {
  auto workspace_size = context->GetInputPointer<uint64_t>(0U);
  GE_ASSERT_NOTNULL(workspace_size);
  bytes = *workspace_size;

  auto tensor_data = context->GetInputValue<gert::GertTensorData *>(1U);
  GE_ASSERT_NOTNULL(tensor_data);
  if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
    addr = static_cast<uintptr_t>(ge::PtrToValue(tensor_data->GetAddr()));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DataDumpFillWorkSpace(const KernelContext *context, gert::DataDumpInfoWrapper &data_dump_wrapper) {
  uintptr_t workspace_addr;
  int64_t workspace_bytes;
  if (GetWorkspaceInfo(context, workspace_addr, workspace_bytes) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Data dump GetWorkspaceInfo failed.");
    return ge::GRAPH_FAILED;
  }
  data_dump_wrapper.AddWorkspace(workspace_addr, workspace_bytes);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExceptionDumpFillWorkSpace(const KernelContext *context,
                                           gert::ExceptionDumpInfoWrapper &exception_dump_wrapper) {
  uintptr_t workspace_addr;
  int64_t workspace_bytes;
  if (GetWorkspaceInfo(context, workspace_addr, workspace_bytes) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Excpetion dump GetWorkspaceInfo failed.");
    return ge::GRAPH_FAILED;
  }
  exception_dump_wrapper.AddWorkspace(workspace_addr, workspace_bytes);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(UpdateExtWorkSpaceInfo).RunFunc(UpdateExtWorkSpaceInfo).DataDumpInfoFiller(DataDumpFillWorkSpace)
    .ExceptionDumpInfoFiller(ExceptionDumpFillWorkSpace);

ge::graphStatus GetExtOutputShapes(KernelContext *context) {
  auto ext_handle =
      context->MutableInputPointer<AicpuExtInfoHandler>(static_cast<size_t>(GetExtInfo::kExtInfoHandler));
  GE_ASSERT_NOTNULL(ext_handle);

  GE_ASSERT_SUCCESS(ext_handle->CopyOutputShapeForThirdOp());
  for (size_t i = 0U; i < ext_handle->GetOutputNum(); i++) {
    ge::DataType type;
    auto shape = context->GetOutputPointer<StorageShape>(i);
    GE_ASSERT_NOTNULL(shape);
    GE_ASSERT_SUCCESS(ext_handle->GetOutputShapeAndType(i, shape->MutableStorageShape(), type));
    GE_ASSERT_SUCCESS(ext_handle->GetOutputShapeAndType(i, shape->MutableOriginShape(), type));
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetExtOutputShapes).RunFunc(GetExtOutputShapes).OutputsCreator(CreateOutputsForExtOutputShapes)
    .TracePrinter(PrintExtOutputShapes);
}  // namespace gert
