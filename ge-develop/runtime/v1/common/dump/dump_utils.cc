/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump_utils.h"
#include "adump_api.h"
#include "runtime/kernel.h"
#include "framework/common/debug/log.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "common/checker.h"

namespace {
constexpr uint64_t kBit8Shift = 56UL;
constexpr uint8_t kCustomLevel2AddrFlag = 0x1U;
constexpr uint8_t kShapeLevel2AddrFlag = 0x2U;
constexpr uint8_t kTilingDataAddrFlag = 0x3U;  // only tiling sink scene need dump tiling data
constexpr size_t kAtomicIndex = 0UL;
constexpr size_t kSizeNumIndex = 1UL;

void GetL0TensorSize(const ge::OpDescPtr &op_desc, std::vector<uint64_t> &tensor_size_list) {
  const auto input_descs = op_desc->GetAllInputsDescPtr();
  const auto output_descs = op_desc->GetAllOutputsDescPtr();
  const std::vector<int64_t> ws_bytes = op_desc->GetWorkspaceBytes();
  const size_t input_size = input_descs.size();
  const size_t output_size = output_descs.size();
  tensor_size_list.resize(input_size + output_size + ws_bytes.size());
  size_t cur_idx = 0UL;
  for (size_t i = 0UL; i < input_size; ++i) {
    const auto &input_desc = input_descs.at(i);
    const int64_t shape_size = input_desc->GetShape().IsScalar() ? 1 : input_desc->GetShape().GetShapeSize();
    const auto data_type = input_desc->GetDataType();
    int64_t tensor_size = ge::GetSizeInBytes(shape_size, data_type);
    if (tensor_size < 0) {
      GELOGW("Op [%s] input [%zu] tensor size [%" PRId64 "] is invalid, set 0 to skip dump.", op_desc->GetNamePtr(), i,
             tensor_size);
      tensor_size = 0;
    }
    tensor_size_list[cur_idx++] = static_cast<uint64_t>(tensor_size);
  }

  for (size_t i = 0UL; i < output_size; ++i) {
    const auto &output_desc = output_descs.at(i);
    const int64_t shape_size = output_desc->GetShape().IsScalar() ? 1 : output_desc->GetShape().GetShapeSize();
    const auto data_type = output_desc->GetDataType();
    int64_t tensor_size = ge::GetSizeInBytes(shape_size, data_type);
    if (tensor_size < 0) {
      GELOGW("Op [%s] output [%zu] tensor size [%" PRId64 "] is invalid, set 0 to skip dump.", op_desc->GetNamePtr(), i,
             tensor_size);
      tensor_size = 0;
    }
    tensor_size_list[cur_idx++] = static_cast<uint64_t>(tensor_size);
  }
  for (const int64_t ws_byte : ws_bytes) {
    tensor_size_list[cur_idx++] = static_cast<uint64_t>(ws_byte);
  }
}
}  // namespace

namespace ge {
static bool OpNeedAssertOrPrintf(const OpDescPtr &op_desc) {
  const std::string kOpDfxOptions = "_op_dfx_options";
  const std::string kOpDfxAssert = "assert";
  const std::string kOpDfxPrintf = "printf";
  std::vector<std::string> dfx_opts;
  (void)ge::AttrUtils::GetListStr(op_desc, kOpDfxOptions, dfx_opts);
  const bool has_assert = std::find(dfx_opts.begin(), dfx_opts.end(), kOpDfxAssert) != dfx_opts.end();
  const bool has_printf = std::find(dfx_opts.begin(), dfx_opts.end(), kOpDfxPrintf) != dfx_opts.end();
  return (has_assert || has_printf);
}


Status ReportL0ExceptionDumpInfo(const OpDescPtr &op_desc, const std::vector<uint64_t> &l0_size_list) {
  bool need_assert_or_printf = OpNeedAssertOrPrintf(op_desc);
  GELOGD("op[%s], need_assert_or_printf[%d]", op_desc->GetName().c_str(), static_cast<int32_t>(need_assert_or_printf));
  if (!gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump) && !need_assert_or_printf) {
    return SUCCESS;
  }
  std::vector<uint64_t> tensor_size_list;
  GetL0TensorSize(op_desc, tensor_size_list);

  rtArgsSizeInfo_t rt_args_size{};
  // if AdumpGetSizeInfoAddr the first param >  Adx::MAX_TENSOR_NUM(1000), it will return nullptr, so truncate it
  const size_t limit_adx_size = (Adx::MAX_TENSOR_NUM - Adx::ADUMP_ARGS_EXCEPTION_HEAD);
  size_t io_nums = (l0_size_list.size() > limit_adx_size) ? limit_adx_size : l0_size_list.size();
  rt_args_size.infoAddr = Adx::AdumpGetSizeInfoAddr(static_cast<uint32_t>(io_nums + Adx::ADUMP_ARGS_EXCEPTION_HEAD),
                                                    rt_args_size.atomicIndex);
  GE_ASSERT_NOTNULL(rt_args_size.infoAddr);
  uint64_t *adump_addr = PtrToPtr<void, uint64_t>(rt_args_size.infoAddr);
  adump_addr[kAtomicIndex] = static_cast<uint64_t>(rt_args_size.atomicIndex);
  adump_addr[kSizeNumIndex] = static_cast<uint64_t>(io_nums);

  GELOGI("Node[%s] atomicIndex[%u], addr[%p] io_num[%zu]", op_desc->GetNamePtr(), rt_args_size.atomicIndex, adump_addr,
         io_nums);

  const auto input_descs = op_desc->GetAllInputsDescPtr();
  const auto output_descs = op_desc->GetAllOutputsDescPtr();
  const size_t input_size = input_descs.size();
  const size_t output_size = output_descs.size();
  for (size_t i = 0UL; i < io_nums; ++i) {
    const uint64_t cur_size = l0_size_list[i];
    const uint8_t bit_flag = static_cast<uint8_t>((cur_size >> kBit8Shift)) & 0xFFU;
    if (cur_size == std::numeric_limits<uint64_t>::max()) {
      adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i] = 0UL;
    } else if ((bit_flag == kCustomLevel2AddrFlag) || (bit_flag == kShapeLevel2AddrFlag) ||
               (bit_flag == kTilingDataAddrFlag)) {
      // level2 addr
      adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i] = l0_size_list[i];
    } else {
      // level1 relevant idx
      GE_ASSERT(cur_size < tensor_size_list.size(), "op [%s] relevant io idx [%zu] is greater than iow size:[%zu].",
                op_desc->GetNamePtr(), cur_size, tensor_size_list.size());
      adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i] = tensor_size_list[cur_size];
      if (need_assert_or_printf && (cur_size == input_size + output_size)) {
        GELOGI("dump op[%s] workspace for assert in static model scenario", op_desc->GetName().c_str());
        adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i] |= (kAssertWorkFlag << kDumpTypeBitNum);
      }
    }
    GELOGI("Node[%s] idx[%zu] l0 size [%" PRIu64 "] cur_size[%" PRIu64 "]", op_desc->GetNamePtr(), i,
           adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i], cur_size);
  }
  GE_CHK_RT_RET(rtSetExceptionExtInfo(&rt_args_size));
  return SUCCESS;
}

Status UpdateL0ExceptionDumpInfoSize(const OpDescPtr &op_desc, std::vector<uint64_t> &l0_size_list,
                                     const size_t &index) {
  bool need_assert_or_printf = OpNeedAssertOrPrintf(op_desc);
  GELOGD("op[%s], need_assert_or_printf[%d]", op_desc->GetName().c_str(), static_cast<int32_t>(need_assert_or_printf));
  if (!gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump) && !need_assert_or_printf) {
    return SUCCESS;
  }

  const size_t nums = l0_size_list.size();
  GE_ASSERT_TRUE(index < nums,
    "op[%s] index[%zu] is greater than l0 list size:[%zu].", op_desc->GetNamePtr(), index, nums);
  for (size_t i = 0U; i < index; ++i) {
    GELOGI("Node[%s] l0 size idx[%zu], val[%" PRIu64 "]", op_desc->GetNamePtr(), i, l0_size_list[i]);
  }

  const auto input_descs = op_desc->GetAllInputsDescPtr();
  const auto output_descs = op_desc->GetAllOutputsDescPtr();
  const size_t input_size = input_descs.size();
  const size_t output_size = output_descs.size();

  std::vector<uint64_t> tensor_size_list;
  GetL0TensorSize(op_desc, tensor_size_list);
  for (size_t i = index; i < nums; i++) {
    const uint64_t cur_size = l0_size_list[i];
    const uint8_t bit_flag = static_cast<uint8_t>((cur_size >> kBit8Shift)) & 0xFFU;
    if (cur_size == std::numeric_limits<uint64_t>::max()) {
      l0_size_list[i] = 0UL;
    } else if ((bit_flag == kCustomLevel2AddrFlag) || (bit_flag == kShapeLevel2AddrFlag) ||
               (bit_flag == kTilingDataAddrFlag)) {
      GELOGI("Node[%s] l0 size idx[%zu], val[%" PRIu64 "], bit flag[%u]",
        op_desc->GetNamePtr(), i, l0_size_list[i], bit_flag);
      continue;
    } else {
      // level1 relevant idx
      GE_ASSERT_TRUE(cur_size < tensor_size_list.size(), "op[%s] relevant io idx[%zu] is greater than iow size[%zu].",
                op_desc->GetNamePtr(), cur_size, tensor_size_list.size());
      l0_size_list[i] = tensor_size_list[cur_size];
      if (need_assert_or_printf && (cur_size == input_size + output_size)) {
        GELOGI("dump op[%s] workspace for assert in static model scenario", op_desc->GetName().c_str());
        l0_size_list[i] |= (kAssertWorkFlag << kDumpTypeBitNum);
      }
    }

    GELOGI("Node[%s] l0 size idx[%zu], val[%" PRIu64 "]", op_desc->GetNamePtr(), i, l0_size_list[i]);
  }

  return SUCCESS;
}

Status SetL0ExceptionSizeInfo(const OpDescPtr &op_desc, const std::vector<uint64_t> &l0_size_list) {
  bool need_assert_or_printf = OpNeedAssertOrPrintf(op_desc);
  GELOGD("op[%s], need_assert_or_printf[%d]", op_desc->GetName().c_str(), static_cast<int32_t>(need_assert_or_printf));
  if (!gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump) && !need_assert_or_printf) {
    return SUCCESS;
  }
  // if AdumpGetSizeInfoAddr the first param >  Adx::MAX_TENSOR_NUM(1000), it will return nullptr, so truncate it
  const size_t limit_adx_size = (Adx::MAX_TENSOR_NUM - Adx::ADUMP_ARGS_EXCEPTION_HEAD);
  size_t nums = (l0_size_list.size() > limit_adx_size) ? limit_adx_size : l0_size_list.size();
  if (nums == 0U) {
    GELOGW("Node [%s] l0 size list is empty", op_desc->GetNamePtr());
    return SUCCESS;
  }

  rtArgsSizeInfo_t rt_args_size{};
  rt_args_size.infoAddr = Adx::AdumpGetSizeInfoAddr(static_cast<uint32_t>(nums + Adx::ADUMP_ARGS_EXCEPTION_HEAD),
                                                    rt_args_size.atomicIndex);
  GE_ASSERT_NOTNULL(rt_args_size.infoAddr);
  uint64_t *adump_addr = PtrToPtr<void, uint64_t>(rt_args_size.infoAddr);
  adump_addr[kAtomicIndex] = static_cast<uint64_t>(rt_args_size.atomicIndex);
  adump_addr[kSizeNumIndex] = static_cast<uint64_t>(nums);

  GELOGI("Node [%s] size info atomic index idx[%zu], val[%" PRIu64 "], size num idx[%zu], val[%" PRIu64 "]",
    op_desc->GetNamePtr(), kAtomicIndex, adump_addr[kAtomicIndex], kSizeNumIndex, adump_addr[kSizeNumIndex]);
  for (size_t i = 0UL; i < nums; i++) {
    adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i] = l0_size_list[i];
    GELOGI("Node [%s] size info idx[%zu], val[%" PRIu64 "]",
      op_desc->GetNamePtr(), Adx::ADUMP_ARGS_EXCEPTION_HEAD + i, adump_addr[Adx::ADUMP_ARGS_EXCEPTION_HEAD + i]);
  }
  GE_CHK_RT_RET(rtSetExceptionExtInfo(&rt_args_size));
  return SUCCESS;
}

}  // namespace ge
