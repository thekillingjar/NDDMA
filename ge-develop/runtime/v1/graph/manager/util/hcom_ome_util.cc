/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/util/hcom_ome_util.h"

#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "common/math/math_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
constexpr int32_t kScalarShapeSize = 1;
}
const std::map<int64_t, HcclDataType> kConstOpHcclDataType = {
    {ge::DT_FLOAT, HCCL_DATA_TYPE_FP32},
    {ge::DT_FLOAT16, HCCL_DATA_TYPE_FP16},
    {ge::DT_DOUBLE, HCCL_DATA_TYPE_FP64},
    {ge::DT_BOOL, HCCL_DATA_TYPE_INT8},  // 对于HcomAllGather类搬移类算子, 将int8当作bool
    {ge::DT_INT8, HCCL_DATA_TYPE_INT8},
    {ge::DT_INT16, HCCL_DATA_TYPE_INT16},
    {ge::DT_INT32, HCCL_DATA_TYPE_INT32},
    {ge::DT_INT64, HCCL_DATA_TYPE_INT64},
    {ge::DT_UINT8, HCCL_DATA_TYPE_UINT8},
    {ge::DT_UINT16, HCCL_DATA_TYPE_UINT16},
    {ge::DT_UINT32, HCCL_DATA_TYPE_UINT32},
    {ge::DT_UINT64, HCCL_DATA_TYPE_UINT64},
    {ge::DT_BF16, HCCL_DATA_TYPE_BFP16},
    {ge::DT_HIFLOAT8, HCCL_DATA_TYPE_HIF8},
    {ge::DT_FLOAT8_E5M2, HCCL_DATA_TYPE_FP8E5M2},
    {ge::DT_FLOAT8_E4M3FN, HCCL_DATA_TYPE_FP8E4M3},
};

static std::map<HcclDataType, int32_t> kConstOpHcclDataTypeSize = {
    {HCCL_DATA_TYPE_FP64, sizeof(float) * 2U},
    {HCCL_DATA_TYPE_FP32, sizeof(float)},
    {HCCL_DATA_TYPE_FP16, sizeof(float) / 2U},
    {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
    {HCCL_DATA_TYPE_INT16, sizeof(int16_t)},
    {HCCL_DATA_TYPE_INT32, sizeof(int32_t)},
    {HCCL_DATA_TYPE_INT64, sizeof(int64_t)},
    {HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)},
    {HCCL_DATA_TYPE_UINT16, sizeof(uint16_t)},
    {HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)},
    {HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)},
    {HCCL_DATA_TYPE_BFP16, sizeof(float) / 2U},
    {HCCL_DATA_TYPE_HIF8, sizeof(uint8_t)},
    {HCCL_DATA_TYPE_FP8E5M2, sizeof(uint8_t)},
    {HCCL_DATA_TYPE_FP8E4M3, sizeof(uint8_t)},
};

static std::map<HorovodReduceOp, HcclReduceOp> kHorovodRedOpToHcclRedOp = {
    {HOROVOD_REDUCE_SUM, HCCL_REDUCE_SUM},           {HOROVOD_REDUCE_MIN, HCCL_REDUCE_MIN},
    {HOROVOD_REDUCE_MAX, HCCL_REDUCE_MAX},           {HOROVOD_REDUCE_PROD, HCCL_REDUCE_PROD},
    {HOROVOD_REDUCE_RESERVED, HCCL_REDUCE_RESERVED}
};

Status HcomOmeUtil::GetHcclDataType(const ge::DataType src_data_type, HcclDataType &hccl_data_type) {
  const auto &iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  GE_CHK_BOOL_RET_STATUS(iter != kConstOpHcclDataType.end(),
                         PARAM_INVALID,
                         "Transform src_data_type:[%d, %s] to hccl_data_type failed.",
                         static_cast<int32_t>(src_data_type),
                         ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
  hccl_data_type = iter->second;
  return SUCCESS;
}

Status HcomOmeUtil::GetHcclDataType(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][KernelHcclInfo] failed, op:%s(%s).",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  GELOGI("GetHcclDataType start, node[%s], opType[%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }
  ge::DataType src_data_type = ge::DT_FLOAT;
  for (size_t i = 0U; i < kernel_hccl_infos.size(); i++) {
    if (op_desc->GetType() == HCOMRECEIVE) {
      const bool ret = ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, src_data_type);
      if (!ret) {
        REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail", HCOM_ATTR_DATA_TYPE.c_str(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(PARAM_INVALID, "[Get][Attr] %s in op:%s(%s) fail", HCOM_ATTR_DATA_TYPE.c_str(),
               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return PARAM_INVALID;
      }
    } else {
      const auto input_desc_ptr = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
      GE_CHECK_NOTNULL(input_desc_ptr);
      src_data_type = input_desc_ptr->GetDataType();
    }

    const auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
    if (iter == kConstOpHcclDataType.end()) {
      REPORT_INNER_ERR_MSG("E19999",
                         "Attr:%s in op:%s(%s), value data_type:%s, not support in kConstOpHcclDataType now, "
                         "check invalid",
                         HCOM_ATTR_DATA_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                         ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in op:%s(%s), value data_type:%s, "
             "not support in kConstOpHcclDataType now", HCOM_ATTR_DATA_TYPE.c_str(), op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), ge::TypeUtils::DataTypeToSerialString(src_data_type).c_str());
      return PARAM_INVALID;
    }

    kernel_hccl_infos[i].dataType = iter->second;
  }
  return SUCCESS;
}

Status HcomOmeUtil::GetHcclTypeSize(const HcclDataType data_type, int32_t &size) {
  const auto iter = kConstOpHcclDataTypeSize.find(data_type);
  GE_CHK_BOOL_EXEC(iter != kConstOpHcclDataTypeSize.end(), return PARAM_INVALID,
                   "[Check][Param] param data_type:%d not find", data_type);
  size = iter->second;
  return SUCCESS;
}

Status HcomOmeUtil::GetAlignedTensorSize(const ge::GeTensorDesc &tensor_desc,
                                         const int32_t align_size,
                                         int64_t &output_size) {
  GE_CHK_BOOL_RET_STATUS(align_size > 0, FAILED, "[Check][Param] align size = %d should be greater than 0.",
                         align_size);
  GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(tensor_desc, output_size),
                    "[Get][Size] from TensorDesc of hcom recv op failed.");
  GE_CHK_STATUS_RET(CheckInt64AddOverflow(output_size, static_cast<int64_t>(align_size)),
                    "[Check][Param] output size = %" PRId64 " and align size = %d beyond INT64_MAX after add.",
                    output_size, align_size);
  output_size = (output_size + align_size - 1) / align_size * align_size;
  return SUCCESS;
}

Status HcomOmeUtil::GetHcomP2pCount(const ge::GeTensorDesc &tensor_desc,
                                    const HcclDataType data_type,
                                    const HcomOperationType p2p_type,
                                    int64_t &count) {
  int64_t total_size = 0;
  constexpr int32_t align_size = 512;
  int32_t size = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(data_type, size), "[Get][HcclTypeSize] failed, datatype = %d",
                    data_type);
  if (p2p_type == HCOM_OP_TYPE_RECV) {
    GE_CHK_STATUS_RET(GetAlignedTensorSize(tensor_desc, align_size, total_size),
                      "[Get][OutputSize] failed, align_size = %d", align_size);
  } else if (p2p_type == HCOM_OP_TYPE_SEND) {
    int64_t input_size = 0;
    int64_t block_size = 0;
    GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(tensor_desc, input_size),
                      "[Get][Size] from TensorDesc of hcom send op failed.");
    const int64_t shape_size =
        tensor_desc.GetShape().IsScalar() ? kScalarShapeSize : tensor_desc.GetShape().GetShapeSize();
    GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                      "[Check][Param] Product of shape size = %lld and size = %d beyond INT64_MAX", shape_size, size);
    block_size = (input_size + align_size - 1) / align_size * align_size;
    GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size),
                      "[Check][Param] Total size:%lld is beyond the INT64_MAX", total_size);
    total_size += block_size;
  } else {
    GELOGE(FAILED, "Unsupported operation type = %d.", p2p_type);
    return PARAM_INVALID;
  }
  GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "[Check][Param] size is zero.");
  count = total_size / size;
  GE_CHK_BOOL_RET_STATUS((total_size % size) == 0, PARAM_INVALID,
                         "[Check][Param] total size = %d is not divisiable by size = %d.",
                         static_cast<int32_t>(total_size), size);
  return SUCCESS;
}

Status HcomOmeUtil::GetHcomCount(const ge::ConstOpDescPtr &op_desc, const HcclDataType data_type,
                                 const bool is_allgather, int64_t &count) {
  const auto GetAlignedSize = [](const int64_t input_size)->int64_t {
    constexpr int32_t align_size = 512;
    return (input_size + align_size - 1) / align_size * align_size;
  };
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHCOMOp(op_desc->GetType())) {
    REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) is not hcom op, check invalid", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Op:%s(%s) is not hcom op", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  int64_t total_size = 0;
  constexpr int32_t align_size = 512;
  int32_t size = 0;
  bool is_continuous_input = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_continuous_input);
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(data_type, size), "[Get][HcclTypeSize] fail, datatype:%d", data_type);
  if (op_desc->GetType() == HCOMRECEIVE) {
    for (uint32_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
      GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(i));
      GE_CHK_STATUS_RET(GetAlignedTensorSize(*op_desc->GetOutputDescPtr(i), align_size, total_size),
                        "[Get][OutputSize] failed, align_size = %" PRId64, align_size);
    }
  } else {
    for (uint32_t i = 0U; i < op_desc->GetInputsSize(); i++) {
      int64_t input_size = 0;
      int64_t block_size = 0;
      GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(i));
      GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(i), input_size),
                        "[Get][Size] from TensorDesc failed, op:%s, input index:%u", op_desc->GetName().c_str(), i);

      if (op_desc->GetType() == HCOMREDUCESCATTER) {
        int32_t rank_size = 0;
        GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_RANK_SIZE, rank_size), PARAM_INVALID,
                               "[Get][Attr] %s in op:%s(%s) failed", HCOM_ATTR_RANK_SIZE.c_str(),
                               op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GE_CHK_BOOL_RET_STATUS(rank_size != 0, PARAM_INVALID, "[Check][Param] rank size is zero");
        const int64_t shape_size = op_desc->GetInputDescPtr(static_cast<uint32_t>(i))->GetShape().GetShapeSize();
        GE_CHK_STATUS_RET(ge::CheckInt64Uint32MulOverflow(shape_size, static_cast<uint32_t>(size)),
                          "[Check][Param] Product of shape size:%" PRId64 " and size:%d beyond INT64_MAX, op:%s(%s)",
                          shape_size, size, op_desc->GetName().c_str(), op_desc->GetType().c_str());
        block_size = is_continuous_input ? GetAlignedSize(input_size) / rank_size : (shape_size * size) / rank_size;
        GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size),
                          "[Check][Param] Total size:%" PRId64 " is beyond the INT64_MAX, op:%s(%s)",
                          total_size, op_desc->GetName().c_str(), op_desc->GetType().c_str());
        total_size = total_size + block_size;
        continue;
      }
      const int64_t shape_size = op_desc->GetInputDescPtr(i)->GetShape().IsScalar() ? 1 :
          op_desc->GetInputDescPtr(i)->GetShape().GetShapeSize();
      GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                        "[Check][Param] Product of shape size:%" PRId64 " and size:%d beyond INT64_MAX, op:%s(%s)",
                        shape_size, size, op_desc->GetName().c_str(), op_desc->GetType().c_str());

      // dynamic shape hccl op get size from output tensor desc
      bool is_unknown_shape = false;
      (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape);
      if (is_unknown_shape && (op_desc->GetOutputDescPtr(i) != nullptr)) {
        input_size = shape_size * size;
      }

      GELOGD("hcom util node %s inputsize %" PRId64 ", shapesize %" PRId64 ", datasize %d.",
        op_desc->GetName().c_str(), input_size, shape_size, size);
      GE_CHK_STATUS_RET(CheckInt64AddOverflow(input_size, align_size),
                        "[Check][Param] input_size:%" PRId64 " and align_size:%" PRId64 " beyond INT64_MAX after add.",
                        input_size, align_size);
      if (op_desc->GetType() == HCOMALLTOALL) {
        block_size = shape_size * size ;
        GELOGI("[%s] block size = %" PRId64, op_desc->GetName().c_str(), block_size);
      } else if (is_allgather) {
        block_size = is_continuous_input ? GetAlignedSize(input_size) : (shape_size * size);
      } else {
        block_size = GetAlignedSize(input_size);
      }
      GE_CHK_STATUS_RET(ge::CheckInt64AddOverflow(total_size, block_size),
                        "[Check][Param] Total size:%" PRId64 " is beyond the INT64_MAX", total_size);
      total_size = total_size + block_size;
    }
  }

  GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "[Check][Param] Size is zero");
  count = total_size / size;

  GE_CHK_BOOL_EXEC((total_size % size) == 0, return PARAM_INVALID,
                   "[Check][Param] total_size:%" PRId64 " is not divisiable by size:%d.", total_size, size);
  GELOGI("GetHcomP2pCount count = %" PRId64 ", data_type = %d, op name = %s, op type = %s", count, data_type,
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return SUCCESS;
}

Status HcomOmeUtil::GetHorovodCount(const ge::ConstOpDescPtr &op_desc,
                                    std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHorovodOp(op_desc->GetType())) {
    REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) is not horovod op, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Call][IsHorovodOp] failed, Op:%s(%s) is not horovod op",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  constexpr int64_t align_size = 512;
  int32_t size = 0;
  if (op_desc->GetInputsSize() > kernel_hccl_infos.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) input size:%zu bigger than kernel_hccl_infos.size:%zu",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       op_desc->GetInputsSize(), kernel_hccl_infos.size());
    GELOGE(PARAM_INVALID, "[Call][IsHorovodOp] failed, Op:%s(%s) input size:%zu bigger than kernel_hccl_infos.size:%zu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetInputsSize(), kernel_hccl_infos.size());
    return PARAM_INVALID;
  }
  for (size_t i = 0U; i < op_desc->GetInputsSize(); i++) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclTypeSize(static_cast<HcclDataType>(kernel_hccl_infos[i].dataType), size),
                      "[Call][GetHcclTypeSize] fail, op:%s(%s)",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    int64_t input_size = 0;
    int64_t block_size = 0;
    GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(static_cast<uint32_t>(i)));
    GE_CHK_STATUS_RET(ge::TensorUtils::GetSize(*op_desc->GetInputDescPtr(static_cast<uint32_t>(i)), input_size),
                      "[Get][Size] from TensorDesc failed, op:%s, input index:%zu", op_desc->GetName().c_str(), i);

    const int64_t shape_size = op_desc->GetInputDescPtr(static_cast<uint32_t>(i))->GetShape().GetShapeSize();
    GE_CHK_STATUS_RET(ge::CheckInt64Int32MulOverflow(shape_size, size),
                      "[Check][Param] Product of shape size:%" PRId64 " and size:%d beyond INT64_MAX", shape_size, size);
    if (kernel_hccl_infos[0U].hccl_type == HVDCALLBACKALLGATHER) {
      block_size = shape_size * size;
    } else {
      GE_CHK_STATUS_RET(CheckInt64AddOverflow(input_size, align_size),
                        "[Check][Param] input_size:%" PRId64 " and align_size:%" PRId64 " beyond INT64_MAX after add.",
                        input_size, align_size);
      block_size = (input_size + align_size - 1) / align_size * align_size;
    }

    GE_CHK_BOOL_RET_STATUS(size != 0, PARAM_INVALID, "[Check][Param] Size is zero");
    GE_CHK_BOOL_EXEC((block_size % size) == 0, return PARAM_INVALID,
                     "[Check][Param] block_size:%" PRId64 " is not divisiable by size:%d.", block_size, size);
    kernel_hccl_infos[i].count = block_size / size;
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclCount(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  Status ret = CheckKernelHcclInfo(op_desc, kernel_hccl_infos);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][KernelHcclInfo] failed, the number of GETaskKernelHcclInfo is invalid, op:%s(%s).",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }
  GELOGI("GetHcclCount start, node[%s], opType[%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (IsHCOMOp(op_desc->GetType())) {
    int64_t count = 0;
    ret = GetHcomCount(op_desc, static_cast<HcclDataType>(kernel_hccl_infos[0U].dataType),
                       kernel_hccl_infos[0U].hccl_type == HCOMALLGATHER, count);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][GetHcomCount] Node:%s Optype:%s get the Hcom operator hccl count fail.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
    kernel_hccl_infos[0U].count = count;
  }

  if (IsHorovodOp(op_desc->GetType())) {
    ret = GetHorovodCount(op_desc, kernel_hccl_infos);
    if (ret != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Call][GetHorovodCount] Node:%s Optype:%s get the Horovod hccl operator count fail.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclOperationType(const ge::ConstOpDescPtr &op_desc, HcclReduceOp &op_type) {
  GE_CHECK_NOTNULL(op_desc);

  if (IsHCOMOp(op_desc->GetType())) {
    std::string hcom_op_type;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::GetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, hcom_op_type),
                     REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail", HCOM_ATTR_REDUCE_TYPE.c_str(),
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     return PARAM_INVALID,
                     "[Get][Attr] %s in op:%s(%s) fail", HCOM_ATTR_REDUCE_TYPE.c_str(),
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());

    if (hcom_op_type == "min") {
      op_type = HCCL_REDUCE_MIN;
    } else if (hcom_op_type == "max") {
      op_type = HCCL_REDUCE_MAX;
    } else if (hcom_op_type == "prod") {
      op_type = HCCL_REDUCE_PROD;
    } else if (hcom_op_type == "sum") {
      op_type = HCCL_REDUCE_SUM;
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Attr:%s in Op:%s(%s), hcom_op_type value:%s is not supported now, "
                         "check invalid", HCOM_ATTR_REDUCE_TYPE.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), hcom_op_type.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in Op:%s(%s), hcom_op_type value:%s is not supported now",
             HCOM_ATTR_REDUCE_TYPE.c_str(), op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), hcom_op_type.c_str());
      return PARAM_INVALID;
    }
  }

  if (IsHorovodOp(op_desc->GetType())) {
    int64_t horovod_op_type;
    GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, ATTR_HOROVOD_ATTR_REDUCE_TYPE, horovod_op_type),
                     REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail",
                                        ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(),
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     return PARAM_INVALID,
                     "[Get][Attr] %s in op:%s(%s) fail", ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(),
                     op_desc->GetName().c_str(), op_desc->GetType().c_str());

    const auto iter = kHorovodRedOpToHcclRedOp.find(static_cast<HorovodReduceOp>(horovod_op_type));
    if (iter == kHorovodRedOpToHcclRedOp.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Attr:%s in Op:%s(%s), horovod_op_type value:%" PRId64 " is not supported now, "
                         "check invalid", ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), horovod_op_type);
      GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in Op:%s(%s), "
             "horovod_op_type value:%" PRId64 " is not supported now",
             ATTR_HOROVOD_ATTR_REDUCE_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             horovod_op_type);
      return PARAM_INVALID;
    }
    op_type = iter->second;
  }

  return SUCCESS;
}

Status HcomOmeUtil::GetHcclRootId(const ge::ConstOpDescPtr &op_desc, int64_t &root_id) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::GetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id),
                   REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail",
                                      HCOM_ATTR_ROOT_RANK.c_str(),
                                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
                   return PARAM_INVALID,
                   "[Get][Attr] %s in op:%s(%s) fail", HCOM_ATTR_ROOT_RANK.c_str(),
                   op_desc->GetName().c_str(), op_desc->GetType().c_str());

  return SUCCESS;
}

Status HcomOmeUtil::GetAllRootId(const ge::ConstOpDescPtr &op_desc,
                                 std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if ((op_desc->GetType() == HCOMBROADCAST) || (op_desc->GetType() == HCOMGATHER) ||
      (op_desc->GetType() == HVDCALLBACKBROADCAST) || (op_desc->GetType() == HCOMREDUCE)) {
    GELOGI("GetAllRootId Node[%s] opType[%s] get hccl rootId.", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    int64_t root_id = 0;
    const Status dmrt = GetHcclRootId(op_desc, root_id);
    if (dmrt != SUCCESS) {
      GELOGE(FAILED, "[Get][HcclRootId] fail! domi error: %u", dmrt);
      return FAILED;
    }

    for (size_t i = 0U; i < kernel_hccl_infos.size(); i++) {
      kernel_hccl_infos[i].rootId = root_id;
    }
  }
  return SUCCESS;
}

bool HcomOmeUtil::IsHCOMOp(const std::string &op_type) {
  return (op_type == HCOMALLREDUCE) || (op_type == HCOMALLGATHER) || (op_type == HCOMBROADCAST) ||
         (op_type == HCOMSEND) || (op_type == HCOMRECEIVE) || (op_type == HCOMREDUCESCATTER) ||
         (op_type == HCOMREDUCE) || (op_type == HCOMALLTOALLV) || (op_type == HCOMALLTOALLVC) ||
         (op_type == HCOMALLTOALL) || (op_type == HCOMGATHER) ||
         (op_type == HCOMALLGATHERV) || (op_type == HCOMREDUCESCATTERV);
}

bool HcomOmeUtil::IsHorovodOp(const std::string &op_type) {
  return (op_type == HVDCALLBACKALLREDUCE) || (op_type == HVDCALLBACKALLGATHER) || (op_type == HVDCALLBACKBROADCAST) ||
         (op_type == HVDWAIT);
}

Status HcomOmeUtil::CheckKernelHcclInfo(const ge::ConstOpDescPtr &op_desc,
                                        const std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (IsHCOMOp(op_desc->GetType()) && (kernel_hccl_infos.size() != 1U)) {
    REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) is not hcom op or param kernel_hccl_infos.size:%zu != 1, "
                       "check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Op:%s(%s) is not hcom op or param kernel_hccl_infos.size:%zu != 1",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_hccl_infos.size());
    return PARAM_INVALID;
  }

  if (IsHorovodOp(op_desc->GetType())) {
    if (op_desc->GetType() == HVDWAIT) {
      return SUCCESS;
    }
    if (kernel_hccl_infos.empty() || (op_desc->GetInputsSize() != kernel_hccl_infos.size())) {
      REPORT_INNER_ERR_MSG("E19999", "Param kernel_hccl_infos.size:%zu is empty or not equal to input_desc size:%zu "
                         "in op:%s(%s), check invalid",
                         kernel_hccl_infos.size(), op_desc->GetInputsSize(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(PARAM_INVALID, "Param kernel_hccl_infos.size:%zu is empty or not equal to "
             "input_desc size:%zu in op:%s(%s)", kernel_hccl_infos.size(), op_desc->GetInputsSize(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

void HcomOmeUtil::GetHcclType(const domi::TaskDef &task_def, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  const auto &hccl_def = task_def.kernel_hccl();
  const std::string hccl_type = hccl_def.hccl_type();
  for (size_t i = 0U; i < kernel_hccl_infos.size(); i++) {
    kernel_hccl_infos[i].hccl_type = hccl_type;
  }
}

Status HcomOmeUtil::GetHorovodInputs(const ge::ConstOpDescPtr &op_desc,
                                     std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos) {
  GE_CHECK_NOTNULL(op_desc);
  if (!IsHorovodOp(op_desc->GetType())) {
    return SUCCESS;
  }

  if (CheckKernelHcclInfo(op_desc, kernel_hccl_infos) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][KernelHcclInfo] Node:%s Optype:%s the number of GETaskKernelHcclInfo is invalid.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  if (op_desc->GetType() == HVDWAIT) {
    return SUCCESS;
  }

  for (size_t i = 0U; i < op_desc->GetInputsSize(); i++) {
    const ConstGeTensorDescPtr input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(input_desc);
    GETaskKernelHcclInfo &kernel_hccl_info = kernel_hccl_infos.at(i);
    kernel_hccl_info.input_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(i));
    kernel_hccl_info.dims = input_desc->GetShape().GetDims();
  }
  return SUCCESS;
}
}  // namespace ge
