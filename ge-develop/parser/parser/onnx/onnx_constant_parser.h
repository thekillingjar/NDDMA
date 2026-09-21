/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_ONNX_CONSTANT_PARSER_H_
#define GE_PARSER_ONNX_ONNX_CONSTANT_PARSER_H_

#include <string>
#include "common/util.h"
#include "graph/tensor.h"
#include "parser/common/data_op_parser.h"
#include "parser/onnx/onnx_op_parser.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxConstantParser : public OnnxOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::Operator &op_def) override;

 private:
  static Status ParseConstFromInput(const ge::onnx::NodeProto *op_src, ge::Operator &op_def);
  static Status ParseConvertTensor(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor);
  static Status ParseConvertData(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor, int count);
  static void ParseConvertDataElements(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor, int count,
                               int64_t data_type);
  static Status ParseConvertDataType(const ge::onnx::TensorProto &tensor_proto, ge::Tensor &tensor);

  template <typename T>
  static Status SetTensorData(int32_t val_size, const google::protobuf::RepeatedField<T> &val_vector, int count,
                              Tensor &tensor) {
    bool zeros_like = (count != val_size && val_size == 1);
    unique_ptr<T[]> addr(new(std::nothrow) T[count]());
    GE_CHECK_NOTNULL(addr);
    int minCount = (count > val_size) ? val_size : count;
    if (!zeros_like) {
      for (int32_t i = 0; i < minCount; i++) {
        *(addr.get() + i) = val_vector.Get(i);
      }
      for (int32_t i = minCount; i < count; i++) {
        *(addr.get() + i) = val_vector.Get(minCount - 1);
      }
    } else {
      for (int32_t i = 0; i < count; i++) {
        *(addr.get() + i) = val_vector.Get(0);
      }
    }

    DataType data_type = tensor.GetTensorDesc().GetDataType();
    switch (data_type) {
      case DT_BOOL: {
        unique_ptr<bool[]> addr_trans(new(std::nothrow) bool[count]());
        GE_CHECK_NOTNULL(addr_trans);
        for (int32_t i = 0; i < count; i++) {
          *(addr_trans.get() + i) = static_cast<bool>(
              std::fabs(*((addr).get() + i)) > std::numeric_limits<T>::epsilon());
        }
        (tensor).SetData(PtrToPtr<bool, uint8_t>(addr_trans.get()), (count) * sizeof(bool));
        break;
      }
#define CASE_SET_DATA(dt_type, value_type, addr, count, tensor)                                     \
  case dt_type: {                                                                                   \
    unique_ptr<value_type[]> addr_trans(new(std::nothrow) value_type[count]());                     \
    GE_CHECK_NOTNULL(addr_trans);                                                                   \
    for (int32_t i = 0; i < (count); i++) {                                                         \
      *(addr_trans.get() + i) = static_cast<value_type>(*((addr).get() + i));                       \
    }                                                                                               \
    (tensor).SetData(reinterpret_cast<uint8_t *>(addr_trans.get()), (count) * sizeof(value_type));  \
    break; }                                                                                        \

      CASE_SET_DATA(DT_FLOAT16, uint16_t, addr, count, tensor)
      CASE_SET_DATA(DT_BF16, uint16_t, addr, count, tensor)
      CASE_SET_DATA(DT_INT16, int16_t, addr, count, tensor)
      CASE_SET_DATA(DT_INT8, int8_t, addr, count, tensor)
      CASE_SET_DATA(DT_UINT16, uint16_t, addr, count, tensor)
      CASE_SET_DATA(DT_UINT8, uint8_t, addr, count, tensor)
      CASE_SET_DATA(DT_UINT32, uint32_t, addr, count, tensor)
      CASE_SET_DATA(DT_FLOAT8_E5M2, uint8_t, addr, count, tensor)
      CASE_SET_DATA(DT_FLOAT8_E4M3FN, uint8_t, addr, count, tensor)
#undef CASE_SET_DATA
      default:
        tensor.SetData(PtrToPtr<T, uint8_t>(addr.get()), count * sizeof(T));
        break;
    }
    return SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_CONSTANT_PARSER_H_
