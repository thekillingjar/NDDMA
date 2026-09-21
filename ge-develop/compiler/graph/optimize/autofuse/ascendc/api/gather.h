/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_GATHER_H__
#define __ASCENDC_API_GATHER_H__

template <typename T1, typename T2>
inline __aicore__ void GatherUbNotFullLoad(const LocalTensor<T1> &dst, const GlobalTensor<T1> &src0,
                                           const GlobalTensor<T2> &src1, uint32_t vectorized_axis_size,
                                           LocalTensor<uint8_t> &tmp_buf) {
  AscendC::DataCopyExtParams param_src1;
  param_src1.blockCount = 1;
  param_src1.blockLen = vectorized_axis_size * sizeof(T2);
  param_src1.srcStride = 0;
  param_src1.dstStride = 0;

  auto src1_ub = tmp_buf[0].template ReinterpretCast<T2>();
  src1_ub.SetSize(vectorized_axis_size);
  AscendC::DataCopyPad(src1_ub, src1, param_src1, AscendC::DataCopyPadExtParams<T2>());
  int32_t event_id_mte2_to_s = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_mte2_to_s);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_mte2_to_s);
  for (uint32_t i = 0; i < vectorized_axis_size; i++) {
    T2 index = src1_ub.GetValue(i);
    T1 value = src0.GetValue(static_cast<uint64_t>(index));
    dst.SetValue(i, value);
  }
}

template <typename T1, typename T2>
inline __aicore__ void GatherUbFullLoad(const LocalTensor<T1> &dst, const GlobalTensor<T1> &src0,
                                        const GlobalTensor<T2> &src1, uint32_t param_last_axis_size,
                                        uint32_t vectorized_axis_size, uint32_t param_last_axis_bytes,
                                        uint32_t vectorized_axis_bytes, LocalTensor<uint8_t> &tmp_buf) {
  AscendC::DataCopyExtParams param_src0;
  param_src0.blockCount = 1;
  param_src0.blockLen = param_last_axis_size * sizeof(T1);
  param_src0.srcStride = 0;
  param_src0.dstStride = 0;

  AscendC::DataCopyExtParams param_src1;
  param_src1.blockCount = 1;
  param_src1.blockLen = vectorized_axis_size * sizeof(T2);
  param_src1.srcStride = 0;
  param_src1.dstStride = 0;

  auto src0_ub = tmp_buf[0].template ReinterpretCast<T1>();
  src0_ub.SetSize(param_last_axis_size);
  uint32_t offset = param_last_axis_bytes;
  auto src1_ub = tmp_buf[offset].template ReinterpretCast<T2>();
  src1_ub.SetSize(vectorized_axis_size);
  offset += vectorized_axis_bytes;

  int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
  int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
  int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
  int32_t event_id_mte3_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
  int32_t event_id_mte3_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
  int32_t event_id_mte2_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));

  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);

  AscendC::DataCopyPad(src0_ub, src0, param_src0, AscendC::DataCopyPadExtParams<T1>());
  AscendC::DataCopyPad(src1_ub, src1, param_src1, AscendC::DataCopyPadExtParams<T2>());

  int32_t data_bytes_t1 = sizeof(T1);
  if constexpr (AscendC::IsSameType<T2, int64_t>::value) {
    auto src1_tmp_ub = tmp_buf[offset].template ReinterpretCast<int32_t>();
    src1_tmp_ub.SetSize(vectorized_axis_size);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id_mte3_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id_mte3_to_v);
    AscendC::Cast(src1_tmp_ub, src1_ub, AscendC::RoundMode::CAST_NONE, vectorized_axis_size);
    AscendC::Muls(src1_tmp_ub, src1_tmp_ub, data_bytes_t1, vectorized_axis_size);
    auto src1_uint32t_ub = src1_tmp_ub.template ReinterpretCast<uint32_t>();
    AscendC::Gather(dst, src0_ub, src1_uint32t_ub, 0, vectorized_axis_size);
  } else {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id_mte3_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id_mte3_to_v);
    AscendC::Muls(src1_ub, src1_ub, data_bytes_t1, vectorized_axis_size);
    auto src1_uint32t_ub = src1_ub.template ReinterpretCast<uint32_t>();
    AscendC::Gather(dst, src0_ub, src1_uint32t_ub, 0, vectorized_axis_size);
  }
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
}

template <typename T1, typename T2>
inline __aicore__ void GatherExtend(const LocalTensor<T1> &dst, const GlobalTensor<T1> &src0,
                                    const GlobalTensor<T2> &src1, uint32_t param_last_axis_size,
                                    uint32_t vectorized_axis_size, LocalTensor<uint8_t> &tmp_buf) {
  if (tmp_buf.GetSize() < vectorized_axis_size * sizeof(T2)) {  // tmpbuffer小于向量化轴空间，param和indices都从GM取数据
    for (uint32_t i = 0; i < vectorized_axis_size; i++) {
      T2 index = src1.GetValue(i);
      T1 value = src0.GetValue(static_cast<uint64_t>(index));
      dst.SetValue(i, value);
    }
  } else {
    uint32_t param_last_axis_bytes = AscendC::AlignUp(param_last_axis_size * sizeof(T1), ONE_BLK_SIZE);
    uint32_t vectorized_axis_bytes = AscendC::AlignUp(vectorized_axis_size * sizeof(T2), ONE_BLK_SIZE);
    uint32_t ub_full_load_bytes = 0;
    if constexpr (AscendC::IsSameType<T2, int64_t>::value) {
      ub_full_load_bytes = param_last_axis_bytes + vectorized_axis_bytes + vectorized_axis_size * sizeof(uint32_t);
    } else {
      ub_full_load_bytes = param_last_axis_bytes + vectorized_axis_bytes;
    }
    if (tmp_buf.GetSize() <
        ub_full_load_bytes) {  // tmpbuffer大于向量化轴空间，小于param尾轴空间+向量化轴空间，indices从UB取数据
      GatherUbNotFullLoad(dst, src0, src1, vectorized_axis_size, tmp_buf);
    } else {  // tmpbuffer大于param尾轴空间+向量化轴空间，param和indices都从UB取数据
      GatherUbFullLoad(dst, src0, src1, param_last_axis_size, vectorized_axis_size, param_last_axis_bytes,
                       vectorized_axis_bytes, tmp_buf);
    }
  }
}
#endif  // __ASCENDC_API_GATHER_H__