/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
using std::vector;
using std::string;

#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("hidden")))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("hidden")))

#ifndef __STUB_ADAPTER
#define __STUB_ADAPTER

namespace ge {


enum Format {
  FORMAT_NCHW = 0,   // NCHW
  FORMAT_NHWC,       // NHWC
  FORMAT_ND,         // Nd Tensor
  FORMAT_NC1HWC0,    // NC1HWC0
  FORMAT_FRACTAL_Z,  // FRACTAL_Z
  FORMAT_NC1C0HWPAD = 5,
  FORMAT_NHWC1C0,
  FORMAT_FSR_NCHW,
  FORMAT_FRACTAL_DECONV,
  FORMAT_C1HWNC0,
  FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
  FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
  FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 is 4
  FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 is 4
  FORMAT_CHWN,
  FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
  FORMAT_HWCN,
  FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  FORMAT_BN_WEIGHT,
  FORMAT_FILTER_HWCK,  // filter input tensor format
  FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
  FORMAT_HASHTABLE_LOOKUP_KEYS,
  FORMAT_HASHTABLE_LOOKUP_VALUE,
  FORMAT_HASHTABLE_LOOKUP_OUTPUT,
  FORMAT_HASHTABLE_LOOKUP_HITS,
  FORMAT_C1HWNCoC0 = 25,
  FORMAT_MD,
  FORMAT_NDHWC,
  FORMAT_FRACTAL_ZZ,
  FORMAT_FRACTAL_NZ,
  FORMAT_NCDHW = 30,
  FORMAT_DHWCN,  // 3D filter input tensor format
  FORMAT_NDC1HWC0,
  FORMAT_FRACTAL_Z_3D,
  FORMAT_CN,
  FORMAT_NC = 35,
  FORMAT_DHWNC,
  FORMAT_FRACTAL_Z_3D_TRANSPOSE, // 3D filter(transpose) input tensor format
  FORMAT_FRACTAL_ZN_LSTM,
  FORMAT_FRACTAL_Z_G,
  FORMAT_RESERVED = 40,
  FORMAT_ALL,
  FORMAT_NULL,
  FORMAT_ND_RNN_BIAS,
  FORMAT_FRACTAL_ZN_RNN,
  FORMAT_NYUV,
  FORMAT_NYUV_A,
  // Add new formats definition here
  NOT_SUPPORT,
  FORMAT_END,
  // FORMAT_MAX defines the max value of Format.
  // Any Format should not exceed the value of FORMAT_MAX.
  // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
  //                   If you want to judge the range of Format, you can use FORMAT_END.
  FORMAT_MAX = 0xff
};

enum DataType {
  DT_FLOAT = 0,            // float type
  DT_FLOAT16 = 1,          // fp16 type
  DT_INT8 = 2,             // int8 type
  DT_INT16 = 6,            // int16 type
  DT_UINT16 = 7,           // uint16 type
  DT_UINT8 = 4,            // uint8 type
  DT_INT32 = 3,            //
  DT_INT64 = 9,            // int64 type
  DT_UINT32 = 8,           // unsigned int32
  DT_UINT64 = 10,          // unsigned int64
  DT_DOUBLE = 11,
  DT_STRING = 13,          // string type
  DT_BF16 = 27,            // bf16 type
  DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
  DT_MAX                   // Mark the boundaries of data types
};

enum YUVSubFormat {
  YUV420_SP = 0,            // float type
  YVU420_SP = 1,          // fp16 type
  YUV422_SP = 2,             // int8 type
  YVU422_SP = 3,            // int16 type
  YUV440_SP = 4,           // uint16 type
  YVU440_SP = 5,            // uint8 type
  YUV444_SP = 6,            //
  YVU444_SP = 7,            // int64 type
  YUV444_PACKED = 8,           // unsigned int32
  YVU444_PACKED = 9,          // unsigned int64
  YUYV422_PACKED = 10,          // string type
  YVYU422_PACKED = 11,            // bf16 type
  YUV400 = 12,       // Used to indicate a DataType field has not been set.
  YUV_BOTTEM                   // Mark the boundaries of data types
};

inline int32_t GetPrimaryFormat(int32_t format) {
  return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xff);
}

inline int32_t GetSubFormat(int32_t format) {
  return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00) >> 8);
}
class GeShape {
public:
    GeShape() = default;
    GeShape(vector<uint32_t>);
    uint32_t GetDim (uint32_t);
    uint32_t GetDimNum() const { return dimNum_; }
protected:
    vector<uint32_t> shape;
    uint32_t dimNum_;
};

class GeTensorDesc {
public:
    GeTensorDesc(vector<uint32_t>, Format);
    Format GetFormat() const;
    GeShape GetOriginShape() const;
    GeShape GetShape() const;
protected:
    GeShape shape;
    Format format;
};

class OpDesc : public std::enable_shared_from_this<OpDesc> {
public:
    OpDesc() = default;
    void SetType(string);
    string GetType();
    void AddInputDesc(GeTensorDesc);
    void AddOutputDesc(GeTensorDesc);
    GeTensorDesc GetInputDesc(uint32_t);
    GeTensorDesc GetOutputDesc(uint32_t);
    uint32_t GetAllInputsSize();
protected:
    vector<GeTensorDesc> inputs, outputs;
    string type_;
};
using OpDescPtr = std::shared_ptr<OpDesc>;
}


// namespace ge {

// class GeShapeImpl {
//     using DimsType = SmallVector<int64_t, kDefaultDimsNum>;
// public:
//     GeShapeImpl() = default;
//     explicit GeShapeImpl(const std::vector<int64_t> &dims);
//     int64_t GetDim(const size_t idx) const;

// private:
//     DimsType dims_;
//     friend class GeTensorDesc;
// };
// using GeShapeImplPtr = std::shared_ptr<GeShapeImpl>;

// class TensorDataImpl;

// }

#endif