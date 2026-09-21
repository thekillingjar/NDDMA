/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_UTILS_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_UTILS_H_

#include "exe_graph/runtime/tensor_data.h"
#include "graph/types.h"

namespace gert {
namespace {
struct PlacementBase {
  virtual ~PlacementBase() = default;
};
struct PlacementDeviceHbm : public PlacementBase {
  ~PlacementDeviceHbm() override = default;
};
struct PlacementDeviceP2p : public PlacementDeviceHbm {
  ~PlacementDeviceP2p() override = default;
};
struct PlacementHost : public PlacementBase {
  ~PlacementHost() override = default;
};

class PlacementClassFactory {
 public:
  const PlacementBase *Get(const TensorPlacement placement) const {
    switch (placement) {
      case kOnDeviceHbm:
        return &hbm_;
      case kOnDeviceP2p:
        return &p2p_;
      case kOnHost:
      case kFollowing:
        return &host_;
      case kTensorPlacementEnd:
        return &base_;
      default:
        return &base_;
    }
  }
  bool CanSrcDynamicCastToDst(const TensorPlacement src, const TensorPlacement dst) const {
    const auto *src_ptr = Get(src);
    bool cast_success;
    switch (dst) {
      case kOnDeviceHbm:
        cast_success = (dynamic_cast<const PlacementDeviceHbm *>(src_ptr) != nullptr);
        break;
      case kOnDeviceP2p:
        cast_success = (dynamic_cast<const PlacementDeviceP2p *>(src_ptr) != nullptr);
        break;
      case kOnHost:
      case kFollowing:
        cast_success = (dynamic_cast<const PlacementHost *>(src_ptr) != nullptr);
        break;
      case kTensorPlacementEnd:
        cast_success = (dynamic_cast<const PlacementBase *>(src_ptr) != nullptr);
        break;
      default:
        cast_success = (dynamic_cast<const PlacementBase *>(src_ptr) != nullptr);
        break;
    }
    return cast_success;
  }

 private:
   PlacementDeviceHbm hbm_;
   PlacementDeviceP2p p2p_;
   PlacementHost host_;
   PlacementBase base_;
};
}

inline const ge::char_t *GetPlacementStr(const TensorPlacement placement) {
  static const ge::char_t *placement_str[static_cast<int32_t>(kTensorPlacementEnd) + 1] = {"DeviceHbm", "HostDDR",
                                                                                           "HostDDR", "DeviceP2p",
                                                                                           "Unknown"};
  if ((placement >= kTensorPlacementEnd) || (placement < kOnDeviceHbm)) {
    return placement_str[kTensorPlacementEnd];
  }
  return placement_str[placement];
}

/**
 * 判断源placement到目的placement是否需要拷贝
 * @param src_placement 源placement
 * @param dst_placement 目的placement
 */
inline bool IsPlacementSrcToDstNeedCopy(const TensorPlacement src_placement, const TensorPlacement dst_placement) {
  if ((src_placement >= kTensorPlacementEnd) || (dst_placement >= kTensorPlacementEnd)) {
    return true;
  }

  static PlacementClassFactory factory;
  const auto *dst_class_ptr = factory.Get(dst_placement);
  const auto *src_class_ptr = factory.Get(src_placement);
  if (dst_class_ptr == src_class_ptr) {
    return false;
  }

  return !factory.CanSrcDynamicCastToDst(src_placement, dst_placement);
}
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_DATA_UTILS_H_
