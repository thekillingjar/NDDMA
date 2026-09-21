/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_COPY_FLOW_LAUNCH_FUSE_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_COPY_FLOW_LAUNCH_FUSE_H_
#include "pass.h"
namespace gert {
namespace bg {
/*
 * before HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *           \            ...  ...
 *            \          LaunchKernelWithHandle1
 *             \          /                \
 *          MakeSureTensorAtDevice  MakeSureTensorAtDevice
 *                      |                          |
 *         CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 *  after HostInputsProcFuse pass:
 *             netoutput
 *               /
 *              /
 *  LaunchKernelWithHandle2
 *           \            ...  ...
 *            \          LaunchKernelWithHandle1
 *             \          /                \
 *          MakeSureTensorAtDevice       CopyFlowLaunch
 *                      |                          |
 *           CalTensorSizeFromStorage  CalTensorSizeFromStorage
 *                      |                          |
 *                 SplitTensor                SplitTensor
 *                      |                          |
 *                    data1                      data2
 *
 */
class CopyFlowLaunchFuse : public Pass {
 public:
  ge::graphStatus Run(ge::ExecuteGraph *const graph, bool &changed) override;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_COPY_FLOW_LAUNCH_FUSE_H_
