/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "default_reg_func_v2.h"

namespace ge {
namespace ascir {

std::vector<std::unique_ptr<ge::TmpBufDesc>> GetCompareSizeV2([[maybe_unused]]const ge::AscNode &node) {
    Expression TmpSize = ge::Symbol(32);
    ge::TmpBufDesc desc = {TmpSize, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
    tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmpBufDescs;
}

}  // namespace ascir
}  // namespace ge