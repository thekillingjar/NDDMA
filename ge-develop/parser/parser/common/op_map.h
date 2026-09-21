/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_OP_MAP_H_
#define GE_COMMON_OP_MAP_H_

#include <map>
#include <string>
#include <vector>

/*lint -e1073*/
namespace ge {
// the operator type mapping table of caffe and  mindspore
extern std::map<std::string, std::string> caffe_op_map;

// the operator type mapping table of TensorFlow and  mindspore
extern std::map<std::string, std::string> tensorflow_op_map;

// the network training operator type mapping table of TensorFlow and  mindspore
extern std::map<std::string, std::string> tensorflow_train_op_map;

// local framework op vec
extern std::vector<std::string> local_framework_op_vec;

// dataset op vec
extern std::vector<std::string> is_dataset_op_vec;

// output tensor num
extern std::map<std::string, int32_t> op_output_tensor_num;
}  // namespace ge
/*lint +e1073*/
#endif  // GE_COMMON_OP_MAP_H_
