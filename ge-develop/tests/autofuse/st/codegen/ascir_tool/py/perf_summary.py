#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import glob
import pandas as pd

# 定义路径模式
path_pattern = './out/PROF_*/mindstudio_profiler_output/op_summary*.csv'

# 查找所有匹配的csv文件
csv_files = glob.glob(path_pattern)

# 初始化一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历所有找到的csv文件
for csv_file in csv_files:
    # 读取csv文件
    df = pd.read_csv(csv_file)

    # 打印Task Duration(us)字段的值
    print("#################Performance Result#####################")
    if 'Task Duration(us)' in df.columns:
        #print(f"Task Duration(us) from {csv_file}:")
        for value in df['Task Duration(us)'].values:
            print('Task Duration(us): ', value)
    
    # 将当前文件的数据追加到all_data中
    all_data = pd.concat([all_data, df], ignore_index=True)

# 检查当前目录下是否存在perf_result.csv
output_file = './out/perf_result.csv'
if os.path.exists(output_file):
    # 如果存在，读取现有数据并追加新数据
    existing_data = pd.read_csv(output_file)
    combined_data = pd.concat([existing_data, all_data], ignore_index=True)
    combined_data.to_csv(output_file, index=False)
else:
    # 如果不存在，创建新的perf_result.csv并写入数据
    all_data.to_csv(output_file, index=False)

print(f"Detailed performance results has been written to out/perf_result.csv")
