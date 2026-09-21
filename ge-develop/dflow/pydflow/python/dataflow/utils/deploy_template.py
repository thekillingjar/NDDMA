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

import json
import dataflow.dataflow as df


def generate_deploy_template(graph, file_path):
    # extract input FlowData
    added_nodes = set()
    used_nodes = []
    input_datas = set()
    batch_deploy_info = []

    for _, output in enumerate(graph._outputs):
        used_nodes.append(output.node)

    while len(used_nodes) > 0:
        node = used_nodes.pop()
        if node.name in added_nodes:
            # skip this node
            continue
        added_nodes.add(node.name)
        deploy_node_name = node.name
        if node.alias is not None:
            deploy_node_name = node.alias
        batch_deploy_info.append(
            {"flow_node_list": [deploy_node_name], "logic_device_list": "0:0:0:0"}
        )
        for anchor in node._input_anchors:
            if not isinstance(anchor, df.FlowData):
                used_nodes.append(anchor.node)

    data_flow_deploy_info = {"batch_deploy_info": batch_deploy_info}
    with open(file_path, "w") as f:
        # use indent 4
        json.dump(data_flow_deploy_info, f, indent=4)
    print("generate deploy template success, file:", file_path)
