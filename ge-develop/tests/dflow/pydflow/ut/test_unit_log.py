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
import unittest
import dataflow.utils.log as df_log


class TestLog(unittest.TestCase):
    def print_log(self):
        df_log.debug("debug")
        df_log.info("info")
        df_log.warning("warning")
        df_log.error("error")
        df_log.log(df_log.INFO, "info")

    def test_set_log_level_debug(self):
        df_log.set_log_level(df_log.INFO)
        logger = df_log.get_logger()
        self.assertEqual(df_log.get_log_level(), df_log.INFO)

    def test_log(self):
        df_log.set_log_level(df_log.DEBUG)
        logger = df_log.get_logger()
        with self.assertLogs(logger, df_log.DEBUG) as log_msg:
            self.print_log()
        self.assertEqual(
            log_msg.output,
            [
                "DEBUG:DATAFLOW:debug",
                "INFO:DATAFLOW:info",
                "WARNING:DATAFLOW:warning",
                "ERROR:DATAFLOW:error",
                "INFO:DATAFLOW:info",
            ],
        )
