/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#define private public
#include "../common/code_printer.h"
namespace att
{
    class CodeprinterTest: public testing::Test {
    public:
        // 前处理：创建一个测试用的空文件
        void SetUp() override {
            // std::string filename = "testfile.cpp";  //在build路径中生成
            // std::ofstream outfile;
            // outfile.open(filename, std::ios::trunc);
            // if (outfile.is_open()){
            //     std::cout << "文件创建成功：" << filename << std::endl;
            //     outfile.close();
            // }
            // else{
            //     std::cerr << "无法创建文件：" << filename << std::endl;
            // }
        }
        // 后处理：删除测试文件
        void TearDown() override {
            // std::string filename = "testfile.cpp";
            // if (std::remove(filename.c_str()) == 0){
            //     std::cout << "文件删除成功：" << filename << std::endl;
            // }
            // else{
            //     std::cerr << "无法删除文件：" << filename <<std::endl;
            // }
        }
    };

    TEST_F(CodeprinterTest, AddIncludeTest){
        const std::string content = "#include \"stdio\"";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddInclude("stdio");
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddNamespaceBeginTest){
        const std::string content = "namespace std {";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddNamespaceBegin("std");
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddNamespaceEndTest){
        const std::string content = "} // namespace std";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddNamespaceEnd("std");
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddClassBeginTest){
        const std::string content = "class ge::CodePrinter {";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.DefineClassBegin("ge::CodePrinter");
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddClassEndTest){
        const std::string content = "};";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.DefineClassEnd();
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddStructBeginTest){
        const std::string content = "struct Student {";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddStructBegin("Student");
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddStructEndTest){
        const std::string content = "};";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddStructEnd();
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddFuncBeginTest){
        const std::string content = "int main()\n{";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.DefineFuncBegin("int", "main", "");
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddFuncEndTest){
        const std::string content = "}";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.DefineFuncEnd();
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, AddLineTest){
        //要添加的字符串
        const std::string content = "Hello, world!";
        //待调用的测试函数
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddLine("Hello, world!");
        //读取文件内容并判断是否和添加内容一致
        auto file_content = CodePrinter1.GetOutputStr();
        file_content.pop_back();
        EXPECT_EQ(content, file_content);
    }

    TEST_F(CodeprinterTest, resetTest){
        const std::string content = "Hello, world!";
        ge::CodePrinter CodePrinter1;
        CodePrinter1.AddLine("Hello, world!");
        auto file_content = CodePrinter1.GetOutputStr();
        EXPECT_FALSE(file_content.empty());
        CodePrinter1.Reset();
        file_content = CodePrinter1.GetOutputStr();
        EXPECT_TRUE(file_content.empty());
    }

} //namespace