# 命名

## 1.文件命名

以特性粒度来拆分文件并命名

​    OK: test_ctrl_flow_infershape.cc

​		表示该组用例主要测试控制流infershape相关特性。

​    **NOK: test_graph_compiler.cc**

​		不明确所测试的特性。

## 2.用例命名

用例名称需明确测试点

​	OK:    TEST_F(ConstantFoldingTest, test_flattenv2_folding_success) 

​		表示该用例测试flattenv2的常量折叠功能的正常场景

​	**NOK:  TEST_F(FftsPlusTest, ffts_plus_graph)** 

​		不明确所测试的功能点。

# 校验点

## 1.用例需要设置校验点，不允许存在空跑流程的用例

`// build_graph through session`

 `ret = session.BuildGraph(2, inputs);`

 `EXPECT_EQ(ret, SUCCESS);`

只校验了接口返回成功，除此之外没有其他校验点。基本没有守护功能的作用。

## 2.用例需要为端到端入口用例，不允许将UT复制过来

