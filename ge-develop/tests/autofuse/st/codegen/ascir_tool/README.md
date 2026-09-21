# ascir_tool

- **配置环境变量**
  source env.sh

- **执行**
* mode=0
  单kernel端到端执行，包括根据构图脚本input_ascir.py进行代码生成、编译，根据ascir.json进行launch kernel，并根据gen_input.py生成输入和gold数据，并校验kernel产生的输出与gold对比
  （1）在testcase目录下（或其他目录）创建用例目录，分别创建input_ascir.py， ascir.json， gen_input.py

  （2）执行 bash test_ascir.sh --mode=0 --case=testcase目录下你的用例名（--path=用例路径）
      如需跑profiling，执行 bash test_ascir.sh 你的用例名 --prof

   (3) 验证好的用例，可以提交上库

* mode=1
  根据已有host和device代码编译、执行并对比结果
  （1）该场景使用config路径下的配置信息ascir.json和input路径下的gen_input.py

  （2）执行 bash test_ascir.sh --mode=1 --path=/kernel_meta/build/
      其中--path参数为kernel路径，路径场景为自动融合开启debug功能时，生成kernel的build路径，build路径下分别有"host"和"device"文件夹，分别保存tiling代码和kernel代码

* mode=2
  根据已有host和device代码仅做编译
  （1）根据device的kernel代码，执行编译

  （2）执行 bash test_ascir.sh --mode=2 --path=/kernel_meta/build/