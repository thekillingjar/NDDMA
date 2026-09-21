# aclmdlConfigAttr<a name="ZH-CN_TOPIC_0000001312641433"></a>

```
typedef enum {
    ACL_MDL_PRIORITY_INT32 = 0,    
    ACL_MDL_LOAD_TYPE_SIZET,       
    ACL_MDL_PATH_PTR,              
    ACL_MDL_MEM_ADDR_PTR,          
    ACL_MDL_MEM_SIZET,             
    ACL_MDL_WEIGHT_ADDR_PTR,        
    ACL_MDL_WEIGHT_SIZET,          
    ACL_MDL_WORKSPACE_ADDR_PTR,     
    ACL_MDL_WORKSPACE_SIZET,        
    ACL_MDL_INPUTQ_NUM_SIZET,      
    ACL_MDL_INPUTQ_ADDR_PTR,        
    ACL_MDL_OUTPUTQ_NUM_SIZET,     
    ACL_MDL_OUTPUTQ_ADDR_PTR,      
    ACL_MDL_WORKSPACE_MEM_OPTIMIZE,
    ACL_MDL_WEIGHT_PATH_PTR,
    ACL_MDL_MODEL_DESC_PTR, 
    ACL_MDL_MODEL_DESC_SIZET,
    ACL_MDL_KERNEL_PTR, 
    ACL_MDL_KERNEL_SIZET,
    ACL_MDL_KERNEL_ARGS_PTR, 
    ACL_MDL_KERNEL_ARGS_SIZET,
    ACL_MDL_STATIC_TASK_PTR, 
    ACL_MDL_STATIC_TASK_SIZET,
    ACL_MDL_DYNAMIC_TASK_PTR,
    ACL_MDL_DYNAMIC_TASK_SIZET,
    ACL_MDL_MEM_MALLOC_POLICY_SIZET,
    ACL_MDL_FIFO_PTR,
    ACL_MDL_FIFO_SIZET,
} aclmdlConfigAttr;
```

**表 1**  模型加载选项配置

<a name="table154428882117"></a>
<table><thead align="left"><tr id="row15442178172115"><th class="cellrowborder" valign="top" width="35.96%" id="mcps1.2.3.1.1"><p id="p10442188112114"><a name="p10442188112114"></a><a name="p10442188112114"></a>选项</p>
</th>
<th class="cellrowborder" valign="top" width="64.03999999999999%" id="mcps1.2.3.1.2"><p id="p84426814219"><a name="p84426814219"></a><a name="p84426814219"></a>取值说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row134425813216"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1288720491044"><a name="p1288720491044"></a><a name="p1288720491044"></a>ACL_MDL_PRIORITY_INT32</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p1831812301993"><a name="p1831812301993"></a><a name="p1831812301993"></a>模型执行的优先级，可选项。该选项对应的值为int32类型。</p>
<p id="p128821749845"><a name="p128821749845"></a><a name="p128821749845"></a>数字越小优先级越高，取值[0,7]，默认值为0。</p>
<p id="p895335610"><a name="p895335610"></a><a name="p895335610"></a></p>
</td>
</tr>
<tr id="row444313892114"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p088324910415"><a name="p088324910415"></a><a name="p088324910415"></a>ACL_MDL_LOAD_TYPE_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p104431301464"><a name="p104431301464"></a><a name="p104431301464"></a>模型加载方式，必选项。该选项对应的值为size_t类型。</p>
<p id="p45342324387"><a name="p45342324387"></a><a name="p45342324387"></a>ACL_MDL_LOAD_TYPE_SIZET（表示模型加载方式）的取值如下：</p>
<a name="ul13551822113910"></a><a name="ul13551822113910"></a><ul id="ul13551822113910"><li>ACL_MDL_LOAD_FROM_FILE<pre class="screen" id="screen1266561714015"><a name="screen1266561714015"></a><a name="screen1266561714015"></a>#define ACL_MDL_LOAD_FROM_FILE 1</pre>
</li><li>ACL_MDL_LOAD_FROM_FILE_WITH_MEM<pre class="screen" id="screen13831721164019"><a name="screen13831721164019"></a><a name="screen13831721164019"></a>#define ACL_MDL_LOAD_FROM_FILE_WITH_MEM 2</pre>
</li><li>ACL_MDL_LOAD_FROM_MEM<pre class="screen" id="screen0165924134020"><a name="screen0165924134020"></a><a name="screen0165924134020"></a>#define ACL_MDL_LOAD_FROM_MEM 3</pre>
</li><li>ACL_MDL_LOAD_FROM_MEM_WITH_MEM<pre class="screen" id="screen6984142620403"><a name="screen6984142620403"></a><a name="screen6984142620403"></a>#define ACL_MDL_LOAD_FROM_MEM_WITH_MEM 4</pre>
</li><li>ACL_MDL_LOAD_FROM_FILE_WITH_Q<pre class="screen" id="screen17936330104010"><a name="screen17936330104010"></a><a name="screen17936330104010"></a>#define ACL_MDL_LOAD_FROM_FILE_WITH_Q 5</pre>
</li><li>ACL_MDL_LOAD_FROM_MEM_WITH_Q<pre class="screen" id="screen11997917182916"><a name="screen11997917182916"></a><a name="screen11997917182916"></a>#define ACL_MDL_LOAD_FROM_MEM_WITH_Q 6</pre>
</li></ul>
<p id="p064818284164"><a name="p064818284164"></a><a name="p064818284164"></a><strong id="b16653153191812"><a name="b16653153191812"></a><a name="b16653153191812"></a>注意</strong>：如果将ACL_MDL_LOAD_TYPE_SIZET设置为ACL_MDL_LOAD_FROM_MEM，表示从内存加载模型数据，还支持使用ACL_MDL_WEIGHT_PATH_PTR选项指定权重文件目录。</p>
</td>
</tr>
<tr id="row1144315862113"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p168821495411"><a name="p168821495411"></a><a name="p168821495411"></a>ACL_MDL_PATH_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p08791491418"><a name="p08791491418"></a><a name="p08791491418"></a>离线模型文件路径的指针，如果选择从文件加载模型，则该选项必选。</p>
</td>
</tr>
<tr id="row6443168192118"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p265494514414"><a name="p265494514414"></a><a name="p265494514414"></a>ACL_MDL_MEM_ADDR_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p465111451746"><a name="p465111451746"></a><a name="p465111451746"></a>模型在内存中的起始地址，如果选择从内存加载模型，则该选项必选。</p>
</td>
</tr>
<tr id="row74431810213"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p3482401347"><a name="p3482401347"></a><a name="p3482401347"></a>ACL_MDL_MEM_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p134654012413"><a name="p134654012413"></a><a name="p134654012413"></a>模型在内存中的大小，如果选择从内存加载模型，则该选项必选，与ACL_MDL_MEM_ADDR_PTR选项配合使用。该选项对应的值为size_t类型。</p>
</td>
</tr>
<tr id="row6443882211"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1943440840"><a name="p1943440840"></a><a name="p1943440840"></a>ACL_MDL_WEIGHT_ADDR_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p1638740142"><a name="p1638740142"></a><a name="p1638740142"></a>Device上模型权值内存（存放权值数据）的指针，如果需要由用户管理权值内存，则该选项必选。若不配置该选项，则表示由系统管理内存。</p>
</td>
</tr>
<tr id="row84434816215"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1682193518420"><a name="p1682193518420"></a><a name="p1682193518420"></a>ACL_MDL_WEIGHT_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p125042374410"><a name="p125042374410"></a><a name="p125042374410"></a>权值内存大小，单位为Byte，如果需要由用户管理权值内存，则该选项必选，与ACL_MDL_WEIGHT_ADDR_PTR选项配合使用。该选项对应的值为size_t类型。</p>
</td>
</tr>
<tr id="row18255111316223"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p177183518414"><a name="p177183518414"></a><a name="p177183518414"></a>ACL_MDL_WORKSPACE_ADDR_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p94991371047"><a name="p94991371047"></a><a name="p94991371047"></a>Device上模型所需工作内存（存放模型执行过程中的临时数据）的指针，如果需要由用户管理工作内存，则该选项必选。若不配置该选项，则表示由系统管理内存。</p>
</td>
</tr>
<tr id="row17511517112220"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1898853215413"><a name="p1898853215413"></a><a name="p1898853215413"></a>ACL_MDL_WORKSPACE_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p149868321446"><a name="p149868321446"></a><a name="p149868321446"></a>模型所需工作内存的大小，单位为Byte，如果需要由用户管理工作内存，则该选项必选，与ACL_MDL_WORKSPACE_ADDR_PTR选项配合使用。该选项对应的值为size_t类型。</p>
</td>
</tr>
<tr id="row104834214459"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1384415291543"><a name="p1384415291543"></a><a name="p1384415291543"></a>ACL_MDL_INPUTQ_NUM_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p5824327297"><a name="p5824327297"></a><a name="p5824327297"></a>模型输入队列大小 ,带队列加载模型时，该选项必选，与ACL_MDL_INPUTQ_ADDR_PTR选项配合使用。该选项对应的值为size_t类型。</p>
</td>
</tr>
<tr id="row14897131819218"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p18388291842"><a name="p18388291842"></a><a name="p18388291842"></a>ACL_MDL_INPUTQ_ADDR_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p983615299411"><a name="p983615299411"></a><a name="p983615299411"></a>模型输入队列ID的指针，带队列加载模型时，该选项必选，一个模型输入对应一个队列ID。</p>
</td>
</tr>
<tr id="row18907112215019"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p138331029944"><a name="p138331029944"></a><a name="p138331029944"></a>ACL_MDL_OUTPUTQ_NUM_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p4831129949"><a name="p4831129949"></a><a name="p4831129949"></a>模型输出队列大小，带队列加载模型时，该选项必选，与ACL_MDL_OUTPUTQ_ADDR_PTR选项配合使用。该选项对应的值为size_t类型。</p>
</td>
</tr>
<tr id="row4457134110107"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1445719413102"><a name="p1445719413102"></a><a name="p1445719413102"></a>ACL_MDL_OUTPUTQ_ADDR_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p16457241141011"><a name="p16457241141011"></a><a name="p16457241141011"></a>模型输出队列ID的指针，带队列加载模型时，该选项必选，一个模型输出对应一个队列ID。</p>
</td>
</tr>
<tr id="row184271543131015"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p15428043121017"><a name="p15428043121017"></a><a name="p15428043121017"></a>ACL_MDL_WORKSPACE_MEM_OPTIMIZE</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p20693122311110"><a name="p20693122311110"></a><a name="p20693122311110"></a>是否开启模型工作内存优化功能，1开启，0不开启。</p>
<p id="p16836031151117"><a name="p16836031151117"></a><a name="p16836031151117"></a>若关注内存规划或内存资源有限时，建议选择由系统管理内存的方式加载模型，并开启工作内存优化功能，此时工作内存中不包含存放模型输入、输出数据的内存，工作内存大小会减小，达到节省内存的目的。</p>
<p id="p18428144310109"><a name="p18428144310109"></a><a name="p18428144310109"></a>在模型执行前，还需要由用户申请存放模型输入、输出数据的内存，因此即使在模型加载时开启工作内存优化功能，也不会影响后续的模型执行。</p>
</td>
</tr>
<tr id="row1115116313523"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p10151193115526"><a name="p10151193115526"></a><a name="p10151193115526"></a>ACL_MDL_WEIGHT_PATH_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p286715841019"><a name="p286715841019"></a><a name="p286715841019"></a>权重文件所在目录的指针。对om模型文件大小有限制的场景下，通过本参数可实现权重文件外置功能。</p>
<p id="p8383834151415"><a name="p8383834151415"></a><a name="p8383834151415"></a>如果将ACL_MDL_LOAD_TYPE_SIZET设置为ACL_MDL_LOAD_FROM_MEM，表示从内存加载模型数据，则支持使用ACL_MDL_WEIGHT_PATH_PTR指定权重文件目录。</p>
<p id="p158091053131817"><a name="p158091053131817"></a><a name="p158091053131817"></a>一般对om模型文件大小有限制或模型文件加密的场景下，需单独加载权重文件，因此需在构建模型时，将权重保存在单独的文件中。例如在使用ATC工具生成om文件时，将--external_weight参数设置为1（1表示将原始网络中的Const/Constant节点的权重保存在单独的文件中）。</p>
<p id="p32616531691"><a name="p32616531691"></a><a name="p32616531691"></a></p>
<p id="p10826538790"><a name="p10826538790"></a><a name="p10826538790"></a><strong id="b1032212592096"><a name="b1032212592096"></a><a name="b1032212592096"></a>注意事项如下：</strong></p>
<p id="p1599434215917"><a name="p1599434215917"></a><a name="p1599434215917"></a>通过ACL_MDL_WEIGHT_PATH_PTR配置权重文件所在目录的指针后，在模型加载过程中，图引擎（Graph Engine，简称GE）将自行申请一块Device内存，读取外置权重文件，并将外置权重数据拷贝至该Device内存中，模型卸载时会释放该内存。若已调用<a href="aclmdlSetExternalWeightAddress.md">aclmdlSetExternalWeightAddress</a>接口配置存放外置权重的Device内存，由用户自行管理Device内存，则无需配置ACL_MDL_WEIGHT_PATH_PTR；若针对同一个模型，既配置了ACL_MDL_WEIGHT_PATH_PTR，也调用了<a href="aclmdlSetExternalWeightAddress.md">aclmdlSetExternalWeightAddress</a>接口，则<a href="aclmdlSetExternalWeightAddress.md">aclmdlSetExternalWeightAddress</a>接口的优先级更高。</p>
</td>
</tr>
<tr id="row1580140114118"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p68080124119"><a name="p68080124119"></a><a name="p68080124119"></a>ACL_MDL_MODEL_DESC_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p48291487416"><a name="p48291487416"></a><a name="p48291487416"></a>存放模型描述信息的内存指针。</p>
<p id="p660752111217"><a name="p660752111217"></a><a name="p660752111217"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row98637410415"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p38639414413"><a name="p38639414413"></a><a name="p38639414413"></a>ACL_MDL_MODEL_DESC_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p186901442143615"><a name="p186901442143615"></a><a name="p186901442143615"></a>存放模型描述信息所需的内存大小，单位Byte。该选项对应的值为size_t类型。</p>
<p id="p361917153415"><a name="p361917153415"></a><a name="p361917153415"></a>可提前调用<a href="aclmdlQueryExeOMDesc.md">aclmdlQueryExeOMDesc</a>接口获取存放模型描述信息所需的内存大小，且本选项需与ACL_MDL_MODEL_DESC_PTR选项配合使用。</p>
<p id="p57101844125118"><a name="p57101844125118"></a><a name="p57101844125118"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row19829589419"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p082915814116"><a name="p082915814116"></a><a name="p082915814116"></a>ACL_MDL_KERNEL_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p125185415370"><a name="p125185415370"></a><a name="p125185415370"></a>存放TBE算子kernel（算子的*.o与*.json）的内存指针。</p>
<p id="p101688489517"><a name="p101688489517"></a><a name="p101688489517"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row1619191510414"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p6619141554120"><a name="p6619141554120"></a><a name="p6619141554120"></a>ACL_MDL_KERNEL_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p10246540377"><a name="p10246540377"></a><a name="p10246540377"></a>存放TBE算子kernel（算子的*.o与*.json）所需的内存大小，单位Byte。该选项对应的值为size_t类型。</p>
<p id="p141336167715"><a name="p141336167715"></a><a name="p141336167715"></a>可提前调用<a href="aclmdlQueryExeOMDesc.md">aclmdlQueryExeOMDesc</a>接口获取存放TBE算子kernel（算子的*.o与*.json）所需的内存大小，且本选项需与ACL_MDL_KERNEL_PTR选项配合使用。</p>
<p id="p17223155215513"><a name="p17223155215513"></a><a name="p17223155215513"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row314761215415"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p15147212104119"><a name="p15147212104119"></a><a name="p15147212104119"></a>ACL_MDL_KERNEL_ARGS_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p125009521777"><a name="p125009521777"></a><a name="p125009521777"></a>存放TBE算子kernel参数的内存指针。</p>
<p id="p148791255125112"><a name="p148791255125112"></a><a name="p148791255125112"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row876631354117"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p476611312416"><a name="p476611312416"></a><a name="p476611312416"></a>ACL_MDL_KERNEL_ARGS_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p2500652870"><a name="p2500652870"></a><a name="p2500652870"></a>存放TBE算子kernel参数所需的内存大小，单位Byte。该选项对应的值为size_t类型。</p>
<p id="p19500552570"><a name="p19500552570"></a><a name="p19500552570"></a>可提前调用<a href="aclmdlQueryExeOMDesc.md">aclmdlQueryExeOMDesc</a>接口获取存放TBE算子kernel参数所需的内存大小，且本选项需与ACL_MDL_KERNEL_ARGS_PTR选项配合使用。</p>
<p id="p163611459145120"><a name="p163611459145120"></a><a name="p163611459145120"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row117378694119"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p127371560411"><a name="p127371560411"></a><a name="p127371560411"></a>ACL_MDL_STATIC_TASK_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p873719694114"><a name="p873719694114"></a><a name="p873719694114"></a>存放静态shape任务描述信息的内存指针。</p>
<p id="p16581926520"><a name="p16581926520"></a><a name="p16581926520"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row8617181018418"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p2617161020413"><a name="p2617161020413"></a><a name="p2617161020413"></a>ACL_MDL_STATIC_TASK_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p1461741017411"><a name="p1461741017411"></a><a name="p1461741017411"></a>存放静态shape任务描述信息所需的内存大小，单位Byte。该选项对应的值为size_t类型。</p>
<p id="p12979152510918"><a name="p12979152510918"></a><a name="p12979152510918"></a>可提前调用<a href="aclmdlQueryExeOMDesc.md">aclmdlQueryExeOMDesc</a>接口获取存放静态shape任务描述信息所需的内存大小，且本选项需与ACL_MDL_STATIC_TASK_PTR选项配合使用。</p>
<p id="p13832116115214"><a name="p13832116115214"></a><a name="p13832116115214"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row13460111294316"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p1946117125433"><a name="p1946117125433"></a><a name="p1946117125433"></a>ACL_MDL_DYNAMIC_TASK_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p00643296"><a name="p00643296"></a><a name="p00643296"></a>存放动态shape任务描述信息的内存指针。</p>
<p id="p12721149135213"><a name="p12721149135213"></a><a name="p12721149135213"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row16648131517437"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p7648181594317"><a name="p7648181594317"></a><a name="p7648181594317"></a>ACL_MDL_DYNAMIC_TASK_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p19014316920"><a name="p19014316920"></a><a name="p19014316920"></a>存放动态shape任务描述信息所需的内存大小，单位Byte。该选项对应的值为size_t类型。</p>
<p id="p12044310916"><a name="p12044310916"></a><a name="p12044310916"></a>可提前调用<a href="aclmdlQueryExeOMDesc.md">aclmdlQueryExeOMDesc</a>接口获取存放动态shape任务描述信息所需的内存大小，且本选项需与ACL_MDL_DYNAMIC_TASK_PTR选项配合使用。</p>
<p id="p5570161315520"><a name="p5570161315520"></a><a name="p5570161315520"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row29391413132416"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p12939151302412"><a name="p12939151302412"></a><a name="p12939151302412"></a>ACL_MDL_MEM_MALLOC_POLICY_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p09395139248"><a name="p09395139248"></a><a name="p09395139248"></a>内存分配规则，该选项对应的值为size_t类型。</p>
<p id="p85032018135213"><a name="p85032018135213"></a><a name="p85032018135213"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row64611711183120"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p7461211173112"><a name="p7461211173112"></a><a name="p7461211173112"></a>ACL_MDL_FIFO_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p8461111113112"><a name="p8461111113112"></a><a name="p8461111113112"></a>模型级别全局内存的起始地址。此处是指Device上的内存。</p>
<p id="p18393727145214"><a name="p18393727145214"></a><a name="p18393727145214"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row4737738143114"><td class="cellrowborder" valign="top" width="35.96%" headers="mcps1.2.3.1.1 "><p id="p673713817311"><a name="p673713817311"></a><a name="p673713817311"></a>ACL_MDL_FIFO_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="64.03999999999999%" headers="mcps1.2.3.1.2 "><p id="p0737738113111"><a name="p0737738113111"></a><a name="p0737738113111"></a>模型级别全局内存的大小，该选项对应的值为size_t类型。</p>
<p id="p19535311526"><a name="p19535311526"></a><a name="p19535311526"></a>当前版本不支持该配置。</p>
</td>
</tr>
</tbody>
</table>

关于如何获取om文件，请参见《ATC离线模型编译工具用户指南》

