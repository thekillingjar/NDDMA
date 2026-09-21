# 关于A、B、C矩阵的Shape及内存大小计算公式<a name="ZH-CN_TOPIC_0000001264922034"></a>

-   A、B、C矩阵是否转置的标记如果设置为ACL\_TRANS\_N或ACL\_TRANS\_T，则用户在申请内存存放A、B、C矩阵数据时，实际申请的内存要和实际数据大小匹配，Shape及内存大小的计算公式如下：
    -   A矩阵：shape = \(m, k\)；内存大小 = m\*k\*sizeof\(dataTypeA\)
    -   B矩阵：shape = \(k, n\)；内存大小 = k\*n\*sizeof\(dataTypeB\)
    -   C矩阵：shape = \(m, n\)；内存大小 = m\*n\*sizeof\(dataTypeC\)

-   A、B、C矩阵是否转置的标记如果设置为ACL\_TRANS\_NZ，表示采用内部数据格式，矩阵Shape为4维，Shape及内存大小的计算公式如下（假设m,k,n分别为原始轴，⌈ ⌉表示向上对齐）：
    -   当矩阵A和矩阵B中数据的类型为aclFloat16，在计算实际内存大小时，m、k、n均按16对齐向上取整计算：
        -   A矩阵：shape = \(⌈k/16⌉, ⌈m/16⌉, 16, 16\)；内存大小 = ⌈m/16⌉\*16\*⌈k/16⌉\*16\*sizeof\(dataTypeA\)
        -   B矩阵：shape = \(⌈n/16⌉, ⌈k/16⌉, 16, 16\)；内存大小 = ⌈k/16⌉\*16\*⌈n/16⌉\*16\*sizeof\(dataTypeB\)
        -   C矩阵：shape = \(⌈n/16⌉, ⌈m/16⌉, 16, 16\)；内存大小 = ⌈m/16⌉\*16\*⌈n/16⌉\*16\*sizeof\(dataTypeC\)

    -   当矩阵A和矩阵B中数据的类型为int8\_t，在计算实际内存大小时，reduce轴按32对齐向上取整计算，非reduce轴按16对齐向上取整计算：
        -   A矩阵：shape = \(⌈k/32⌉, ⌈m/16⌉, 16, 32\)；内存大小 = ⌈m/16⌉\*16\*⌈k/32⌉\*32\*sizeof\(dataTypeA\)
        -   B矩阵：shape = \(⌈k/32⌉, ⌈n/16⌉, 32, 16\)；内存大小 = ⌈k/32⌉\*32\*⌈n/16⌉\*16\*sizeof\(dataTypeB\)
        -   C矩阵：shape = \(⌈n/16⌉, ⌈m/16⌉, 16, 16\)；内存大小 = ⌈m/16⌉\*16\*⌈n/16⌉\*16\*sizeof\(dataTypeC\)

