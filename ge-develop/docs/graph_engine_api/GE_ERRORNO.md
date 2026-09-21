# GE\_ERRORNO<a name="ZH-CN_TOPIC_0000001264926018"></a>

-   **GE\_ERRORNO**

    错误码及描述注册宏，该宏对外提供了SUCCESS和FAILED两个错误，供用户使用：

    GE\_ERRORNO\(0, 0, 0, 0, 0, SUCCESS, 0, "success"\);

    GE\_ERRORNO\(0b11, 0b11, 0b111, 0xFF, 0b11111, FAILED, 0xFFF, "failed"\);

-   **GE\_ERRORNO\_DEFINE**

    错误码定义宏，该宏对外提供了END\_OF\_SEQUENCE错误，供用户使用：

    GE\_ERRORNO\_DEFINE\(0b01, 0b01, 0b000, 8, 0, END\_OF\_SEQUENCE, 7\);

-   **GE\_ERRORNO\_EXTERNAL**

    错误码及其错误描述信息宏，name是uint32\_t类型，desc是std::string类型：

    GE\_ERRORNO\_EXTERNAL\(ACL\_ERROR\_GE\_PARAM\_INVALID, "Parameter invalid."\);

