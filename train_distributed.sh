#!/bin/bash

# 分布式花卉分类训练脚本
# 注意：这是一个Unix/Linux环境的shell脚本，如果您在Windows上运行，
# 请参考下方的Windows批处理脚本替代方案

# 设置中文字符集，确保输出正常显示中文
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 帮助信息函数
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --config <配置文件路径>    指定配置文件路径，默认为 'configs/config_Distributed.toml'"
    echo "  --world-size <进程数>      指定分布式训练的进程总数，默认为可用GPU数量"
    echo "  --help                    显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                         使用默认配置和GPU数量运行"
    echo "  $0 --config configs/my_config.toml --world-size 8   使用自定义配置和8个进程运行"
}

# 默认参数
CONFIG_FILE="configs/config_Distributed.toml"
WORLD_SIZE="8"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 构建命令
CMD="python train_distributed.py --config $CONFIG_FILE"

# 如果指定了进程数，则添加到命令中
if [ -n "$WORLD_SIZE" ]; then
    CMD="$CMD --world-size $WORLD_SIZE"
fi

# 显示将要执行的命令
 echo "执行命令: $CMD"

# 执行训练命令
exec $CMD

# Windows批处理脚本替代方案（将以下内容保存为train_distributed.bat）
# @echo off
# chcp 65001 >nul
#
# set CONFIG_FILE=configs\config_Distributed.toml
# set WORLD_SIZE=
#
# :parse_args
# if "%1" == "--config" (
#     set CONFIG_FILE=%2
#     shift
#     shift
#     goto parse_args
# ) else if "%1" == "--world-size" (
#     set WORLD_SIZE=%2
#     shift
#     shift
#     goto parse_args
# ) else if "%1" == "--help" (
#     echo 用法: %0 [选项]
#     echo
#     echo 选项:
#     echo   --config ^<配置文件路径^>    指定配置文件路径，默认为 'configs\config_Distributed.toml'
#     echo   --world-size ^<进程数^>      指定分布式训练的进程总数，默认为可用GPU数量
#     echo   --help                    显示帮助信息
#     echo
#     echo 示例:
#     echo   %0                         使用默认配置和GPU数量运行
#     echo   %0 --config configs\my_config.toml --world-size 4   使用自定义配置和4个进程运行
#     exit /b 0
# ) else if not "%1" == "" (
#     echo 未知选项: %1
#     exit /b 1
# )
#
# if not exist "%CONFIG_FILE%" (
#     echo 错误: 配置文件不存在: %CONFIG_FILE%
#     exit /b 1
# )
#
# set CMD=python train_distributed.py --config "%CONFIG_FILE%"
#
# if not "%WORLD_SIZE%" == "" (
#     set CMD=%CMD% --world-size %WORLD_SIZE%
# )
#
# echo 执行命令: %CMD%
#
# %CMD%
