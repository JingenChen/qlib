#!/bin/bash

# ======================================================
# Qlib 数据自动下载脚本 (增强版)
# ======================================================
TARGET_DIR="$HOME/.qlib/qlib_data/cn_data"
TMP_FILE="qlib_bin.tar.gz"
DOWNLOAD_URL="https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"

# 错误处理：一旦出错立即退出
set -e

echo "======================================================"
echo "开始自动下载并解压最新的 qlib_bin.tar.gz 数据"
echo "目标目录: ${TARGET_DIR}"
echo "======================================================"

# 0. 检查依赖
HAS_CURL=false
HAS_WGET=false
if command -v curl &> /dev/null; then HAS_CURL=true; fi
if command -v wget &> /dev/null; then HAS_WGET=true; fi

if [ "$HAS_CURL" = false ] && [ "$HAS_WGET" = false ]; then
    echo "❌ 错误: 未找到 'curl' 或 'wget' 命令，请先安装其中之一。"
    exit 1
fi

if ! command -v tar &> /dev/null; then
    echo "❌ 错误: 未找到 'tar' 命令，请先安装。"
    exit 1
fi

# 1. 创建并检查目标文件夹
if [ -d "${TARGET_DIR}" ] && [ "$(ls -A "${TARGET_DIR}")" ]; then
    echo "⚠️ 警告: 目标目录 '${TARGET_DIR}' 已存在且不为空。"
    echo "   数据将被合并或覆盖。"
fi
mkdir -p "${TARGET_DIR}"

# 2. 清理机制：无论脚本成功还是失败，都会尝试清理临时文件
cleanup() {
    if [ -f "${TMP_FILE}" ]; then
        echo "正在清理临时文件..."
        rm -f "${TMP_FILE}"
    fi
}
trap cleanup EXIT

# 3. 下载数据压缩包
echo "[2/3] 正在下载数据文件..."
if [ "$HAS_CURL" = true ]; then
    # -L: 跟随重定向; -f: HTTP 错误时失败; --retry: 重试
    curl -L -f --retry 3 -o "${TMP_FILE}" "${DOWNLOAD_URL}"
else
    # wget 备选
    wget -O "${TMP_FILE}" "${DOWNLOAD_URL}"
fi

# 验证文件类型 (防止下载了 HTML 错误页面)
if command -v file &> /dev/null; then
    if ! file "${TMP_FILE}" | grep -q "gzip compressed data"; then
        echo "❌ 错误: 下载的文件不是有效的 gzip 压缩包。内容可能是错误提示："
        head -n 5 "${TMP_FILE}"
        exit 1
    fi
fi

# 4. 解压缩
echo "[3/3] 下载完成，正在解压到 ${TARGET_DIR}..."
# --strip-components=1 用于移除压缩包内的根目录层级 (qlib_bin/)
tar -zxf "${TMP_FILE}" -C "${TARGET_DIR}" --strip-components=1

echo "✅ 数据解压成功！"
echo "🎉 数据已保存在: ${TARGET_DIR}"

# 提示用户编译 Qlib 扩展 (如果需要)
echo "------------------------------------------------------"
echo "提示: 如果运行 Qlib 时遇到 'ModuleNotFoundError', 请确保已编译扩展:"
echo "      python setup.py build_ext --inplace"
echo "------------------------------------------------------"
