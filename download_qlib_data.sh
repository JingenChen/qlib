#!/bin/bash

TARGET_DIR="./qlib_data/cn_data"
TMP_FILE="qlib_bin.tar.gz"
DOWNLOAD_URL="https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"

echo "======================================================"
echo "开始自动下载并解压最新的 qlib_bin.tar.gz 数据"
echo "======================================================"

# 0. 检查依赖
for cmd in curl tar; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 错误: 未找到命令 '$cmd'，请先安装。"
        exit 1
    fi
done

# 1. 创建目标文件夹
mkdir -p "${TARGET_DIR}"

# 2. 下载数据压缩包
echo "[2/3] 正在下载数据文件..."
# -f: HTTP 错误时返回非零状态码; --retry: 网络不稳定时重试
curl -L -f --retry 3 -o "${TMP_FILE}" "${DOWNLOAD_URL}"

if [ $? -ne 0 ]; then
    echo "❌ 下载失败！检测到网络连接问题或代理限制 (如 403 Forbidden)。"
    rm -f "${TMP_FILE}"
    exit 1
fi

# 验证文件类型 (防止下载了 HTML 错误页面)
if ! command -v file &> /dev/null; then
    echo "⚠️ 警告: 未找到 'file' 命令，跳过文件类型验证。"
else
    if ! file "${TMP_FILE}" | grep -q "gzip compressed data"; then
        echo "❌ 错误: 下载的文件不是有效的 gzip 压缩包。内容可能是错误提示："
        head -n 5 "${TMP_FILE}"
        rm -f "${TMP_FILE}"
        exit 1
    fi
fi

# 3. 解压缩
echo "[3/3] 下载完成，正在解压到 ${TARGET_DIR}..."
tar -zxf "${TMP_FILE}" -C "${TARGET_DIR}" --strip-components=1

if [ $? -eq 0 ]; then
    echo "✅ 数据解压成功！"
    rm -f "${TMP_FILE}"
    echo "🎉 数据已保存在: ${TARGET_DIR}"
else
    echo "❌ 解压失败！可能是压缩包结构不符合 --strip-components=1 的预期。"
    rm -f "${TMP_FILE}"
    exit 1
fi
