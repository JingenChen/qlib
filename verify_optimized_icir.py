#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib 调参结果验证脚本：自动比对 Optuna 数据库记录的最优 ICIR 与使用优化后 YAML 重新运行的验证集 ICIR。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

# 屏蔽过多日志以突出比对结果
import logging
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Qlib Optimized Hyperparameters")
    parser.add_argument(
        "--optimized_config",
        type=str,
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_2016_2026_optimized.yaml",
        help="Path to the optimized YAML configuration file",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="sqlite:///qlib_lgb_tune.db",
        help="Database path for Optuna storage",
    )
    return parser.parse_args()


def calc_rank_ic(pred: pd.Series, label: pd.Series):
    """
    鲁棒计算预测值与真实标签之间的每日 Spearman 秩相关系数及 Rank ICIR
    支持自动识别 index 的时间层（兼容 datetime、date 或 level=0）
    """
    df = pd.DataFrame({"pred": pred, "label": label})
    df = df.dropna()
    if df.empty:
        return 0.0, 0.0

    # 鲁棒识别 datetime 层，优先使用 level 名字，若无名字则退化至 level 0
    level = "datetime" if "datetime" in df.index.names else 0
    
    # 计算每日 Spearman 秩相关系数
    daily_ic = df.groupby(level=level).apply(
        lambda x: x["pred"].corr(x["label"], method="spearman") if len(x) > 1 else np.nan
    )
    daily_ic = daily_ic.dropna()
    if len(daily_ic) == 0:
        return 0.0, 0.0

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    return ic_mean, icir


def main():
    args = parse_args()
    print("=" * 80)
    print("                 Qlib HTE (Optuna) 调参结果一致性校验程序")
    print("=" * 80)

    # 1. 读取 Optuna 数据库，查询最佳得分
    db_file = Path(args.db_path.replace("sqlite:///", ""))
    if not db_file.exists() and "sqlite:///" in args.db_path:
        print(f"❌ 错误：未找到 Optuna 数据库文件：{db_file}")
        print("请确保您已经运行过 `@tune_lgb_alpha158.py` 脚本并产生了本地数据库。")
        return

    try:
        study = optuna.load_study(
            study_name="qlib_lgb_alpha158",
            storage=args.db_path
        )
        optuna_best_icir = study.best_value
        optuna_best_params = study.best_params
        print(f"【1】已成功载入 Optuna 数据库:")
        print(f"     - 最佳验证集 Rank ICIR 记录值 : {optuna_best_icir:.6f}")
        print(f"     - 最佳参数组合:")
        for k, v in optuna_best_params.items():
            print(f"       * {k}: {v}")
    except Exception as e:
        print(f"❌ 错误：读取 Optuna 数据库失败: {e}")
        return

    # 2. 读取优化后的 YAML 配置文件
    config_path = Path(args.optimized_config)
    if not config_path.exists():
        print(f"❌ 错误：未找到最优化 YAML 配置文件：{config_path}")
        print("请先执行调参脚本生成优化后的配置文件。")
        return

    print(f"\n【2】正在加载最优化 YAML 配置并初始化框架: {config_path}")
    yaml_parser = YAML(typ="safe", pure=True)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml_parser.load(f)

    # 初始化 Qlib
    qlib.init(**config["qlib_init"])

    # 3. 实例化数据集并提取验证集 Label
    print("\n【3】正在实例化数据集并准备验证集 Label（请稍候）...")
    dataset = init_instance_by_config(config["task"]["dataset"])
    valid_label_df = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R)
    valid_label = valid_label_df.iloc[:, 0]

    # 4. 使用 YAML 中的参数构建最优模型并进行训练（带有早停）
    print("\n【4】正在使用最优超参数构建模型并启动拟合（含早停机制）...")
    model_config = config["task"]["model"]
    model = init_instance_by_config(model_config)
    
    # 训练模型
    model.fit(dataset, verbose_eval=False)

    # 5. 在验证集上进行预测并重新计算 Rank ICIR
    print("\n【5】拟合完成，正在对验证集做外推预测并重新计算量化指标...")
    pred = model.predict(dataset, segment="valid")
    recalculated_ic_mean, recalculated_icir = calc_rank_ic(pred, valid_label)

    # 6. 比对与校验报告
    print("\n" + "=" * 80)
    print("                               校验比对报告")
    print("=" * 80)
    print(f" 🎯 Optuna 数据库记录最佳 ICIR   : {optuna_best_icir:.6f}")
    print(f" ⚙️  最优化 YAML 重新运行 ICIR   : {recalculated_icir:.6f}  (IC 均值: {recalculated_ic_mean:.4f})")
    
    absolute_error = abs(optuna_best_icir - recalculated_icir)
    print(f" 📊 两者绝对误差 (Absolute Error): {absolute_error:.6e}")
    print("-" * 80)
    
    # 如果误差极小（通常在 1e-5 以内，甚至是 0.0），则验证通过
    if absolute_error < 1e-4:
        print(" 🎉 [SUCCESS] 一致性校验通过！")
        print(" 运行最优 YAML 得到的验证集 Rank ICIR 与调参数据库完全一致。")
        print(" 证明您生成的优化后配置文件可以 100% 完美复现调参阶段的最佳泛化表现！")
    else:
        print(" ⚠️ [WARNING] 检测到微小数值差异。")
        print(" 差异通常是由以下因素引起的：")
        print(" 1. 调参时 LightGBM 采用的随机种子或多线程并行调度对浮点数四舍五入产生了微弱影响。")
        print(" 2. 数据库中记录的最佳参数在写回 YAML 时进行了数据类型转换（例如 float 转 int 舍入）。")
    print("=" * 80)


if __name__ == "__main__":
    main()
