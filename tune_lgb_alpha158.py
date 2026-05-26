#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib LightGBM 自动超参数调优与优化配置生成脚本 (Robust 工业升级版)
功能特色：
1. 滚动/多段验证（Multi-Segment Validation）：将验证集按年份切分为独立风格的切片（如 2022 熊市 vs 2023 震荡市）。
2. 稳健得分设计（Variance Penalty）：评分函数引入“年度业绩波动惩罚”，偏好在所有年份都能及格的“全能型参数”，抛弃特定年份暴利但偏科的参数。
3. 软上限截断惩罚（Skepticism of High ICIR）：对超高 Raw ICIR (>0.25) 启动反向折返惩罚，防止贝叶斯搜索误入“数据泄露”或“过拟合”陷阱。
4. 基准参数继承（Params Inheritance）：自动保留并继承原始 YAML 配置文件中除调优参数外的所有自定义字段。
5. 索引安全降级（Fallback Indexing）：计算 Rank ICIR 时自动兼容并识别各种格式的时间戳和索引结构，防止 Pandas 崩溃。
6. 动态 Study 命名：study_name 自动绑定当前调优的配置文件名，防止 SQLite 数据库记录发生冲突。
"""

import os
import argparse
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

# 屏蔽 Optuna 和 LightGBM 的过多冗余日志
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib LightGBM Hyperparameter Tuning Script")
    parser.add_argument(
        "--config_path",
        type=str,
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_2016_2026.yaml",
        help="Path to the baseline YAML configuration file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_2016_2026_optimized.yaml",
        help="Path to save the optimized YAML configuration file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna trials to run",
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
    
    # 按 datetime 分组计算每日 Spearman 秩相关系数
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
    print("           Qlib 工业级 LightGBM 稳健超参数调优系统 (Walk-Forward / MSV)")
    print("=" * 80)
    print(f"1. 正在载入基准配置文件: {args.config_path}")

    # 1. 载入原始 YAML 文件以保留格式和数据结构
    yaml_parser = YAML(typ="safe", pure=True)
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml_parser.load(f)

    # 2. 初始化 Qlib 框架
    print("2. 正在初始化 Qlib 框架...")
    qlib.init(**config["qlib_init"])

    # 3. 实例化数据集 (Train/Valid/Test)
    print("3. 正在加载数据集与特征计算（这在日频大数据集下可能需要一点时间）...")
    dataset = init_instance_by_config(config["task"]["dataset"])

    # 4. 提前提取验证集（Validation Set）的 Label 以大幅提高调参循环速度
    print("4. 提取验证集真实 Label 用于优化评测...")
    valid_label_df = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R)
    valid_label = valid_label_df.iloc[:, 0]

    # 定义 Optuna 的优化目标函数
    def objective(trial):
        # 建议抗过拟合、高泛化的参数搜索区间
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01)
        max_depth = trial.suggest_int("max_depth", 3, 7)
        num_leaves = trial.suggest_int("num_leaves", 15, min(127, 2**max_depth - 1))
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 500, 3000, step=100)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.8, step=0.05)
        subsample = trial.suggest_float("subsample", 0.5, 0.9, step=0.05)
        lambda_l1 = trial.suggest_float("lambda_l1", 10.0, 1000.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 10.0, 1000.0, log=True)
        
        # 允许搜索不同的 Loss 提升抗噪能力
        objective_loss = trial.suggest_categorical("objective", ["regression", "regression_l1", "huber"])

        # 4.1 基准参数继承：保留原 YAML 中除了我们要优化的参数以外的其它重要 kwargs 参数
        base_kwargs = dict(config["task"]["model"]["kwargs"])
        base_kwargs.update({
            "loss": "mse",  # 绕过 Qlib 的硬编码检查
            "objective": objective_loss,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "bagging_freq": 1 if subsample < 1.0 else 0,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "num_threads": 4, # 调参时每个试验限制线程数，防止多卡/多核竞争冲突
            "verbosity": -1,
        })

        if objective_loss == "huber":
            base_kwargs["alpha"] = trial.suggest_float("huber_alpha", 0.85, 0.95)
        elif "alpha" in base_kwargs:
            del base_kwargs["alpha"]

        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": base_kwargs,
        }

        # 实例化并拟合模型
        try:
            model = init_instance_by_config(model_config)
            # 使用 dataset 自动进行训练与验证集早停
            model.fit(dataset, verbose_eval=False)
            
            # 在验证集上进行预测
            pred = model.predict(dataset, segment="valid")
            
            # 计算全体验证集指标
            global_ic_mean, global_icir = calc_rank_ic(pred, valid_label)
            
            # 4.2 工业级设计1：多段分年度验证与平稳度波动惩罚
            # 提取时间戳序列
            dt_index = pred.index.get_level_values("datetime" if "datetime" in pred.index.names else 0)
            try:
                datetime_series = pd.to_datetime(dt_index)
                years = datetime_series.year.unique()
                is_datetime_valid = True
            except Exception:
                is_datetime_valid = False

            if is_datetime_valid and len(years) > 1:
                sub_icir_list = []
                sub_ic_mean_list = []
                for yr in years:
                    mask = datetime_series.year == yr
                    if mask.sum() > 5:
                        sub_pred = pred[mask]
                        sub_label = valid_label[mask]
                        sub_ic_mean, sub_icir = calc_rank_ic(sub_pred, sub_label)
                        sub_ic_mean_list.append(sub_ic_mean)
                        sub_icir_list.append(sub_icir)

                # 计算多段业绩波动惩罚分
                mean_sub_icir = np.mean(sub_icir_list)
                std_sub_icir = np.std(sub_icir_list) if len(sub_icir_list) > 1 else 0.0
                
                # 稳健得分 = 平均分 - 0.5 * 标准差 （偏好在所有年度都表现平稳的参数组合）
                robust_score = mean_sub_icir - 0.5 * std_sub_icir
                
                # 惩罚项：如果有任意一年的 IC 均值为负（风格严重踩雷），则进行一刀切归零惩罚
                if any(m <= 0 for m in sub_ic_mean_list):
                    robust_score = 0.0
            else:
                # 无法分段或仅有1年，直接降级使用全局 ICIR
                robust_score = global_icir if global_ic_mean > 0 else 0.0

            # 4.3 工业级设计2：对超高 Valid ICIR 保持警惕（软上限截断折返惩罚）
            # A股日频 Raw Rank ICIR 稳定在 0.20-0.25 (未年化) 已经是极值，超过 0.25 极大概率发生过拟合或信息泄露。
            # 我们对 >0.25 之后的值不予鼓励，反而进行 2 倍溢出扣分惩罚，迫使贝叶斯算法回归平稳可靠区间。
            final_score = robust_score
            if final_score > 0.25:
                penalty = 2.0 * (final_score - 0.25)
                final_score = max(0.0, 0.25 - penalty)

            print(f"Trial {trial.number:02d} | Loss: {objective_loss:13s} | LR: {learning_rate:.2f} | Depth: {max_depth} | Global ICIR: {global_icir:.4f} | Robust Score: {robust_score:.4f} | Opt Score: {final_score:.4f}")
            return final_score
        except Exception as e:
            print(f"Trial {trial.number:02d} | 运行出错: {e}")
            return 0.0

    print("=" * 80)
    print(f"5. 开始运行贝叶斯搜索优化 (共 {args.trials} 轮试验)...")
    print("=" * 80)

    # 动态命名 study_name 防止不同数据集/配置的 SQLite 记录发生交叉污染
    config_name = Path(args.config_path).stem
    study_name = f"tune_lgb_{config_name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=args.db_path,
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=args.trials)

    print("=" * 80)
    print("6. 调参完成！最优参数组合如下:")
    print("=" * 80)
    best_params = study.best_params
    for k, v in best_params.items():
        print(f" - {k}: {v}")
    print(f" - 最佳验证集稳健设计得分: {study.best_value:.4f}")

    # 5. 读取基准 YAML 重新构建并输出最优化配置文件
    # 采用带有注释保留的 YAML 加载器重新读写，保持文件可读性
    with open(args.config_path, "r", encoding="utf-8") as f:
        raw_yaml_text = f.read()

    # 重新加载 ruamel.yaml 并更新具体字段
    yaml_rw = YAML()
    yaml_rw.preserve_quotes = True
    config_rw = yaml_rw.load(raw_yaml_text)

    # 提取最优化超参数值
    opt_lr = best_params["learning_rate"]
    opt_depth = best_params["max_depth"]
    # 保证 num_leaves 不越界
    opt_leaves = int(min(best_params["num_leaves"], 2**opt_depth - 1))
    opt_leaf_size = int(best_params["min_data_in_leaf"])
    opt_colsample = best_params["colsample_bytree"]
    opt_subsample = best_params["subsample"]
    opt_l1 = best_params["lambda_l1"]
    opt_l2 = best_params["lambda_l2"]
    opt_obj = best_params["objective"]

    # 替换至配置字典中
    model_kwargs = config_rw["task"]["model"]["kwargs"]
    model_kwargs["learning_rate"] = float(opt_lr)
    model_kwargs["max_depth"] = int(opt_depth)
    model_kwargs["num_leaves"] = int(opt_leaves)
    model_kwargs["min_data_in_leaf"] = int(opt_leaf_size)
    model_kwargs["colsample_bytree"] = float(opt_colsample)
    model_kwargs["subsample"] = float(opt_subsample)
    model_kwargs["bagging_freq"] = 1 if opt_subsample < 1.0 else 0
    model_kwargs["lambda_l1"] = float(opt_l1)
    model_kwargs["lambda_l2"] = float(opt_l2)
    model_kwargs["objective"] = str(opt_obj)

    if opt_obj == "huber":
        model_kwargs["alpha"] = float(best_params["huber_alpha"])
    elif "alpha" in model_kwargs:
        del model_kwargs["alpha"]

    # 保存新的 YAML 配置文件
    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml_rw.dump(config_rw, f)

    print("=" * 80)
    print(f"7. 已自动生成最优化配置文件: {args.output_path}")
    print("您可以直接执行以下命令运行完整训练与测试集回测流程:")
    print(f"   .venv/bin/python -m qlib.cli.run {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
