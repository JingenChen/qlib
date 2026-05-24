import os
import json
import argparse
import pandas as pd

def load_current_positions(position_file):
    """Load currently held stock tickers from a local file."""
    if not os.path.exists(position_file):
        print(f"ℹ️ 未找到持仓文件 '{position_file}'，默认当前为空仓 (100% 现金)。")
        return []
    
    with open(position_file, "r") as f:
        # Read lines, strip whitespace/newlines, ignore comments and empty lines
        positions = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return positions

def save_new_positions(position_file, positions, model_name):
    """Save the updated list of held stock tickers to a local file."""
    with open(position_file, "w") as f:
        f.write(f"# 当前实盘/虚拟盘持仓清单 - {model_name.upper()} 策略 (每日自动更新)\n")
        for pos in positions:
            f.write(f"{pos}\n")
    print(f"💾 已将最新持仓更新并保存至: {position_file}")

def generate_signals(pred_path, current_positions, topk=20, n_drop=2):
    """Implement the exact math of Qlib's TopkDropoutStrategy."""
    df = pd.read_pickle(pred_path)
    
    # Get the latest prediction date in the dataset
    latest_date = df.index.get_level_values('datetime').max()
    print(f"📅 加载预测基准日: {latest_date.strftime('%Y-%m-%d')}")
    
    # Filter predictions for the latest date
    df_latest = df.xs(latest_date, level='datetime')
    pred_score = df_latest['score']
    
    # 1. Classify current positions
    current_stock_list = [p for p in current_positions if p in pred_score.index]
    missing_stock_list = [p for p in current_positions if p not in pred_score.index]
    if missing_stock_list:
        print(f"⚠️ 警告: 有 {len(missing_stock_list)} 只持仓股目前不在模型预测股票池中: {missing_stock_list}")
    
    # Sort currently held stocks by their current scores (last index list)
    last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
    
    # 2. Get new buy candidates (not already held, sorted by score)
    not_held = pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index
    max_to_buy = n_drop + topk - len(last)
    today = not_held[:max_to_buy]
    
    # 3. Create combination of currently held + top candidates for dropout evaluation
    comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
    
    # 4. Determine sell signals (bottom n_drop in the combination pool)
    bottom_n = comb[-n_drop:] if len(comb) >= n_drop else comb
    sell = last[last.isin(bottom_n)]
    
    # 5. Determine buy signals (top candidates to fill the gap)
    buy_count = len(sell) + topk - len(last)
    buy = today[:buy_count]
    
    # 6. Determine hold signals (held stocks that are NOT sold)
    hold = [s for p in last if (s := str(p)) not in sell]
    
    return latest_date, list(sell), list(buy), list(hold), pred_score

def main():
    parser = argparse.ArgumentParser(description="Qlib 每日调仓信号生成与持仓管理脚本 (Git 归档多策略版)")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["csi300", "csi500"], 
        default="csi300", 
        help="选择调仓策略模型: csi300 (大盘蓝筹) 或 csi500 (中盘成长). 默认: csi300"
    )
    parser.add_argument("--topk", type=int, help="最大持仓股票数 K (默认: csi300为20, csi500为20)")
    parser.add_argument("--n_drop", type=int, default=2, help="每日最大换手机会数 N (默认: 2)")
    parser.add_argument("--update_holdings", action="store_true", help="是否在生成信号后自动将虚拟持仓更新写入持仓文件")
    parser.add_argument("--position_file", type=str, help="本地持仓记录文件路径 (默认自动路由: positions_<model>.txt)")
    parser.add_argument("--output_json", type=str, help="输出的 JSON 指令清单文件路径 (默认自动路由: signals_<model>.json)")
    
    args = parser.parse_args()
    
    # Multi-strategy automatic routing
    model_name = args.model
    pred_path = f"models/{model_name}/pred.pkl"
    
    # Configure default parameters based on strategy
    default_topk = 20
    topk = args.topk if args.topk is not None else default_topk
    
    position_file = args.position_file if args.position_file else f"positions_{model_name}.txt"
    output_json = args.output_json if args.output_json else f"signals_{model_name}.json"
    
    print("=" * 70)
    print(f" ⚙️ 正在执行多策略路由: {model_name.upper()} 选股配置")
    print(f"  • 模型预测路径: {pred_path}")
    print(f"  • 持仓管理路径: {position_file}")
    print(f"  • 指令输出路径: {output_json}")
    print("=" * 70)
    
    if not os.path.exists(pred_path):
        print(f"❌ 错误: 未能在 Git 归档目录中找到模型预测数据 {pred_path}！")
        return
        
    # 2. Load current holdings
    current_positions = load_current_positions(position_file)
    print(f"📊 当前持仓数: {len(current_positions)} / {topk}")
    
    # 3. Generate Signals
    latest_date, sell, buy, hold, pred_score = generate_signals(
        pred_path, current_positions, topk=topk, n_drop=args.n_drop
    )
    
    # 4. Prepare JSON Payload for target virtual platform
    trade_date = (latest_date + pd.Timedelta(days=3) if latest_date.dayofweek == 4 else latest_date + pd.Timedelta(days=1)) # Simulating next trading day
    payload = {
        "model_strategy": model_name.upper(),
        "prediction_date": latest_date.strftime("%Y-%m-%d"),
        "target_execution_date": trade_date.strftime("%Y-%m-%d"),
        "strategy_params": {
            "topk": topk,
            "n_drop": args.n_drop
        },
        "instructions": {
            "SELL": [{"ticker": str(code), "score": float(pred_score[code])} for code in sell],
            "BUY": [{"ticker": str(code), "score": float(pred_score[code])} for code in buy],
            "HOLD": [{"ticker": str(code), "score": float(pred_score[code])} for code in hold]
        }
    }
    
    # Save JSON instructions
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print(f"\n✅ 调仓指令清单已导出至 JSON 文件: {output_json}")
    
    # 5. Print Human-Readable Table
    print("\n" + "=" * 70)
    print(f" 🎯 {model_name.upper()} 策略执行操作指引清单 (目标交易日: {trade_date.strftime('%Y-%m-%d')})")
    print("=" * 70)
    
    if sell:
        print("\n🔻 【卖出清单】(请在明日开盘/收盘时全额卖出以下股票):")
        print("-" * 70)
        for i, code in enumerate(sell, 1):
            print(f"  [{i:02d}] 卖出代码: {code} | 预测超额评分: {pred_score[code]:.6f}")
    else:
        print("\n🔻 【卖出清单】: 无需卖出股票。")
        
    if buy:
        print("\n🔺 【买入清单】(请将释放的资金均分买入以下新推荐股):")
        print("-" * 70)
        for i, code in enumerate(buy, 1):
            print(f"  [{i:02d}] 买入代码: {code} | 预测超额评分: {pred_score[code]:.6f}")
    else:
        print("\n🔺 【买入清单】: 无需买入新股票。")
        
    print("\n🔄 【继续持有清单】(无调仓指令，保持不动):")
    print("-" * 70)
    for i, code in enumerate(hold, 1):
        print(f"  [{i:02d}] 持有代码: {code} | 预测超额评分: {pred_score[code]:.6f}")
        
    print("=" * 70)
    
    # 6. Auto Update holdings if flagged
    if args.update_holdings:
        # New holdings are hold + buy
        new_holdings = hold + buy
        save_new_positions(position_file, new_holdings, model_name)
    else:
        print("\n💡 提示: 若要自动将以上更新后的持仓写入持仓文件，请在运行脚本时加上 `--update_holdings` 参数。")

if __name__ == "__main__":
    main()
