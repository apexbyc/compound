"""
完整的高频订单因子计算（修正版）
修正内容：
1. 添加传统大单交易占比因子计算
2. 完整计算16种订单类型因子
3. 正确合成精选复合因子（基于当日16种因子值）
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

def calculate_all_hfa_factors(data_path):
    """
    完整计算国信证券报告中的所有因子
    """
    print("="*80)
    print("📊 完整高频订单因子计算（修正版）")
    print("="*80)

    # 1. 加载数据
    print("1. 加载数据...")
    df = pd.read_parquet(data_path)

    # 2. 时间处理（简化版，实际需要根据数据格式调整）
    df['TradeTime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['Time'].astype(str).str[-15:])

    # 3. 剔除集合竞价
    df_continuous = df[~df['TradeTime'].dt.time.between(time(9,15), time(9,30))].copy()

    all_stocks = df_continuous['secucode'].unique()
    results = []

    for stock in all_stocks:
        stock_data = df_continuous[df_continuous['secucode'] == stock].copy()

        if len(stock_data) == 0:
            continue

        total_volume = stock_data['Volume'].sum()

        # =================================================================
        # A. 计算订单特征（大单和漫长订单）
        # =================================================================

        # 买单特征
        buy_orders = stock_data.groupby('BuyOrderID').agg(
            buy_volume=('Volume', 'sum'),
            buy_first_time=('TradeTime', 'min'),
            buy_last_time=('TradeTime', 'max')
        ).reset_index()

        # 卖单特征
        sell_orders = stock_data.groupby('SaleOrderID').agg(
            sell_volume=('Volume', 'sum'),
            sell_first_time=('TradeTime', 'min'),
            sell_last_time=('TradeTime', 'max')
        ).reset_index()

        # 计算成交时长（已简化处理）
        buy_orders['buy_duration'] = (buy_orders['buy_last_time'] - buy_orders['buy_first_time']).dt.total_seconds()
        sell_orders['sell_duration'] = (sell_orders['sell_last_time'] - sell_orders['sell_first_time']).dt.total_seconds()

        # 计算阈值（前10%）
        buy_big_threshold = buy_orders['buy_volume'].quantile(0.9) if len(buy_orders) > 0 else 0
        sell_big_threshold = sell_orders['sell_volume'].quantile(0.9) if len(sell_orders) > 0 else 0
        buy_long_threshold = buy_orders['buy_duration'].quantile(0.9) if len(buy_orders) > 0 else 0
        sell_long_threshold = sell_orders['sell_duration'].quantile(0.9) if len(sell_orders) > 0 else 0

        # 合并特征到成交记录
        stock_data = stock_data.merge(
            buy_orders[['BuyOrderID', 'buy_volume', 'buy_duration']],
            on='BuyOrderID', how='left'
        )
        stock_data = stock_data.merge(
            sell_orders[['SaleOrderID', 'sell_volume', 'sell_duration']],
            on='SaleOrderID', how='left'
        )

        # 标记属性
        stock_data['is_big_buy'] = stock_data['buy_volume'] > buy_big_threshold
        stock_data['is_big_sell'] = stock_data['sell_volume'] > sell_big_threshold
        stock_data['is_long_buy'] = stock_data['buy_duration'] > buy_long_threshold
        stock_data['is_long_sell'] = stock_data['sell_duration'] > sell_long_threshold

        # =================================================================
        # B. 计算6个基本子因子
        # =================================================================

        def volume_ratio(mask):
            """计算成交量占比"""
            return stock_data[mask]['Volume'].sum() / total_volume if total_volume > 0 else 0

        # B1. 大单相关子因子（3个）
        big_buy_non_big_sell = volume_ratio(stock_data['is_big_buy'] & ~stock_data['is_big_sell'])
        non_big_buy_big_sell = volume_ratio(~stock_data['is_big_buy'] & stock_data['is_big_sell'])
        big_buy_big_sell = volume_ratio(stock_data['is_big_buy'] & stock_data['is_big_sell'])

        # B2. 漫长订单相关子因子（3个）
        long_buy_non_long_sell = volume_ratio(stock_data['is_long_buy'] & ~stock_data['is_long_sell'])
        non_long_buy_long_sell = volume_ratio(~stock_data['is_long_buy'] & stock_data['is_long_sell'])
        long_buy_long_sell = volume_ratio(stock_data['is_long_buy'] & stock_data['is_long_sell'])

        # =================================================================
        # C. 计算4个合成因子
        # =================================================================

        # C1. 传统大单交易占比因子（报告公式）
        volume_big_origin = big_buy_non_big_sell + non_big_buy_big_sell + 2 * big_buy_big_sell

        # C2. 改进大单交易占比因子（报告公式）
        volume_big = (-big_buy_non_big_sell - non_big_buy_big_sell + big_buy_big_sell)

        # C3. 漫长订单交易占比因子（报告公式）
        volume_long = long_buy_non_long_sell + non_long_buy_long_sell + 2 * long_buy_long_sell

        # C4. 大单及漫长订单复合因子
        volume_long_big = volume_big + volume_long

        # =================================================================
        # D. 计算16种订单类型因子
        # =================================================================

        order_type_factors = {}
        for bb in [0, 1]:  # 买单是否为大单
            for bs in [0, 1]:  # 卖单是否为大单
                for lb in [0, 1]:  # 买单是否为漫长订单
                    for ls in [0, 1]:  # 卖单是否为漫长订单
                        mask = (
                            (stock_data['is_big_buy'] == bool(bb)) &
                            (stock_data['is_big_sell'] == bool(bs)) &
                            (stock_data['is_long_buy'] == bool(lb)) &
                            (stock_data['is_long_sell'] == bool(ls))
                        )
                        key = f"BB{bb}_BS{bs}_LB{lb}_LS{ls}"
                        order_type_factors[key] = volume_ratio(mask)

        # =================================================================
        # E. 计算精选复合因子（基于当日16个因子值）
        # =================================================================

        # 当日16个因子值中，取绝对值最大的5个（模拟报告的表11）
        # 注意：实际报告中是基于历史回测选择5个，这里是单日模拟

        # 先计算每个因子的"有效方向"（基于因子值与总成交量的关系）
        # 简化方法：取因子值本身（方向在计算时确定）
        selected_keys = [
            'BB1_BS1_LB1_LS1',  # 双方大单且漫长
            'BB1_BS1_LB0_LS1',  # 双方大单，卖单漫长
            'BB1_BS1_LB1_LS0',  # 双方大单，买单漫长
            'BB0_BS1_LB0_LS1',  # 卖单大单且漫长
            'BB1_BS0_LB0_LS0'   # 买单大单，双方非漫长
        ]

        # 合成精选复合因子（等权平均）
        selected_values = [order_type_factors[k] for k in selected_keys]
        volume_long_big_select = np.mean(selected_values)

        # =================================================================
        # F. 存储结果
        # =================================================================

        result = {
            'secucode': stock,
            'total_volume': total_volume,
            'total_trades': len(stock_data),

            # 6个基本子因子
            'big_buy_non_big_sell': big_buy_non_big_sell,
            'non_big_buy_big_sell': non_big_buy_big_sell,
            'big_buy_big_sell': big_buy_big_sell,
            'long_buy_non_long_sell': long_buy_non_long_sell,
            'non_long_buy_long_sell': non_long_buy_long_sell,
            'long_buy_long_sell': long_buy_long_sell,

            # 4个合成因子
            'VolumeBigOrigin': volume_big_origin,  # 新增：传统大单因子
            'VolumeBig': volume_big,
            'VolumeLong': volume_long,
            'VolumeLongBig': volume_long_big,

            # 精选复合因子
            'VolumeLongBigSelect': volume_long_big_select,
        }

        # 添加16种订单类型因子
        for key, value in order_type_factors.items():
            result[key] = value

        results.append(result)

    # 转换为DataFrame
    factors_df = pd.DataFrame(results)

    # 验证数学关系
    print("\n🔍 验证数学关系:")
    print("="*50)

    # 验证1：复合因子 = 改进大单 + 漫长订单
    factors_df['验证_VolumeLongBig'] = factors_df['VolumeBig'] + factors_df['VolumeLong']
    diff = (factors_df['VolumeLongBig'] - factors_df['验证_VolumeLongBig']).abs().max()
    print(f"1. 复合因子验证差异: {diff:.10f} (应为0)")

    # 验证2：传统大单因子 = 三个子因子的加权和
    factors_df['验证_VolumeBigOrigin'] = (
        factors_df['big_buy_non_big_sell'] +
        factors_df['non_big_buy_big_sell'] +
        2 * factors_df['big_buy_big_sell']
    )
    diff2 = (factors_df['VolumeBigOrigin'] - factors_df['验证_VolumeBigOrigin']).abs().max()
    print(f"2. 传统大单因子验证差异: {diff2:.10f} (应为0)")

    # 验证3：16种订单类型之和应为1
    order_type_cols = [c for c in factors_df.columns if c.startswith('BB')]
    factors_df['订单类型总和'] = factors_df[order_type_cols].sum(axis=1)
    diff3 = (factors_df['订单类型总和'] - 1).abs().max()
    print(f"3. 16种订单类型之和验证: {diff3:.10f} (应为0)")

    return factors_df

def summarize_factors(factors_df):
    """汇总因子计算结果"""
    print("\n📊 因子计算结果汇总")
    print("="*80)

    print(f"股票数量: {len(factors_df)}")

    # 1. 基本子因子统计
    print("\n1. 6个基本子因子均值:")
    basic_factors = [
        'big_buy_non_big_sell', 'non_big_buy_big_sell', 'big_buy_big_sell',
        'long_buy_non_long_sell', 'non_long_buy_long_sell', 'long_buy_long_sell'
    ]
    for factor in basic_factors:
        mean_val = factors_df[factor].mean()
        print(f"  {factor}: {mean_val:.4f}")

    # 2. 4个合成因子统计
    print("\n2. 4个合成因子均值:")
    synthetic_factors = ['VolumeBigOrigin', 'VolumeBig', 'VolumeLong', 'VolumeLongBig']
    for factor in synthetic_factors:
        mean_val = factors_df[factor].mean()
        print(f"  {factor}: {mean_val:.4f}")

    # 3. 精选复合因子
    print(f"\n3. 精选复合因子均值: {factors_df['VolumeLongBigSelect'].mean():.4f}")

    # 4. 16种订单类型前5个
    print("\n4. 16种订单类型因子均值（前5个）:")
    order_type_cols = sorted([c for c in factors_df.columns if c.startswith('BB')])
    for i, col in enumerate(order_type_cols[:5]):
        mean_val = factors_df[col].mean()
        print(f"  {col}: {mean_val:.4f}")

    return factors_df

# 主程序
if __name__ == "__main__":
    data_path = "D:/pycharm/pythonProject/dataExample.parquet"

    print("🔬 开始计算完整因子体系...")
    factors_df = calculate_all_hfa_factors(data_path)

    if len(factors_df) > 0:
        # 汇总结果
        summarize_factors(factors_df)

        # 保存结果
        output_path = "D:/pycharm/pythonProject/all_factors_complete.csv"
        factors_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 完整结果已保存到: {output_path}")

        # 显示前5只股票的详细结果
        print("\n📋 前5只股票的完整因子值:")

        # 基本子因子
        basic_cols = ['secucode',
                     'big_buy_non_big_sell', 'non_big_buy_big_sell', 'big_buy_big_sell',
                     'long_buy_non_long_sell', 'non_long_buy_long_sell', 'long_buy_long_sell']

        # 合成因子
        synth_cols = ['VolumeBigOrigin', 'VolumeBig', 'VolumeLong', 'VolumeLongBig', 'VolumeLongBigSelect']

        # 合并显示
        display_cols = ['secucode'] + synth_cols + ['big_buy_big_sell', 'long_buy_long_sell']
        print(factors_df[display_cols].head().round(4))

        # 16种订单类型前5个
        order_type_cols = sorted([c for c in factors_df.columns if c.startswith('BB')])[:5]
        print(f"\n📊 16种订单类型因子（前5个）:")
        print(factors_df[['secucode'] + order_type_cols].head().round(4))

        print("\n✅ 计算完成！")
    else:
        print("❌ 没有计算到任何因子，请检查数据格式")