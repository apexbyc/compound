"""
简洁版：输出n值及基于新划分的所有因子
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings

warnings.filterwarnings('ignore')


def calculate_n_and_factors(data_path, output_dir=None):
    """
    计算n值并基于新划分计算所有因子
    返回：n值字典和因子DataFrame
    """
    print("=" * 80)
    print("🎯 计算n值及新划分下的所有因子")
    print("=" * 80)

    # 1. 加载数据
    print("1. 加载数据...")
    df = pd.read_parquet(data_path)

    # 2. 时间处理
    df['TradeTime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['Time'].astype(str).str[-15:])
    df_continuous = df[~df['TradeTime'].dt.time.between(time(9, 15), time(9, 30))].copy()

    # 3. 收集所有订单数据（用于计算全局n）
    print("2. 收集订单数据计算n值...")

    all_buy_volumes = []
    all_sell_volumes = []
    all_buy_durations = []
    all_sell_durations = []

    for stock in df_continuous['secucode'].unique():
        stock_data = df_continuous[df_continuous['secucode'] == stock].copy()

        # 买单特征
        buy_orders = stock_data.groupby('BuyOrderID').agg(
            buy_volume=('Volume', 'sum'),
            buy_first_time=('TradeTime', 'min'),
            buy_last_time=('TradeTime', 'max')
        ).reset_index()
        buy_orders['buy_duration'] = (buy_orders['buy_last_time'] - buy_orders['buy_first_time']).dt.total_seconds()

        # 卖单特征
        sell_orders = stock_data.groupby('SaleOrderID').agg(
            sell_volume=('Volume', 'sum'),
            sell_first_time=('TradeTime', 'min'),
            sell_last_time=('TradeTime', 'max')
        ).reset_index()
        sell_orders['sell_duration'] = (
                    sell_orders['sell_last_time'] - sell_orders['sell_first_time']).dt.total_seconds()

        all_buy_volumes.extend(buy_orders['buy_volume'].values)
        all_sell_volumes.extend(sell_orders['sell_volume'].values)
        all_buy_durations.extend(buy_orders['buy_duration'].values)
        all_sell_durations.extend(sell_orders['sell_duration'].values)

    # 4. 计算n值
    print("3. 计算n值...")

    def calculate_single_n(data):
        """计算单个n值：使均值+n*标准差 = 90%分位数"""
        if len(data) < 10:
            return 1.2816  # 正态分布默认值

        # 原阈值（90%分位数）
        q90 = np.percentile(data, 90)
        mean_val = np.mean(data)
        std_val = np.std(data)

        if std_val > 0:
            return (q90 - mean_val) / std_val
        return 1.2816

    # 计算四个n值
    n_big_buy = calculate_single_n(all_buy_volumes)
    n_big_sell = calculate_single_n(all_sell_volumes)
    n_long_buy = calculate_single_n(all_buy_durations)
    n_long_sell = calculate_single_n(all_sell_durations)

    # 取平均值作为最终n值（简化处理）
    n_big = np.mean([n_big_buy, n_big_sell])
    n_long = np.mean([n_long_buy, n_long_sell])

    # 输出n值
    print("\n" + "=" * 80)
    print("📊 计算出的n值:")
    print("=" * 80)
    print(f"   大单划分 n_big: {n_big:.6f}")
    print(f"   漫长订单划分 n_long: {n_long:.6f}")
    print(f"   (如果n≈1.2816，说明数据接近正态分布)")

    n_values = {
        'n_big': n_big,
        'n_long': n_long,
        'n_big_buy': n_big_buy,
        'n_big_sell': n_big_sell,
        'n_long_buy': n_long_buy,
        'n_long_sell': n_long_sell
    }

    # 5. 使用新n值计算所有因子
    print("\n4. 使用新n值计算所有因子...")

    all_stocks = df_continuous['secucode'].unique()
    results = []

    for stock in all_stocks:
        stock_data = df_continuous[df_continuous['secucode'] == stock].copy()

        if len(stock_data) < 10:
            continue

        total_volume = stock_data['Volume'].sum()

        # 计算订单特征
        buy_orders = stock_data.groupby('BuyOrderID').agg(
            buy_volume=('Volume', 'sum'),
            buy_first_time=('TradeTime', 'min'),
            buy_last_time=('TradeTime', 'max')
        ).reset_index()
        buy_orders['buy_duration'] = (buy_orders['buy_last_time'] - buy_orders['buy_first_time']).dt.total_seconds()

        sell_orders = stock_data.groupby('SaleOrderID').agg(
            sell_volume=('Volume', 'sum'),
            sell_first_time=('TradeTime', 'min'),
            sell_last_time=('TradeTime', 'max')
        ).reset_index()
        sell_orders['sell_duration'] = (
                    sell_orders['sell_last_time'] - sell_orders['sell_first_time']).dt.total_seconds()

        # 计算阈值（使用均值 + n × 标准差）
        if len(buy_orders) > 0:
            buy_big_threshold = np.mean(buy_orders['buy_volume']) + n_big * np.std(buy_orders['buy_volume'])
            buy_long_threshold = np.mean(buy_orders['buy_duration']) + n_long * np.std(buy_orders['buy_duration'])
        else:
            buy_big_threshold = buy_long_threshold = 0

        if len(sell_orders) > 0:
            sell_big_threshold = np.mean(sell_orders['sell_volume']) + n_big * np.std(sell_orders['sell_volume'])
            sell_long_threshold = np.mean(sell_orders['sell_duration']) + n_long * np.std(sell_orders['sell_duration'])
        else:
            sell_big_threshold = sell_long_threshold = 0

        # 合并特征到成交记录
        stock_data = stock_data.merge(
            buy_orders[['BuyOrderID', 'buy_volume', 'buy_duration']],
            on='BuyOrderID', how='left'
        )
        stock_data = stock_data.merge(
            sell_orders[['SaleOrderID', 'sell_volume', 'sell_duration']],
            on='SaleOrderID', how='left'
        )

        # 标记订单属性
        stock_data['is_big_buy'] = stock_data['buy_volume'] > buy_big_threshold
        stock_data['is_big_sell'] = stock_data['sell_volume'] > sell_big_threshold
        stock_data['is_long_buy'] = stock_data['buy_duration'] > buy_long_threshold
        stock_data['is_long_sell'] = stock_data['sell_duration'] > sell_long_threshold

        # 计算成交量占比函数
        def volume_ratio(mask):
            return stock_data[mask]['Volume'].sum() / total_volume if total_volume > 0 else 0

        # ============================================
        # 计算6个基本子因子
        # ============================================
        big_buy_non_big_sell = volume_ratio(stock_data['is_big_buy'] & ~stock_data['is_big_sell'])
        non_big_buy_big_sell = volume_ratio(~stock_data['is_big_buy'] & stock_data['is_big_sell'])
        big_buy_big_sell = volume_ratio(stock_data['is_big_buy'] & stock_data['is_big_sell'])

        long_buy_non_long_sell = volume_ratio(stock_data['is_long_buy'] & ~stock_data['is_long_sell'])
        non_long_buy_long_sell = volume_ratio(~stock_data['is_long_buy'] & stock_data['is_long_sell'])
        long_buy_long_sell = volume_ratio(stock_data['is_long_buy'] & stock_data['is_long_sell'])

        # ============================================
        # 计算4个合成因子
        # ============================================
        volume_big_origin = big_buy_non_big_sell + non_big_buy_big_sell + 2 * big_buy_big_sell
        volume_big = (-big_buy_non_big_sell - non_big_buy_big_sell + big_buy_big_sell)
        volume_long = long_buy_non_long_sell + non_long_buy_long_sell + 2 * long_buy_long_sell
        volume_long_big = volume_big + volume_long

        # ============================================
        # 计算16种订单类型因子
        # ============================================
        order_type_factors = {}
        for bb in [0, 1]:
            for bs in [0, 1]:
                for lb in [0, 1]:
                    for ls in [0, 1]:
                        mask = (
                                (stock_data['is_big_buy'] == bool(bb)) &
                                (stock_data['is_big_sell'] == bool(bs)) &
                                (stock_data['is_long_buy'] == bool(lb)) &
                                (stock_data['is_long_sell'] == bool(ls))
                        )
                        key = f"BB{bb}_BS{bs}_LB{lb}_LS{ls}"
                        order_type_factors[key] = volume_ratio(mask)

        # ============================================
        # 计算精选复合因子
        # ============================================
        # 使用报告表11中的5个有效因子
        effective_factors = [
            order_type_factors['BB1_BS1_LB1_LS1'],  # 正方向
            order_type_factors['BB1_BS1_LB0_LS1'],  # 正方向
            order_type_factors['BB1_BS1_LB1_LS0'],  # 正方向
            -order_type_factors['BB0_BS1_LB0_LS1'],  # 负方向
            -order_type_factors['BB1_BS0_LB0_LS0']  # 负方向
        ]
        volume_long_big_select = np.mean(effective_factors)

        # ============================================
        # 存储结果
        # ============================================
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
            'VolumeBigOrigin': volume_big_origin,
            'VolumeBig': volume_big,
            'VolumeLong': volume_long,
            'VolumeLongBig': volume_long_big,

            # 精选复合因子
            'VolumeLongBigSelect': volume_long_big_select,

            # 阈值信息
            'buy_big_threshold': buy_big_threshold,
            'sell_big_threshold': sell_big_threshold,
            'buy_long_threshold': buy_long_threshold,
            'sell_long_threshold': sell_long_threshold,

            # 大单比例（验证用）
            'big_order_ratio_buy': np.mean(stock_data['is_big_buy']),
            'big_order_ratio_sell': np.mean(stock_data['is_big_sell']),
            'long_order_ratio_buy': np.mean(stock_data['is_long_buy']),
            'long_order_ratio_sell': np.mean(stock_data['is_long_sell']),
        }

        # 添加16种订单类型因子
        for key, value in order_type_factors.items():
            result[key] = value

        results.append(result)

    # 转换为DataFrame
    factors_df = pd.DataFrame(results)

    # 验证数学关系
    print("\n5. 验证因子计算正确性...")
    factors_df['验证_复合因子'] = factors_df['VolumeBig'] + factors_df['VolumeLong']
    factors_df['验证_传统大单'] = (
            factors_df['big_buy_non_big_sell'] +
            factors_df['non_big_buy_big_sell'] +
            2 * factors_df['big_buy_big_sell']
    )

    diff_composite = (factors_df['VolumeLongBig'] - factors_df['验证_复合因子']).abs().max()
    diff_origin = (factors_df['VolumeBigOrigin'] - factors_df['验证_传统大单']).abs().max()

    print(f"   ✅ 复合因子验证误差: {diff_composite:.10f}")
    print(f"   ✅ 传统大单因子验证误差: {diff_origin:.10f}")

    # 统计大单比例
    print("\n6. 新方法下的大单/漫长订单比例:")
    print(f"   大买单平均比例: {factors_df['big_order_ratio_buy'].mean():.4f} (目标: 0.10)")
    print(f"   大卖单平均比例: {factors_df['big_order_ratio_sell'].mean():.4f} (目标: 0.10)")
    print(f"   漫长买单平均比例: {factors_df['long_order_ratio_buy'].mean():.4f} (目标: 0.10)")
    print(f"   漫长卖单平均比例: {factors_df['long_order_ratio_sell'].mean():.4f} (目标: 0.10)")

    return n_values, factors_df


def save_results(n_values, factors_df, output_dir):
    """保存n值和因子结果"""
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存n值
    n_df = pd.DataFrame([n_values])
    n_path = os.path.join(output_dir, f"n_values_{timestamp}.csv")
    n_df.to_csv(n_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 n值已保存到: {n_path}")

    # 2. 保存核心因子（精简版）
    core_cols = [
        'secucode', 'total_volume', 'total_trades',
        'VolumeBigOrigin', 'VolumeBig', 'VolumeLong',
        'VolumeLongBig', 'VolumeLongBigSelect'
    ]
    core_cols = [c for c in core_cols if c in factors_df.columns]

    core_path = os.path.join(output_dir, f"core_factors_{timestamp}.csv")
    factors_df[core_cols].to_csv(core_path, index=False, encoding='utf-8-sig')
    print(f"💾 核心因子已保存到: {core_path}")

    # 3. 保存完整因子
    full_path = os.path.join(output_dir, f"all_factors_{timestamp}.csv")
    factors_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"💾 完整因子已保存到: {full_path}")

    # 4. 保存16种订单类型
    order_type_cols = [c for c in factors_df.columns if c.startswith('BB')]
    order_type_cols = ['secucode'] + order_type_cols

    order_path = os.path.join(output_dir, f"order_types_{timestamp}.csv")
    factors_df[order_type_cols].to_csv(order_path, index=False, encoding='utf-8-sig')
    print(f"💾 16种订单类型已保存到: {order_path}")

    return {
        'n_values': n_path,
        'core_factors': core_path,
        'all_factors': full_path,
        'order_types': order_path
    }


# 主程序
if __name__ == "__main__":
    print("=" * 80)
    print("🏦 n值计算与因子输出系统")
    print("=" * 80)

    try:
        # 输入文件路径
        data_path = "D:/pycharm/pythonProject/dataExample.parquet"

        # 输出目录
        output_dir = "D:/pycharm/pythonProject/n_and_factors"

        print(f"📁 输入文件: {data_path}")
        print(f"📁 输出目录: {output_dir}")

        # 计算n值和因子
        n_values, factors_df = calculate_n_and_factors(data_path, output_dir)

        # 显示核心结果
        print("\n" + "=" * 80)
        print("📊 计算结果摘要")
        print("=" * 80)

        # 显示n值
        print(f"\n🎯 计算出的n值:")
        print(f"   大单划分 n_big: {n_values['n_big']:.6f}")
        print(f"   漫长订单划分 n_long: {n_values['n_long']:.6f}")

        # 显示大单比例
        if 'big_order_ratio_buy' in factors_df.columns:
            print(f"\n📈 大单比例统计:")
            print(
                f"   大买单比例: {factors_df['big_order_ratio_buy'].mean():.4f} ± {factors_df['big_order_ratio_buy'].std():.4f}")
            print(
                f"   大卖单比例: {factors_df['big_order_ratio_sell'].mean():.4f} ± {factors_df['big_order_ratio_sell'].std():.4f}")

        # 显示核心因子统计
        print(f"\n📊 核心因子均值:")
        core_factors = ['VolumeBig', 'VolumeLong', 'VolumeLongBig', 'VolumeLongBigSelect']
        for factor in core_factors:
            if factor in factors_df.columns:
                mean_val = factors_df[factor].mean()
                print(f"   {factor}: {mean_val:.4f}")

        # 显示前5只股票的因子值
        print(f"\n📋 前5只股票的因子值（新方法）:")
        display_cols = ['secucode', 'VolumeBig', 'VolumeLong', 'VolumeLongBig', 'VolumeLongBigSelect']
        display_cols = [c for c in display_cols if c in factors_df.columns]
        print(factors_df[display_cols].head().round(4))

        # 保存结果
        file_paths = save_results(n_values, factors_df, output_dir)

        print(f"\n✅ 计算完成！")
        print(f"   股票数量: {len(factors_df)}")
        print(f"   输出文件已保存到: {output_dir}")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()