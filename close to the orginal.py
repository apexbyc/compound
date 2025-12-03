"""
修改版：寻找n值使得新划分的大单/漫长订单集合尽可能接近原10%分位数划分
然后基于这个新划分计算所有因子
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')


def find_optimal_n_for_stock(data, order_type='volume', target_percentile=0.9):
    """
    为单只股票寻找最优n，使得均值+n*标准差划分与原分位数划分最相似

    参数：
    data: 订单数据（成交量或成交时长）
    order_type: 'volume'（大单）或'duration'（漫长订单）
    target_percentile: 原划分的分位数（默认为0.9，即90%分位数）
    """
    if len(data) < 10:
        return 1.2816  # 默认值

    # 原划分阈值（90%分位数）
    original_threshold = np.percentile(data, target_percentile * 100)

    # 原划分下的大单标签
    original_labels = data > original_threshold

    # 目标函数：最大化Jaccard相似度
    def jaccard_similarity(n):
        # 新划分阈值
        new_threshold = np.mean(data) + n * np.std(data)
        # 新划分标签
        new_labels = data > new_threshold
        # 计算Jaccard相似度
        intersection = np.sum(original_labels & new_labels)
        union = np.sum(original_labels | new_labels)
        if union == 0:
            return 0
        return intersection / union

    # 负相似度（因为最小化）
    def objective(n):
        return -jaccard_similarity(n)

    # 搜索n的范围（-3到10，但通常为正）
    bounds = (-3, 10)

    try:
        result = minimize_scalar(objective, bounds=bounds, method='bounded')
        optimal_n = result.x

        # 验证最优n下的相似度
        similarity = jaccard_similarity(optimal_n)

        return optimal_n, similarity
    except:
        # 如果优化失败，使用简单方法
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            n_simple = (original_threshold - mean_val) / std_val
            return n_simple, jaccard_similarity(n_simple)
        else:
            return 1.2816, 0


def calculate_factors_with_optimal_n(data_path, output_dir=None):
    """
    1. 为每只股票找到最优n，使得新划分接近原10%分位数划分
    2. 计算所有股票的n值统计
    3. 使用最优n（全局）计算所有因子
    """
    print("=" * 80)
    print("🎯 寻找最优n值并计算因子（接近原10%分位数划分）")
    print("=" * 80)

    # 1. 加载数据
    print("1. 加载数据...")
    df = pd.read_parquet(data_path)

    # 2. 时间处理
    df['TradeTime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['Time'].astype(str).str[-15:])
    df_continuous = df[~df['TradeTime'].dt.time.between(time(9, 15), time(9, 30))].copy()

    # 3. 收集所有股票的n值
    print("2. 为每只股票计算最优n值...")

    all_stocks = df_continuous['secucode'].unique()
    n_big_values = []
    n_long_values = []
    similarity_big_values = []
    similarity_long_values = []

    # 进度计数器
    progress_count = 0
    total_stocks = min(len(all_stocks), 50)  # 最多处理50只股票以减少计算时间

    for stock in all_stocks[:total_stocks]:
        progress_count += 1
        if progress_count % 10 == 0:
            print(f"   进度: {progress_count}/{total_stocks}")

        stock_data = df_continuous[df_continuous['secucode'] == stock].copy()

        if len(stock_data) < 10:
            continue

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

        # 合并买单和卖单数据
        all_volumes = np.concatenate([buy_orders['buy_volume'].values, sell_orders['sell_volume'].values])
        all_durations = np.concatenate([buy_orders['buy_duration'].values, sell_orders['sell_duration'].values])

        # 过滤异常值
        all_volumes = all_volumes[~np.isnan(all_volumes) & ~np.isinf(all_volumes)]
        all_durations = all_durations[~np.isnan(all_durations) & ~np.isinf(all_durations)]

        if len(all_volumes) >= 10:
            n_big, sim_big = find_optimal_n_for_stock(all_volumes, 'volume', 0.9)
            n_big_values.append(n_big)
            similarity_big_values.append(sim_big)

        if len(all_durations) >= 10:
            n_long, sim_long = find_optimal_n_for_stock(all_durations, 'duration', 0.9)
            n_long_values.append(n_long)
            similarity_long_values.append(sim_long)

    # 4. 计算全局n值（使用中位数，更稳健）
    if len(n_big_values) > 0:
        global_n_big = np.median(n_big_values)
        global_n_long = np.median(n_long_values) if len(n_long_values) > 0 else 1.2816
    else:
        global_n_big = 1.2816
        global_n_long = 1.2816

    print("\n3. n值计算结果:")
    print("=" * 50)
    print(f"   大单n值统计:")
    print(f"     中位数: {global_n_big:.6f}")
    if n_big_values:
        print(f"     均值: {np.mean(n_big_values):.6f}")
        print(f"     标准差: {np.std(n_big_values):.6f}")
        print(f"     范围: [{np.min(n_big_values):.6f}, {np.max(n_big_values):.6f}]")

    print(f"\n   漫长订单n值统计:")
    print(f"     中位数: {global_n_long:.6f}")
    if n_long_values:
        print(f"     均值: {np.mean(n_long_values):.6f}")
        print(f"     标准差: {np.std(n_long_values):.6f}")
        print(f"     范围: [{np.min(n_long_values):.6f}, {np.max(n_long_values):.6f}]")

    print(f"\n   划分相似度统计:")
    if similarity_big_values:
        print(f"     大单划分平均Jaccard相似度: {np.mean(similarity_big_values):.4f}")
    if similarity_long_values:
        print(f"     漫长订单划分平均Jaccard相似度: {np.mean(similarity_long_values):.4f}")

    # 5. 使用全局n值计算所有股票的因子
    print("\n4. 使用全局n值计算所有股票的因子...")

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

        # 计算阈值（使用全局n）
        if len(buy_orders) > 0:
            buy_big_threshold = np.mean(buy_orders['buy_volume']) + global_n_big * np.std(buy_orders['buy_volume'])
            buy_long_threshold = np.mean(buy_orders['buy_duration']) + global_n_long * np.std(
                buy_orders['buy_duration'])
        else:
            buy_big_threshold = buy_long_threshold = 0

        if len(sell_orders) > 0:
            sell_big_threshold = np.mean(sell_orders['sell_volume']) + global_n_big * np.std(sell_orders['sell_volume'])
            sell_long_threshold = np.mean(sell_orders['sell_duration']) + global_n_long * np.std(
                sell_orders['sell_duration'])
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

            # 阈值信息
            'buy_big_threshold': buy_big_threshold,
            'sell_big_threshold': sell_big_threshold,
            'buy_long_threshold': buy_long_threshold,
            'sell_long_threshold': sell_long_threshold,

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

            # 大单比例（新划分下）
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

    # 6. 验证数学关系
    print("\n5. 验证因子计算正确性...")

    if len(factors_df) == 0:
        print("❌ 没有计算到任何因子")
        return None, pd.DataFrame()

    # 验证复合因子公式
    factors_df['验证_复合因子'] = factors_df['VolumeBig'] + factors_df['VolumeLong']
    diff_composite = (factors_df['VolumeLongBig'] - factors_df['验证_复合因子']).abs().max()
    print(f"   ✅ 复合因子验证误差: {diff_composite:.10f}")

    # 验证传统大单因子公式
    factors_df['验证_传统大单'] = (
            factors_df['big_buy_non_big_sell'] +
            factors_df['non_big_buy_big_sell'] +
            2 * factors_df['big_buy_big_sell']
    )
    diff_origin = (factors_df['VolumeBigOrigin'] - factors_df['验证_传统大单']).abs().max()
    print(f"   ✅ 传统大单因子验证误差: {diff_origin:.10f}")

    # 验证16种订单类型之和为1
    order_type_cols = [c for c in factors_df.columns if c.startswith('BB')]
    factors_df['订单类型总和'] = factors_df[order_type_cols].sum(axis=1)
    diff_order_types = (factors_df['订单类型总和'] - 1).abs().max()
    print(f"   ✅ 16种订单类型总和验证: {diff_order_types:.10f}")

    # 7. 统计新划分下的大单比例
    print("\n6. 新划分下的大单/漫长订单比例统计:")
    print(
        f"   大买单平均比例: {factors_df['big_order_ratio_buy'].mean():.4f} ± {factors_df['big_order_ratio_buy'].std():.4f}")
    print(
        f"   大卖单平均比例: {factors_df['big_order_ratio_sell'].mean():.4f} ± {factors_df['big_order_ratio_sell'].std():.4f}")
    print(
        f"   漫长买单平均比例: {factors_df['long_order_ratio_buy'].mean():.4f} ± {factors_df['long_order_ratio_buy'].std():.4f}")
    print(
        f"   漫长卖单平均比例: {factors_df['long_order_ratio_sell'].mean():.4f} ± {factors_df['long_order_ratio_sell'].std():.4f}")

    # 8. 保存全局n值
    global_n_values = {
        'global_n_big': global_n_big,
        'global_n_long': global_n_long,
        'n_big_values_count': len(n_big_values),
        'n_long_values_count': len(n_long_values),
        'avg_similarity_big': np.mean(similarity_big_values) if similarity_big_values else 0,
        'avg_similarity_long': np.mean(similarity_long_values) if similarity_long_values else 0,
    }

    return global_n_values, factors_df


def save_results(global_n_values, factors_df, output_dir):
    """保存n值和因子结果"""
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存n值
    n_df = pd.DataFrame([global_n_values])
    n_path = os.path.join(output_dir, f"optimal_n_values_{timestamp}.csv")
    n_df.to_csv(n_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 最优n值已保存到: {n_path}")

    # 2. 保存详细n值统计（如果可用）
    if 'n_big_values' in global_n_values:
        n_detailed = {
            'metric': ['中位数', '均值', '标准差', '最小值', '最大值'],
            'n_big': [
                np.median(global_n_values['n_big_values']),
                np.mean(global_n_values['n_big_values']),
                np.std(global_n_values['n_big_values']),
                np.min(global_n_values['n_big_values']),
                np.max(global_n_values['n_big_values'])
            ],
            'n_long': [
                np.median(global_n_values['n_long_values']),
                np.mean(global_n_values['n_long_values']),
                np.std(global_n_values['n_long_values']),
                np.min(global_n_values['n_long_values']),
                np.max(global_n_values['n_long_values'])
            ]
        }
        n_detailed_df = pd.DataFrame(n_detailed)
        n_detailed_path = n_path.replace('.csv', '_detailed.csv')
        n_detailed_df.to_csv(n_detailed_path, index=False, encoding='utf-8-sig')
        print(f"💾 详细n值统计已保存到: {n_detailed_path}")

    # 3. 保存核心因子
    core_cols = [
        'secucode', 'total_volume', 'total_trades',
        'VolumeBigOrigin', 'VolumeBig', 'VolumeLong',
        'VolumeLongBig', 'VolumeLongBigSelect',
        'big_order_ratio_buy', 'big_order_ratio_sell'
    ]
    core_cols = [c for c in core_cols if c in factors_df.columns]

    core_path = os.path.join(output_dir, f"core_factors_optimal_n_{timestamp}.csv")
    factors_df[core_cols].to_csv(core_path, index=False, encoding='utf-8-sig')
    print(f"💾 核心因子已保存到: {core_path}")

    # 4. 保存完整因子
    full_path = os.path.join(output_dir, f"all_factors_optimal_n_{timestamp}.csv")
    factors_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"💾 完整因子已保存到: {full_path}")

    # 5. 保存16种订单类型
    order_type_cols = [c for c in factors_df.columns if c.startswith('BB')]
    order_type_cols = ['secucode'] + order_type_cols

    order_path = os.path.join(output_dir, f"order_types_optimal_n_{timestamp}.csv")
    factors_df[order_type_cols].to_csv(order_path, index=False, encoding='utf-8-sig')
    print(f"💾 16种订单类型已保存到: {order_path}")

    return {
        'n_values': n_path,
        'core_factors': core_path,
        'all_factors': full_path,
        'order_types': order_path
    }


def compare_with_original_factors(factors_df_new, factors_df_original=None):
    """
    比较新划分因子与原划分因子的差异
    如果没有原划分因子数据，只显示新划分的结果
    """
    print("\n" + "=" * 80)
    print("📊 新划分因子结果摘要")
    print("=" * 80)

    print(f"股票数量: {len(factors_df_new)}")

    # 核心因子统计
    core_factors = ['VolumeBig', 'VolumeLong', 'VolumeLongBig', 'VolumeLongBigSelect']

    print(f"\n核心因子统计（新划分）:")
    for factor in core_factors:
        if factor in factors_df_new.columns:
            mean_val = factors_df_new[factor].mean()
            std_val = factors_df_new[factor].std()
            min_val = factors_df_new[factor].min()
            max_val = factors_df_new[factor].max()
            print(f"  {factor}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 范围=[{min_val:.4f}, {max_val:.4f}]")

    # 显示前5只股票的因子值
    print(f"\n📋 前5只股票的因子值（新划分）:")
    display_cols = ['secucode', 'VolumeBig', 'VolumeLong', 'VolumeLongBig', 'VolumeLongBigSelect']
    display_cols = [c for c in display_cols if c in factors_df_new.columns]
    print(factors_df_new[display_cols].head().round(4))

    return factors_df_new


# 主程序
if __name__ == "__main__":
    print("=" * 80)
    print("🏦 寻找最优n值并计算因子（接近原10%分位数划分）")
    print("=" * 80)

    try:
        # 输入文件路径
        data_path = "D:/pycharm/pythonProject/dataExample.parquet"

        # 输出目录
        output_dir = "D:/pycharm/pythonProject/optimal_n_factors"

        print(f"📁 输入文件: {data_path}")
        print(f"📁 输出目录: {output_dir}")

        # 计算最优n值和因子
        global_n_values, factors_df = calculate_factors_with_optimal_n(data_path, output_dir)

        if factors_df is not None and len(factors_df) > 0:
            # 显示关键结果
            print("\n" + "=" * 80)
            print("🎯 计算完成！")
            print("=" * 80)

            print(f"\n📊 全局最优n值:")
            print(f"   大单划分 n_big: {global_n_values['global_n_big']:.6f}")
            print(f"   漫长订单划分 n_long: {global_n_values['global_n_long']:.6f}")

            if 'avg_similarity_big' in global_n_values:
                print(f"   大单划分平均相似度: {global_n_values['avg_similarity_big']:.4f}")
                print(f"   漫长订单划分平均相似度: {global_n_values['avg_similarity_long']:.4f}")

            # 比较因子结果
            factors_df = compare_with_original_factors(factors_df)

            # 保存结果
            file_paths = save_results(global_n_values, factors_df, output_dir)

            print(f"\n✅ 所有计算完成！")
            print(f"   股票数量: {len(factors_df)}")
            print(f"   输出文件已保存到: {output_dir}")

        else:
            print("\n❌ 没有计算到任何因子，请检查数据格式")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()