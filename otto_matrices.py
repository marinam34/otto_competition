import os
import gc
import polars as pl
from collections import defaultdict
from tqdm.auto import tqdm


TRAIN_CV_DIR = './data_parquet/train_cv/'
files = sorted([os.path.join(TRAIN_CV_DIR, f) for f in os.listdir(TRAIN_CV_DIR)])

def build_buy2buy_matrix():
    print("Шаг 1: Строим граф Buy2Buy (Совместные покупки и корзины)...")
    matrix = defaultdict(lambda: defaultdict(int))
    for file_name in tqdm(files, desc="Обработка чанков Buy2Buy"):
        df = pl.read_parquet(file_name)
        df_buys = df.filter(pl.col("type").is_in([1, 2]))
        df_buys = df_buys.sort(["session", "ts"])
        sessions = df_buys.group_by("session").agg(pl.col("aid").alias("items"))
        for items in sessions["items"].to_list():
            unique_items = list(set(items))
            if len(unique_items) < 2:
                continue
            for i in range(len(unique_items)):
                for j in range(i + 1, len(unique_items)):
                    item_a = unique_items[i]
                    item_b = unique_items[j]
                    matrix[item_a][item_b] += 1
                    matrix[item_b][item_a] += 1
    print(f"Граф связей построен! Уникальных товаров с покупками: {len(matrix)}")
    
    print("Отбираем ТОП-20 самых прочных связей...")
    top_20_matrix = {}
    for item, connections in list(matrix.items()):
        sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:20]
        top_20_matrix[item] = [x[0] for x in sorted_connections]
        del matrix[item]
    gc.collect()
    return top_20_matrix

def build_click2click_matrix():
    print("\nШаг 2: Строим граф Click2Click (Клики к Кликам)...")
    matrix = defaultdict(lambda: defaultdict(float))
    DAY_MS = 24 * 60 * 60 * 1000
    for file_name in tqdm(files, desc="Обработка чанков Click2Click"):
        df = pl.read_parquet(file_name)
        df_clicks = df.filter(pl.col("type") == 0)
        df_clicks = df_clicks.sort(["session", "ts"])
        df_clicks = df_clicks.group_by("session").tail(30)
        sessions = df_clicks.group_by("session").agg([
            pl.col("aid").alias("items"),
            pl.col("ts").alias("timestamps")
        ])
        for items, timestamps in zip(sessions["items"].to_list(), sessions["timestamps"].to_list()):
            if len(items) < 2:
                continue
            min_ts = timestamps[0]
            max_ts = timestamps[-1]
            session_duration = max_ts - min_ts
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item_a = items[i]
                    item_b = items[j]
                    if item_a == item_b:
                        continue
                    time_diff = timestamps[j] - timestamps[i]
                    if time_diff > DAY_MS:
                        continue
                    if session_duration > 0:
                        weight = 1.0 + 3.0 * (timestamps[j] - min_ts) / session_duration
                    else:
                        weight = 1.0
                    matrix[item_a][item_b] += weight
                    matrix[item_b][item_a] += weight
    print(f"Граф Click2Click построен! Уникальных товаров с кликами: {len(matrix)}")
    print("Отбираем ТОП-20 самых прочных связей для кликов...")
    top_20_matrix = {}
    for item, connections in list(matrix.items()):
        sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:20]
        top_20_matrix[item] = [x[0] for x in sorted_connections]
        del matrix[item]
    gc.collect()
    return top_20_matrix

def build_click2buy_matrix():
    print("\nШаг 3: Строим граф Click2Buy (Клики к Корзинам/Заказам)...")
    matrix = defaultdict(lambda: defaultdict(float))
    DAY_MS = 24 * 60 * 60 * 1000
    for file_name in tqdm(files, desc="Обработка чанков Click2Buy"):
        df = pl.read_parquet(file_name)
        df_all = df.sort(["session", "ts"])
        df_all = df_all.group_by("session").tail(30)
        sessions = df_all.group_by("session").agg([
            pl.col("aid").alias("items"),
            pl.col("ts").alias("timestamps"),
            pl.col("type").alias("types")
        ])
        for items, timestamps, types in zip(sessions["items"].to_list(), 
                                            sessions["timestamps"].to_list(), 
                                            sessions["types"].to_list()):
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(len(items)):
                    if i == j or items[i] == items[j]:
                        continue
                    item_a = items[i]
                    item_b = items[j]
                    if types[j] == 0:
                        continue
                    time_diff = abs(timestamps[j] - timestamps[i])
                    if time_diff > DAY_MS:
                        continue
                    weight = 1.0 + 3.0 * (1.0 - time_diff / DAY_MS)
                    matrix[item_a][item_b] += weight
                    matrix[item_b][item_a] += weight
    print(f"Граф Click2Buy построен! Уникальных товаров с кликами/покупками: {len(matrix)}")
    print("Отбираем ТОП-20 самых прочных связей для Click2Buy...")
    top_20_matrix = {}
    for item, connections in list(matrix.items()):
        sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:20]
        top_20_matrix[item] = [x[0] for x in sorted_connections]
        del matrix[item]
    gc.collect()
    return top_20_matrix

if __name__ == "__main__":
    if not os.path.exists("top_20_buy2buy.parquet"):
        buy2buy = build_buy2buy_matrix()
        df_top20_buy = pl.DataFrame({
            "aid": list(buy2buy.keys()),
            "candidates": list(buy2buy.values())
        })
        df_top20_buy.write_parquet("top_20_buy2buy.parquet")
        print("Файл top_20_buy2buy.parquet готов.")
        del buy2buy, df_top20_buy
        gc.collect()
    else:
        print("Файл top_20_buy2buy.parquet уже существует, пропускаем Шаг 1.")
        
    if not os.path.exists("top_20_click2click.parquet"):
        click2click = build_click2click_matrix()
        df_top20_click = pl.DataFrame({
            "aid": list(click2click.keys()),
            "candidates": list(click2click.values())
        })
        df_top20_click.write_parquet("top_20_click2click.parquet")
        print("Файл top_20_click2click.parquet готов.")
        del click2click, df_top20_click
        gc.collect()
    else:
        print("Файл top_20_click2click.parquet уже существует, пропускаем Шаг 2.")
        
    if not os.path.exists("top_20_click2buy.parquet"):
        click2buy = build_click2buy_matrix()
        df_top20_c2b = pl.DataFrame({
            "aid": list(click2buy.keys()),
            "candidates": list(click2buy.values())
        })
        df_top20_c2b.write_parquet("top_20_click2buy.parquet")
        print("Файл top_20_click2buy.parquet готов.")
        del click2buy, df_top20_c2b
        gc.collect()
    else:
        print("Файл top_20_click2buy.parquet уже существует, пропускаем Шаг 3.")
