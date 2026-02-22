import os
import polars as pl
from tqdm.auto import tqdm

print("Загрузка 3-х матриц со-визитации в память")
try:
    dict_clicks = pl.read_parquet('top_20_click2click.parquet').to_pandas().set_index('aid')['candidates'].to_dict()
    dict_buy2buy = pl.read_parquet('top_20_buy2buy.parquet').to_pandas().set_index('aid')['candidates'].to_dict()
    dict_click2buy = pl.read_parquet('top_20_click2buy.parquet').to_pandas().set_index('aid')['candidates'].to_dict()
except Exception as e:
    print("Ошибка загрузки матриц.")
    raise e
print("Матрицы загружены")

def suggest_clicks(history_aids):
    unique_aids = list(dict.fromkeys(history_aids[::-1]))
    if len(unique_aids) >= 20:
        return unique_aids[:20]
    
    candidates = unique_aids.copy()
    for aid in unique_aids:
        if aid in dict_clicks:
            for neighbor in dict_clicks[aid]:
                if neighbor not in candidates:
                    candidates.append(neighbor)
                if len(candidates) >= 20:
                    return candidates
    return candidates[:20]

def suggest_buys(history_aids, history_types):
    unique_aids = list(dict.fromkeys(history_aids[::-1]))
    if len(unique_aids) >= 20:
        return unique_aids[:20]
        
    candidates = unique_aids.copy()
    for aid in unique_aids:
        if aid in dict_click2buy:
            for neighbor in dict_click2buy[aid]:
                if neighbor not in candidates:
                    candidates.append(neighbor)
                if len(candidates) >= 20:
                    return candidates

    buys_in_history = [aid for aid, typ in zip(history_aids, history_types) if typ in [1, 2]]
    unique_buys = list(dict.fromkeys(buys_in_history[::-1]))
    for aid in unique_buys:
        if aid in dict_buy2buy:
            for neighbor in dict_buy2buy[aid]:
                if neighbor not in candidates:
                    candidates.append(neighbor)
                if len(candidates) >= 20:
                    return candidates
    return candidates[:20]

TEST_DIR = './data_parquet/test/'

def generate_submission():
    print("\n2. Генерация submission_baseline.csv по тестовому набору Kaggle")
    test_files = sorted(os.listdir(TEST_DIR))
    
    with open('submission_baseline.csv', 'w') as f:
        f.write("session_type,labels\n")
        
        for file_name in tqdm(test_files, desc="Обработка файлов test"):
            test_df = pl.read_parquet(os.path.join(TEST_DIR, file_name))
            if len(test_df) == 0: continue
            
            test_sessions = test_df.sort(["session", "ts"]).group_by("session").agg([
                pl.col("aid").alias("aids"),
                pl.col("type").alias("types")
            ]).to_pandas()
            
            for _, row in test_sessions.iterrows():
                session = row['session']
                history_aids = row['aids'].tolist()
                history_types = row['types'].tolist()
                
                pred_clicks = suggest_clicks(history_aids)
                pred_carts = suggest_buys(history_aids, history_types)
                pred_orders = pred_carts 
                
                str_clicks = ' '.join(map(str, pred_clicks))
                str_carts  = ' '.join(map(str, pred_carts))
                str_orders = ' '.join(map(str, pred_orders))
                
                f.write(f"{session}_clicks,{str_clicks}\n")
                f.write(f"{session}_carts,{str_carts}\n")
                f.write(f"{session}_orders,{str_orders}\n")

    print("\n submission_baseline.csv успешно сгенерирован")

if __name__ == "__main__":
    generate_submission()
