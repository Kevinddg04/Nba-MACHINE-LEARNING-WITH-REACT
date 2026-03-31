import json
with open('Machine_Learning_para_la_NBA_con_CatBoost (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        print(f"--- CELL {i} ---")
        if "win_streak" in source or "shift" in source or "rolling" in source or "predict" in source or "features" in source.lower():
            print(source)
        elif "2024" in source or "2025" in source or "temporada" in source:
            print(source)

