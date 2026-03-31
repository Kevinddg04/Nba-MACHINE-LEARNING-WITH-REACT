import json
with open('Machine_Learning_para_la_NBA_con_CatBoost (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][:10]):
    if cell['cell_type'] == 'code':
        print(f"--- CELL {i} ---")
        print("".join(cell['source']))
