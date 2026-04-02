import pandas as pd

ps = pd.read_csv("box_gate_cell_ids.txt")
ps['celltype'] = 'ESC(D3)'
ps.to_csv("box_gate_cell_ids_celltype.csv")

