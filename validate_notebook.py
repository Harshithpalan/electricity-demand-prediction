from pathlib import Path
import json
path = Path('notebooks/OPSD_exploration.ipynb')
text = path.read_text(encoding='utf-8')
nb = json.loads(text)
print('valid json, cells:', len(nb.get('cells', [])))
