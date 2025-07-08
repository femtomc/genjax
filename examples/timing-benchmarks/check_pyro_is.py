import pandas as pd
import json
from pathlib import Path

# Load Pyro IS data
pyro_is = []
for n in [1000, 5000, 10000]:
    f = Path(f'data/curvefit/pyro/is_n{n}.json')
    if f.exists():
        with open(f) as file:
            data = json.load(file)
            print(f'Pyro IS n={n}: mean_time={data.get("mean_time", "N/A")}')
    else:
        print(f'Pyro IS n={n}: FILE NOT FOUND')