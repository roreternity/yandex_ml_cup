# Сниппеты

## Seed

```python
import os
import random
import numpy as np

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
```

## Быстрый отчёт по колонкам

```python
def schema_report(df):
    return (
        df.dtypes.astype(str).to_frame("dtype")
        .assign(missing=df.isna().mean(), nunique=df.nunique(dropna=False))
        .sort_values(["missing", "nunique"], ascending=False)
    )
```

