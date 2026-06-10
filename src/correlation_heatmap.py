import pandas as pd
import matplotlib.pyplot as plt

# se não tiver seaborn:
# pip install seaborn

import seaborn as sns

df = pd.read_csv(
    "data/cities_pollution_diabetes.csv"
)

colunas = [
    "PM25",
    "PM10",
    "NO2",
    "O3",
    "Temperatura",
    "Umidade",
    "Diabetes"
]

corr = df[colunas].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Matriz de Correlação - InsulinAirML"
)

plt.tight_layout()

plt.savefig(
    "graphs/correlation_heatmap.png",
    dpi=300
)

plt.show()