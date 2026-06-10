import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ==========================
# CARREGAR DADOS
# ==========================

df = pd.read_csv(
    "data/cities_pollution_diabetes.csv"
)

# ==========================
# CORRELAÇÃO
# ==========================

correlacao, pvalor = pearsonr(
    df["PM25"],
    df["Diabetes"]
)

print("\nRESULTADOS ESTATÍSTICOS")

print(
    f"Correlação de Pearson: {correlacao:.4f}"
)

print(
    f"P-valor: {pvalor:.6f}"
)

# ==========================
# REGRESSÃO
# ==========================

X = df[["PM25"]]

y = df["Diabetes"]

modelo = LinearRegression()

modelo.fit(X, y)

r2 = modelo.score(X, y)

print(
    f"R²: {r2:.4f}"
)

# ==========================
# GRÁFICO
# ==========================

plt.figure(figsize=(10,6))

plt.scatter(
    df["PM25"],
    df["Diabetes"],
    s=100
)

x_linha = df["PM25"]

y_linha = modelo.predict(
    X
)

plt.plot(
    x_linha,
    y_linha,
    linewidth=3
)

plt.title(
    "PM2.5 vs Diabetes"
)

plt.xlabel("PM2.5")

plt.ylabel("Diabetes (%)")

plt.grid(True)

plt.savefig(
    "graphs/pm25_diabetes_regression.png"
)

plt.show()