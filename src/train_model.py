import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Ler dados
df = pd.read_csv(
    "data/pollution_diabetes.csv"
)

# Variável de entrada
X = df[
    [
        "PM25",
        "PM10",
        "NO2",
        "O3",
        "Temperatura",
        "Umidade"
    ]
]

# Variável alvo
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modelo
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Treinamento
model.fit(
    X_train,
    y_train
)

print("Modelo treinado com sucesso!")

y_pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

print(f"R² = {r2:.4f}")
print(f"MAE = {mae:.4f}")

# Nova previsão
novo_valor = pd.DataFrame({
    "PM25": [25],
    "PM10": [40],
    "NO2": [30],
    "O3": [60],
    "Temperatura": [30],
    "Umidade": [60]
})

previsao = model.predict(novo_valor)

print(
    f"Taxa prevista de diabetes: {previsao[0]:.2f}%"
)

importancias = model.feature_importances_

colunas = X.columns

print("\nImportância das variáveis:")

for coluna, importancia in zip(colunas, importancias):
    print(
        f"{coluna}: {importancia:.4f}"
    )

plt.figure(figsize=(10,5))

plt.bar(
    colunas,
    importancias
)

plt.title(
    "Importância das Variáveis Ambientais"
)

plt.ylabel("Importância")

plt.grid(axis="y")

plt.show()



