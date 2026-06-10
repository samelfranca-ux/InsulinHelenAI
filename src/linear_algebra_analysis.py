import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================
# 1 - LEITURA DO DATASET
# ==========================

df = pd.read_csv(
    "data/cities_pollution_diabetes.csv"
)

print("\n============================")
print("MATRIZ DE DADOS AMBIENTAIS X")
print("============================")

variaveis = [
    "PM25",
    "PM10",
    "NO2",
    "O3",
    "Temperatura",
    "Umidade"
]

X = df[variaveis].values

print(X)


# ==========================
# 2 - NORMALIZAÇÃO MANUAL
# ==========================

media = np.mean(X, axis=0)
desvio = np.std(X, axis=0)

Z = (X - media) / desvio


print("\n============================")
print("MATRIZ NORMALIZADA Z")
print("============================")

print(Z)


# ==========================
# 3 - MATRIZ DE COVARIÂNCIA
# ==========================

C = np.cov(
    Z,
    rowvar=False
)


print("\n============================")
print("MATRIZ DE COVARIÂNCIA C")
print("============================")

print(C)


# ==========================
# 4 - AUTOVALORES E AUTOVETORES
# ==========================

autovalores, autovetores = np.linalg.eig(C)


print("\n============================")
print("AUTOVALORES λ")
print("============================")

print(autovalores)


print("\n============================")
print("AUTOVETORES V")
print("============================")

print(autovetores)


# ==========================
# 5 - ORGANIZAÇÃO DOS COMPONENTES
# ==========================

ordem = np.argsort(
    autovalores
)[::-1]


autovalores = autovalores[ordem]
autovetores = autovetores[:, ordem]


# ==========================
# 6 - VARIÂNCIA EXPLICADA
# ==========================

variancia = (
    autovalores /
    np.sum(autovalores)
) * 100


print("\n============================")
print("VARIÂNCIA EXPLICADA")
print("============================")


for i, valor in enumerate(variancia):
    print(
        f"Componente {i+1}: {valor:.2f}%"
    )


# ==========================
# 7 - GRÁFICO DOS AUTOVALORES
# ==========================

plt.figure(figsize=(8, 5))

plt.bar(
    range(1, len(variancia)+1),
    variancia
)

plt.xlabel(
    "Componente Principal"
)

plt.ylabel(
    "Variância Explicada (%)"
)

plt.title(
    "Análise Espectral dos Dados Ambientais"
)

plt.grid()

plt.show()


# ==========================
# 8 - PCA MANUAL
# ==========================

V2 = autovetores[:, :2]

Y = np.dot(
    Z,
    V2
)


resultado = pd.DataFrame({
    "PC1": Y[:, 0],
    "PC2": Y[:, 1],
    "Cidade": df["Cidade"]
})


print("\n============================")
print("PROJEÇÃO NO ESPAÇO DOS AUTOVETORES")
print("============================")

print(resultado)


# ==========================
# 9 - MAPA DOS COMPONENTES
# ==========================

plt.figure(figsize=(10, 6))


for cidade in resultado["Cidade"].unique():

    dados = resultado[
        resultado["Cidade"] == cidade
    ]

    plt.scatter(
        dados["PC1"],
        dados["PC2"],
        s=80,
        label=cidade
    )


plt.xlabel(
    "Primeiro Autovetor (PC1)"
)

plt.ylabel(
    "Segundo Autovetor (PC2)"
)

plt.title(
    "Separação das Cidades no Espaço Vetorial"
)

plt.legend()

plt.grid()

plt.show()


# ==========================
# 10 - INTERPRETAÇÃO FINAL
# ==========================

principal = np.argmax(
    variancia
)


print("\n============================")
print("INTERPRETAÇÃO MATEMÁTICA")
print("============================")


print(
    f"O primeiro padrão dominante do sistema "
    f"ambiental explica {variancia[principal]:.2f}% "
    f"da variabilidade total dos dados."
)


print(
    "Os autovalores indicam a quantidade de informação "
    "contida em cada direção do espaço vetorial."
)


print(
    "Os autovetores representam as direções matemáticas "
    "onde os padrões de poluição se manifestam com maior intensidade."
)


print(
    "A projeção PCA permite comparar Fortaleza, Sobral e São Paulo "
    "em um espaço reduzido, preservando as características ambientais mais importantes."
)