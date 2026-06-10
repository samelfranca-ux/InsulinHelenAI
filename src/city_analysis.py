import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CARREGAR DATASET
# ==========================

df = pd.read_csv(
    "data/cities_pollution_diabetes.csv"
)

cidades = [
    "Fortaleza",
    "Sobral",
    "Sao Paulo"
]

# ==========================
# GERAR GRÁFICOS
# ==========================

for cidade in cidades:

    dados = df[
        df["Cidade"] == cidade
    ]

    # ---------------------------------
    # 1 - PM25 vs Ano
    # ---------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        dados["Ano"],
        dados["PM25"],
        marker="o"
    )

    plt.title(
        f"{cidade} - PM2.5 ao Longo do Tempo"
    )

    plt.xlabel("Ano")
    plt.ylabel("PM2.5")

    plt.grid(True)

    plt.savefig(
        f"graphs/{cidade}_pm25.png"
    )

    plt.close()

    # ---------------------------------
    # 2 - Diabetes vs Ano
    # ---------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        dados["Ano"],
        dados["Diabetes"],
        marker="o"
    )

    plt.title(
        f"{cidade} - Diabetes ao Longo do Tempo"
    )

    plt.xlabel("Ano")
    plt.ylabel("Diabetes (%)")

    plt.grid(True)

    plt.savefig(
        f"graphs/{cidade}_diabetes.png"
    )

    plt.close()

    # ---------------------------------
    # 3 - PM25 vs Diabetes
    # ---------------------------------

    plt.figure(figsize=(8,5))

    plt.scatter(
        dados["PM25"],
        dados["Diabetes"],
        s=80
    )

    plt.title(
        f"{cidade} - PM2.5 x Diabetes"
    )

    plt.xlabel("PM2.5")
    plt.ylabel("Diabetes (%)")

    plt.grid(True)

    plt.savefig(
        f"graphs/{cidade}_correlacao.png"
    )

    plt.close()

print(
    "9 gráficos gerados com sucesso!"
)