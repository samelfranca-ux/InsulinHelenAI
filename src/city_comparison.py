import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CARREGAR DADOS
# ==========================

df = pd.read_csv(
    "data/cities_pollution_diabetes.csv"
)

# ==========================
# PM2.5 COMPARATIVO
# ==========================

plt.figure(figsize=(10,6))

for cidade in df["Cidade"].unique():

    dados = df[
        df["Cidade"] == cidade
    ]

    plt.plot(
        dados["Ano"],
        dados["PM25"],
        marker="o",
        linewidth=3,
        label=cidade
    )

plt.title(
    "Comparação PM2.5 entre Cidades"
)

plt.xlabel("Ano")
plt.ylabel("PM2.5")

plt.legend()
plt.grid(True)

plt.savefig(
    "graphs/comparativo_pm25.png"
)

plt.show()

# ==========================
# DIABETES COMPARATIVO
# ==========================

plt.figure(figsize=(10,6))

for cidade in df["Cidade"].unique():

    dados = df[
        df["Cidade"] == cidade
    ]

    plt.plot(
        dados["Ano"],
        dados["Diabetes"],
        marker="o",
        linewidth=3,
        label=cidade
    )

plt.title(
    "Comparação Diabetes entre Cidades"
)

plt.xlabel("Ano")
plt.ylabel("Diabetes (%)")

plt.legend()
plt.grid(True)

plt.savefig(
    "graphs/comparativo_diabetes.png"
)

plt.show()

print(
    "Gráficos comparativos gerados!"
)