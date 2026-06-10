import pandas as pd
import plotly.express as px

# ==========================
# CARREGAR DADOS
# ==========================

df = pd.read_csv(
    "data/cities_pollution_diabetes.csv"
)

# ==========================
# MÉDIAS POR CIDADE
# ==========================

resumo = df.groupby(
    "Cidade"
).agg({
    "PM25": "mean",
    "Diabetes": "mean"
}).reset_index()

# Coordenadas aproximadas

coordenadas = {
    "Fortaleza": (-3.7319, -38.5267),
    "Sobral": (-3.6891, -40.3488),
    "Sao Paulo": (-23.5505, -46.6333)
}

resumo["lat"] = resumo["Cidade"].map(
    lambda x: coordenadas[x][0]
)

resumo["lon"] = resumo["Cidade"].map(
    lambda x: coordenadas[x][1]
)

# ==========================
# MAPA
# ==========================

fig = px.scatter_geo(
    resumo,
    lat="lat",
    lon="lon",
    size="PM25",
    color="Diabetes",
    hover_name="Cidade",
    hover_data={
        "PM25": True,
        "Diabetes": True,
        "lat": False,
        "lon": False
    },
    title="InsulinAirML - Poluição e Diabetes por Cidade"
)

fig.update_geos(
    scope="south america"
)

fig.write_html(
    "city_map.html"
)

print(
    "Mapa salvo: city_map.html"
)