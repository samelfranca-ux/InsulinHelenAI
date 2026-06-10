from Bio.PDB import PDBParser
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================
# TREINAMENTO IA
# =====================================

df = pd.read_csv(
    "data/pollution_diabetes.csv"
)

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

y = df["Diabetes"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# =====================================
# INSULINA REAL
# =====================================

parser = PDBParser()

estrutura = parser.get_structure(
    "INSULINA",
    "data/pdb4ins.ent"
)

x = []
y = []
z = []

for atomo in estrutura.get_atoms():

    if atomo.get_name() == "CA":

        coord = atomo.get_coord()

        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])

x = np.array(x)
y = np.array(y)
z = np.array(z)

# =====================================
# CENÁRIOS
# =====================================

cenarios = [
    {
        "titulo": "Baixa Poluição",
        "PM25": 10,
        "PM10": 20,
        "NO2": 15,
        "O3": 40,
        "Temperatura": 28,
        "Umidade": 60
    },
    {
        "titulo": "Poluição Moderada",
        "PM25": 30,
        "PM10": 50,
        "NO2": 35,
        "O3": 70,
        "Temperatura": 30,
        "Umidade": 55
    },
    {
        "titulo": "Alta Poluição",
        "PM25": 60,
        "PM10": 90,
        "NO2": 60,
        "O3": 100,
        "Temperatura": 32,
        "Umidade": 50
    }
]

# =====================================
# FIGURA
# =====================================

fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "scene"},
            {"type": "scene"},
            {"type": "scene"}]],
    subplot_titles=[
        "Baixa Poluição",
        "Poluição Moderada",
        "Alta Poluição"
    ]
)

for i, cenario in enumerate(cenarios):

    entrada = pd.DataFrame({
        "PM25": [cenario["PM25"]],
        "PM10": [cenario["PM10"]],
        "NO2": [cenario["NO2"]],
        "O3": [cenario["O3"]],
        "Temperatura": [cenario["Temperatura"]],
        "Umidade": [cenario["Umidade"]]
    })

    risco = model.predict(
        entrada
    )[0]

    intensidade = risco / 10

    np.random.seed(42)

    ruido = np.random.normal(
        0,
        intensidade,
        (len(x), 3)
    )

    x_mod = x + ruido[:,0]
    y_mod = y + ruido[:,1]
    z_mod = z + ruido[:,2]

    deslocamento = np.sqrt(
        (x_mod - x)**2 +
        (y_mod - y)**2 +
        (z_mod - z)**2
    )

    fig.add_trace(

        go.Scatter3d(

            x=x,
            y=y,
            z=z,

            mode="lines+markers",

            line=dict(
                width=8,
                color="lightgray"
            ),

            marker=dict(
                size=7,
                color=deslocamento,
                colorscale="Turbo",
                showscale=(i == 2),
                colorbar=dict(
                    title="Impacto"
                )
            ),

            name=cenario["titulo"]

        ),

        row=1,
        col=i+1

    )

# =====================================
# LAYOUT
# =====================================

fig.update_layout(

    template="plotly_white",

    width=1800,
    height=700,

    title=(
        "InsulinAirML<br>"
        "Comparação Estrutural sob Diferentes Níveis de Poluição"
    )
)

fig.write_html(
    "insulin_scenarios.html"
)

print(
    "Arquivo salvo: insulin_scenarios.html"
)