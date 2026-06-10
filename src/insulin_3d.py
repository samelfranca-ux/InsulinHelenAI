from Bio.PDB import PDBParser
from collections import Counter
import plotly.graph_objects as go

# Arquivo da insulina
arquivo_pdb = "data/pdb4ins.ent"

# Ler estrutura
parser = PDBParser()

estrutura = parser.get_structure(
    "INSULINA",
    arquivo_pdb
)

# Listas para armazenar coordenadas
x = []
y = []
z = []
elementos = []

# Extrair coordenadas e elementos
for atomo in estrutura.get_atoms():

    coord = atomo.get_coord()

    x.append(coord[0])
    y.append(coord[1])
    z.append(coord[2])

    elementos.append(
        atomo.element
    )

# Informações gerais
print(f"\nTotal de átomos: {len(x)}")

# Contagem dos elementos químicos
contagem = Counter(elementos)

print("\nElementos encontrados:\n")

for elemento, quantidade in sorted(contagem.items()):
    print(
        f"{elemento}: {quantidade}"
    )

# Visualização 3D
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=4
            )
        )
    ]
)

fig.update_layout(
    title="Estrutura 3D da Insulina",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

# Salvar HTML
fig.write_html(
    "insulina_3d.html"
)

print(
    "\nArquivo salvo: insulina_3d.html"
)