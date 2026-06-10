from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ler estrutura
parser = PDBParser()

estrutura = parser.get_structure(
    "INSULINA",
    "data/pdb4ins.ent"
)

coords = []

for atomo in estrutura.get_atoms():
    coords.append(
        atomo.get_coord()
    )

coords = np.array(coords)

anos = [0, 5, 10, 20]
intensidades = [0.0, 0.2, 0.5, 1.0]

rmsd_resultados = []

for ano, intensidade in zip(
    anos,
    intensidades
):

    ruido = np.random.normal(
        0,
        intensidade,
        coords.shape
    )

    coords_modificadas = (
        coords + ruido
    )

    diferenca = (
        coords_modificadas - coords
    )

    rmsd = np.sqrt(
        np.mean(
            diferenca**2
        )
    )

    rmsd_resultados.append(
        rmsd
    )

    print(
        f"{ano} anos -> RMSD = {rmsd:.4f}"
    )

# Salvar tabela

df = pd.DataFrame({
    "Anos": anos,
    "RMSD": rmsd_resultados
})

df.to_csv(
    "data/rmsd_results.csv",
    index=False
)

# Gráfico

plt.figure(figsize=(8,5))

plt.plot(
    anos,
    rmsd_resultados,
    marker="o"
)

plt.title(
    "RMSD da Insulina vs Tempo de Exposição"
)

plt.xlabel(
    "Tempo (anos)"
)

plt.ylabel(
    "RMSD"
)

plt.grid(True)

plt.show()