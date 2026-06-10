import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from Bio.PDB import PDBParser
from math import sqrt

# ==========================
# CONFIGURAÇÃO DA PÁGINA
# ==========================

st.set_page_config(
    page_title="Insulin.HelenAI",
    page_icon="🧬",
    layout="wide"
)

# ==========================
# MENU LATERAL
# ==========================

st.sidebar.title("🧬 Insulin.HelenAI")

pagina = st.sidebar.radio(
    "Navegação",
    [
        "Início",
        "Dados das Cidades",
        "Estatística",
        "Machine Learning",
        "Álgebra Linear",
        "Bioinformática"
    ]
)


# ==========================
# PÁGINA INICIAL
# ==========================

if pagina == "Início":

    st.title("🧬 Insulin.HelenAI")

    st.subheader(
        "Inteligência Artificial aplicada ao estudo da relação entre poluição atmosférica e diabetes"
    )

    st.divider()

    st.markdown("""
    ## Sobre o projeto
    
    O InsulinAirML integra diferentes áreas da ciência:

    -  Ciências ambientais;
    -  Dados de Fortaleza, Sobral e São Paulo;
    -  Estatística e correlação;
    -  Machine Learning com Random Forest;
    -  Álgebra Linear (autovalores e autovetores);
    -  Bioinformática estrutural da insulina.

    Objetivo:
    
    Investigar como a exposição prolongada a poluentes atmosféricos pode estar associada ao aumento do risco de diabetes e possíveis alterações estruturais simuladas da insulina.
    """)

    st.success("Sistema carregado com sucesso.")


# ==========================
# DADOS DAS CIDADES
# ==========================

elif pagina == "Dados das Cidades":

    st.title("Dados das Cidades")

    st.write("""
    O conjunto de dados utilizado no InsulinAirML reúne
    informações ambientais e indicadores de diabetes
    em três centros urbanos brasileiros.
    """)

    # Carregar dados
    df = pd.read_csv(
        "data/cities_pollution_diabetes.csv"
    )

    st.subheader("Base de Dados")

    st.dataframe(
        df,
        use_container_width=True
    )

    # Médias por cidade
    medias = (
        df.groupby("Cidade")
        .mean(numeric_only=True)
        .reset_index()
    )

    # ----------------------------
    # PM2.5
    # ----------------------------

    fig_pm = px.bar(
        medias,
        x="Cidade",
        y="PM25",
        text="PM25",
        title="Concentração média de PM2.5 por cidade"
    )

    fig_pm.update_layout(
        xaxis_title="Cidade",
        yaxis_title="PM2.5 (µg/m³)",
        xaxis_tickangle=0
    )

    st.plotly_chart(
        fig_pm,
        use_container_width=True
    )

    # ----------------------------
    # Diabetes
    # ----------------------------

    fig_diabetes = px.bar(
        medias,
        x="Cidade",
        y="Diabetes",
        text="Diabetes",
        title="Taxa média de diabetes por cidade"
    )

    fig_diabetes.update_layout(
        xaxis_title="Cidade",
        yaxis_title="Diabetes (%)",
        xaxis_tickangle=0
    )

    st.plotly_chart(
        fig_diabetes,
        use_container_width=True
    )

    st.success(
        "Comparação entre cidades carregada com sucesso."
    )

# ==========================
# ESTATÍSTICA
# ==========================

elif pagina == "Estatística":

    st.title("Análise Estatística")

    st.write("""
    Nesta etapa avaliamos a associação entre a exposição
    aos poluentes atmosféricos e a taxa de diabetes.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Correlação de Pearson",
            "0.9565"
        )

    with col2:
        st.metric(
            "P-valor",
            "< 0.000001"
        )

    with col3:
        st.metric(
            "R² da Regressão",
            "0.9149"
        )

    st.divider()

    st.subheader(
        "Interpretação científica"
    )

    st.info("""
    A análise mostrou uma correlação positiva forte entre
    os níveis de poluição atmosférica e o aumento da taxa
    de diabetes no conjunto de dados analisado.

    O valor elevado do coeficiente de Pearson indica uma
    associação linear intensa entre as variáveis.
    """)

    st.success(
        "Análise estatística concluída."
    )



# ==========================
# MACHINE LEARNING
# ==========================

elif pagina == "Machine Learning":

    st.title("Machine Learning - Random Forest")

    st.write("""
    Foi utilizado o algoritmo Random Forest Regressor para
    estimar a relação entre poluição atmosférica e taxa de diabetes.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "R² do modelo",
            "0.8850"
        )

    with col2:
        st.metric(
            "Erro médio absoluto (MAE)",
            "0.3742"
        )

    st.divider()

    st.subheader("Importância das variáveis ambientais")


    importancias = pd.DataFrame({
        "Variável": [
            "PM2.5",
            "PM10",
            "NO₂",
            "O₃",
            "Temperatura",
            "Umidade"
        ],

        "Importância": [
            64.23,
            24.03,
            7.79,
            1.49,
            1.32,
            1.14
        ]
    })


    fig = px.bar(
        importancias,
        x="Variável",
        y="Importância",
        text="Importância",
        title="Contribuição das variáveis no modelo Random Forest"
    )


    fig.update_layout(
        xaxis_title="Variável Ambiental",
        yaxis_title="Importância (%)",
        xaxis_tickangle=0
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.success(
        """
        O PM2.5 foi a variável com maior influência no modelo,
        representando aproximadamente 64% da importância total.
        """
    )


# ==========================
# ÁLGEBRA LINEAR
# ==========================

elif pagina == "Álgebra Linear":

    st.title("Álgebra Linear e Análise Espectral")

    st.write("""
    Nesta etapa aplicamos conceitos fundamentais de 
    Álgebra Linear para identificar padrões matemáticos
    nos dados ambientais.
    """)


    # ======================
    # MATRIZ X
    # ======================

    df = pd.read_csv(
        "data/cities_pollution_diabetes.csv"
    )

    variaveis = [
        "PM25",
        "PM10",
        "NO2",
        "O3",
        "Temperatura",
        "Umidade"
    ]

    X = df[variaveis].values


    st.subheader("1. Matriz ambiental X")

    st.write(
        "Cada linha representa uma observação e cada coluna uma variável ambiental."
    )

    st.dataframe(
        pd.DataFrame(
            X,
            columns=variaveis
        )
    )


    # ======================
    # NORMALIZAÇÃO
    # ======================

    media = np.mean(X, axis=0)
    desvio = np.std(X, axis=0)

    Z = (X - media) / desvio


    st.subheader("2. Matriz normalizada Z")

    st.dataframe(
        pd.DataFrame(
            Z,
            columns=variaveis
        )
    )


    # ======================
    # COVARIÂNCIA
    # ======================

    C = np.cov(
        Z,
        rowvar=False
    )


    st.subheader("3. Matriz de covariância")

    st.dataframe(
        pd.DataFrame(
            C,
            index=variaveis,
            columns=variaveis
        )
    )


    # ======================
    # AUTOVALORES
    # ======================

    autovalores, autovetores = np.linalg.eig(C)


    ordem = np.argsort(
        autovalores
    )[::-1]


    autovalores = autovalores[ordem]
    autovetores = autovetores[:, ordem]


    variancia = (
        autovalores /
        np.sum(autovalores)
    ) * 100


    st.subheader("4. Autovalores e variância explicada")

    tabela = pd.DataFrame({
        "Componente":
            [f"PC{i+1}" for i in range(len(variancia))],
        "Variância (%)":
            variancia.round(2)
    })

    st.dataframe(tabela)


    # ======================
    # GRÁFICO AUTOVALORES
    # ======================

    fig = px.bar(
        tabela,
        x="Componente",
        y="Variância (%)",
        text="Variância (%)",
        title="Importância dos autovalores"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ======================
    # AUTOVETORES
    # ======================

    st.subheader("5. Autovetores principais")


    vetor_df = pd.DataFrame(
        autovetores,
        index=variaveis,
        columns=[
            f"PC{i+1}"
            for i in range(
                len(variaveis)
            )
        ]
    )


    st.dataframe(
        vetor_df
    )


    # ======================
    # PCA MANUAL
    # ======================


    V2 = autovetores[:, :2]

    Y = np.dot(
        Z,
        V2
    )


    resultado = pd.DataFrame({
        "PC1": Y[:,0],
        "PC2": Y[:,1],
        "Cidade": df["Cidade"]
    })


    st.subheader(
        "6. Espaço dos componentes principais"
    )


    fig2 = px.scatter(
        resultado,
        x="PC1",
        y="PC2",
        color="Cidade",
        text="Cidade",
        size_max=20,
        title="Separação matemática das cidades"
    )


    fig2.update_traces(
        textposition="top center"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    st.success("""
    Conclusão matemática:

    A decomposição espectral da matriz de covariância 
    permitiu encontrar as direções de maior variabilidade 
    dos dados ambientais.

    Os autovalores indicam a quantidade de informação 
    presente em cada componente principal.

    Os autovetores definem as novas direções do espaço 
    matemático utilizadas para comparar os perfis de 
    poluição entre Fortaleza, Sobral e São Paulo.
    """)

# ==========================
# BIOINFORMÁTICA
# ==========================

elif pagina == "Bioinformática":

    st.title(
        "Bioinformática Estrutural da Insulina"
    )

    st.write("""
    A insulina é um hormônio proteico responsável pela 
    regulação da glicose sanguínea.

    Sua estrutura tridimensional pode sofrer alterações
    quando exposta a condições ambientais de estresse,
    como o aumento de radicais livres associado à poluição.
    """)

    st.divider()


    # ======================
    # 1. Estrutura da proteína
    # ======================

    st.subheader(
        "1. Estrutura molecular"
    )

    st.info("""
    A molécula de insulina humana possui:

    • Cadeia A: 21 aminoácidos

    • Cadeia B: 30 aminoácidos

    • Pontes dissulfeto responsáveis pela estabilidade
      da sua conformação tridimensional.
    """)


    st.divider()


    # ==========================
    # 2. MODELO 3D DA INSULINA
    # ==========================

    st.subheader(
        "2. Modelo molecular 3D da insulina"
    )


    arquivo_pdb = "data/pdb4ins.ent"


    parser = PDBParser(
        QUIET=True
    )


    estrutura = parser.get_structure(
        "INSULINA",
        arquivo_pdb
    )


    # Coordenadas dos átomos
    x = []
    y = []
    z = []

    cores = []
    tamanhos = []
    elementos = []


    for atomo in estrutura.get_atoms():

        coord = atomo.get_coord()

        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])


        elemento = atomo.element

        elementos.append(elemento)


        # Cores químicas dos átomos

        if elemento == "C":

            cores.append("gray")
            tamanhos.append(4)


        elif elemento == "O":

            cores.append("red")
            tamanhos.append(5)


        elif elemento == "N":

            cores.append("blue")
            tamanhos.append(5)


        elif elemento == "S":

            cores.append("yellow")
            tamanhos.append(7)


        else:

            cores.append("green")
            tamanhos.append(3)


    st.success(
        f"Modelo carregado com {len(x)} átomos da insulina."
    )


# #==========================
# VISUALIZAÇÃO 3D
# ==========================

    fig = go.Figure()
        # ==========================
    # Ligações entre átomos
    # ==========================

    for i in range(len(x)):

        for j in range(i + 1, len(x)):

            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            distancia = (
                dx**2 +
                dy**2 +
                dz**2
            ) ** 0.5


            # Distância típica de ligação química
            if distancia < 1.9:

                cor_ligacao = "white"
                espessura = 3


                # Destaque para ponte dissulfeto
                if (
                    elementos[i] == "S"
                    and elementos[j] == "S"
                ):

                    cor_ligacao = "yellow"
                    espessura = 8


                fig.add_trace(

                    go.Scatter3d(

                        x=[
                            x[i],
                            x[j]
                        ],

                        y=[
                            y[i],
                            y[j]
                        ],

                        z=[
                            z[i],
                            z[j]
                        ],

                        mode="lines",

                        line=dict(
                            color=cor_ligacao,
                            width=espessura
                        ),

                        hoverinfo="none",

                        showlegend=False

                    )

                )


    # Átomos
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=tamanhos,
                color=cores,
                opacity=0.9
            ),
            text=elementos,
            hovertemplate=
            "Átomo: %{text}<br>" +
            "X: %{x:.2f}<br>" +
            "Y: %{y:.2f}<br>" +
            "Z: %{z:.2f}<extra></extra>",
            name="Átomos"
        )
    )


    # Configuração visual
    fig.update_layout(

        title={
            "text":
            "Estrutura tridimensional da molécula de insulina",
            "x":0.5
        },

        height=700,

        scene=dict(

            xaxis=dict(
                title="Eixo X",
                backgroundcolor="black",
                color="white",
                gridcolor="gray"
            ),

            yaxis=dict(
                title="Eixo Y",
                backgroundcolor="black",
                color="white",
                gridcolor="gray"
            ),

            zaxis=dict(
                title="Eixo Z",
                backgroundcolor="black",
                color="white",
                gridcolor="gray"
            ),

            bgcolor="black",

            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.5,
                    z=1.2
                )
            )

        ),

        paper_bgcolor="black",
        font=dict(
            color="white",
            size=14
        ),

        showlegend=True

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.divider()
    


    # ==========================
    # 3. Simulação temporal da degradação
    # ==========================

    st.subheader(
        "3. Degradação estrutural ao longo do tempo"
    )

    st.write("""
    O modelo compara a estrutura original da insulina
    com uma estrutura simulada após exposição contínua
    a poluentes atmosféricos.
    """)


    # Controle temporal
    anos = st.slider(
        "Tempo de exposição (anos)",
        0,
        20,
        5
    )


    # Intensidade da alteração
    fator = anos * 0.08


    # Gerar estrutura alterada
    x_alt = []
    y_alt = []
    z_alt = []


    for i in range(len(x)):

        desloc_x = np.random.normal(0, fator)
        desloc_y = np.random.normal(0, fator)
        desloc_z = np.random.normal(0, fator)


        x_alt.append(
            x[i] + desloc_x
        )

        y_alt.append(
            y[i] + desloc_y
        )

        z_alt.append(
            z[i] + desloc_z
        )


    # ==========================
    # Gráfico comparativo
    # ==========================

    fig_comp = go.Figure()


    # Molécula original
    fig_comp.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=3,
                color="cyan"
            ),
            name="Insulina original"
        )
    )


    # Molécula alterada
    fig_comp.add_trace(
        go.Scatter3d(
            x=x_alt,
            y=y_alt,
            z=z_alt,
            mode="markers",
            marker=dict(
                size=3,
                color="red"
            ),
            name="Insulina exposta"
        )
    )


    # Linhas de deslocamento
    for i in range(len(x)):

        fig_comp.add_trace(
            go.Scatter3d(
                x=[x[i], x_alt[i]],
                y=[y[i], y_alt[i]],
                z=[z[i], z_alt[i]],
                mode="lines",
                line=dict(
                    color="white",
                    width=2
                ),
                showlegend=False
            )
        )


    # Visualização
    fig_comp.update_layout(

        title=f"Alterações estruturais após {anos} anos de exposição",

        height=800,

        scene=dict(
            bgcolor="black",
            xaxis=dict(
                title="X",
                color="white"
            ),
            yaxis=dict(
                title="Y",
                color="white"
            ),
            zaxis=dict(
                title="Z",
                color="white"
            )
        ),

        paper_bgcolor="black",

        font=dict(
            color="white",
            size=14
        )
    )


    st.plotly_chart(
        fig_comp,
        use_container_width=True
    )


    # ==========================
    # RMSD calculado
    # ==========================

    deslocamento = np.sqrt(
        (np.array(x_alt)-np.array(x))**2 +
        (np.array(y_alt)-np.array(y))**2 +
        (np.array(z_alt)-np.array(z))**2
    )

    rmsd = np.sqrt(
        np.mean(deslocamento**2)
    )


    st.metric(
        "RMSD da estrutura simulada",
        f"{rmsd:.2f} Å"
    )


    # Interpretação automática

    if rmsd < 0.5:

        st.success(
            "Baixa alteração estrutural."
        )

    elif rmsd < 1.5:

        st.warning(
            "Alteração estrutural moderada."
        )

    else:

        st.error(
            "Alta alteração estrutural simulada."
        )
        st.divider()

    st.subheader(
        "4. Geometria molecular atômica: Insulina normal vs exposta à poluição"
    )

    st.write("""
    Esta visualização representa a geometria molecular em nível atômico.
    Cada esfera representa um átomo e as linhas representam ligações químicas aproximadas.

    A estrutura da direita apresenta uma simulação de alteração estrutural causada por exposição prolongada à poluição atmosférica.
    """)


    # ===============================
    # Coletar todos os átomos reais
    # ===============================

    x_real = []
    y_real = []
    z_real = []

    cores = []
    elementos = []


    for atomo in estrutura.get_atoms():

        coord = atomo.get_coord()

        x_real.append(coord[0])
        y_real.append(coord[1])
        z_real.append(coord[2])

        elemento = atomo.element

        elementos.append(elemento)


        if elemento == "C":
            cores.append("gray")

        elif elemento == "O":
            cores.append("red")

        elif elemento == "N":
            cores.append("blue")

        elif elemento == "S":
            cores.append("yellow")

        else:
            cores.append("white")


    # ===============================
    # Criar estrutura alterada
    # ===============================

    x_alt = []
    y_alt = []
    z_alt = []


    for i in range(len(x_real)):

        fator = np.random.normal(
            0,
            0.8
        )

        x_alt.append(
            x_real[i] + fator
        )

        y_alt.append(
            y_real[i] + fator
        )

        z_alt.append(
            z_real[i] + fator
        )


    # ===============================
    # Função para desenhar ligações
    # ===============================

    def criar_ligacoes(fig, x, y, z):

        quantidade = len(x)


        for i in range(quantidade):

            for j in range(i+1, quantidade):

                distancia = sqrt(
                    (x[i]-x[j])**2 +
                    (y[i]-y[j])**2 +
                    (z[i]-z[j])**2
                )


                # distância aproximada de ligação
                if distancia < 1.8:

                    fig.add_trace(
                        go.Scatter3d(
                            x=[x[i], x[j]],
                            y=[y[i], y[j]],
                            z=[z[i], z[j]],

                            mode="lines",

                            line=dict(
                                color="white",
                                width=4
                            ),

                            showlegend=False
                        )
                    )
    def adicionar_legenda(fig):

        elementos = [
            ("Carbono (C)", "gray"),
            ("Oxigênio (O)", "red"),
            ("Nitrogênio (N)", "blue"),
            ("Enxofre (S)", "yellow"),
            ("Hidrogênio / Outros", "white")
        ]

        for nome, cor in elementos:

            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",

                    marker=dict(
                        size=8,
                        color=cor
                    ),

                    name=nome,
                    showlegend=True
                )
            )



    # ===============================
    # Criar colunas lado a lado
    # ===============================

    col1, col2 = st.columns(2)


# -------------------------------
# INSULINA NORMAL
# -------------------------------

    with col1:

        st.markdown("### Insulina normal")

        fig_normal = go.Figure()

        criar_ligacoes(
            fig_normal,
            x_real,
            y_real,
            z_real
        )

        fig_normal.add_trace(
            go.Scatter3d(
                x=x_real,
                y=y_real,
                z=z_real,
                mode="markers",
                marker=dict(
                    size=5,
                    color=cores
                ),
                text=elementos,
                name="Átomos"
            )
        )

        # Legenda dos elementos
        adicionar_legenda(fig_normal)

        fig_normal.update_layout(
            height=600,

            showlegend=True,

            legend=dict(
                bgcolor="rgba(0,0,0,0.6)",
                font=dict(
                    color="white",
                    size=12
                )
            ),

            scene=dict(
                bgcolor="black"
            )
        )

        st.plotly_chart(
            fig_normal,
            use_container_width=True
        )


    # -------------------------------
    # INSULINA ALTERADA
    # -------------------------------

    with col2:

        st.markdown("### Exposição prolongada à poluição")

        fig_alt = go.Figure()

        criar_ligacoes(
            fig_alt,
            x_alt,
            y_alt,
            z_alt
        )

        fig_alt.add_trace(
            go.Scatter3d(
                x=x_alt,
                y=y_alt,
                z=z_alt,
                mode="markers",
                marker=dict(
                    size=5,
                    color=cores
                ),
                text=elementos,
                name="Átomos alterados"
            )
        )

        # Legenda dos elementos
        adicionar_legenda(fig_alt)

        fig_alt.update_layout(
            height=600,

            showlegend=True,

            legend=dict(
                bgcolor="rgba(0,0,0,0.6)",
                font=dict(
                    color="white",
                    size=12
                )
            ),

            scene=dict(
                bgcolor="black"
            )
        )

        st.plotly_chart(
            fig_alt,
            use_container_width=True
        )


    st.success(
        """
        Comparação concluída:

        A estrutura da esquerda representa a conformação
        molecular original da insulina.

        A estrutura da direita representa uma simulação de
        alteração geométrica causada pelo estresse oxidativo
        associado à exposição prolongada à poluição atmosférica.
        """
    )