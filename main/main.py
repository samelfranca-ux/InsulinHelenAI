import os

while True:

    print("\n======================")
    print("INSULINAIRML")
    print("======================")

    print("1 - Álgebra Linear")
    print("2 - Correlação")
    print("3 - Comparação de Cidades")
    print("4 - Mapa das Cidades")
    print("5 - Molécula 3D")
    print("0 - Sair")

    opcao = input(
        "\nEscolha: "
    )

    if opcao == "1":

        os.system(
            "python src/linear_algebra_analysis.py"
        )

    elif opcao == "2":

        os.system(
            "python src/statistical_analysis.py"
        )

    elif opcao == "3":

        os.system(
            "python src/city_comparison.py"
        )

    elif opcao == "4":

        os.system(
            "python src/city_map.py"
        )

    elif opcao == "5":

        os.system(
            "python src/insulin_3d.py"
        )

    elif opcao == "0":

        break

    else:

        print(
            "Opção inválida."
        )