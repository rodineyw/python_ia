import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Importa a base de dados
tabela = pd.read_csv("clientes.csv")
display(tabela)

# Verificar se temos valores vazios ou valores reconhecido em formato errado
print(tabela.info())
print(tabela.columns)

# vai transformar as coluna de texto em números
codificador = LabelEncoder()

# só não aplicamos na coluna de score_crédito que é o nosso objetivo
for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

# verificando se realmente todas as colunas foram modificadas
for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        print(coluna)

display(tabela)

# escolhendo quais coluans vamos usar para treinar o modelo
# y é a coluna que queremos que o modelo calcule
# x é a coluna que vamos usar para prever o score de crédito
x = tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]

# separamos os dados em treino e teste.
# Treino vamos dar para os modelos aprenderem e teste vamos usar para ver se o modelo aprendeu corretamente
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=1
)

modelo_arvore = RandomForestClassifier()  # modelo arvore de decisao
modelo_knn = (
    KNeighborsClassifier()
)  # modelo do KNN (nearest neighbors - vizinhos mais próximos)

# Treinando os modelos
modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

# Se o nosso modelo chutasse tudo 'Standard', qual seria a acurácia do modelo?
contagem_scores = tabela["score_credito"].value_counts()
print(contagem_scores["Standard"] / sum(contagem_scores))

# Calculamos as previsões
previsao_arvore = modelo_arvore.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

# Comparamos as previsões com o y_teste
# esse score queremos o maior (maior acuracia, mas também tem que ser maior do que o chute de todo Standard)
print(accuracy_score(y_teste, previsao_arvore))
print(accuracy_score(y_teste, previsao_knn))

# quais as caracteristicas mais importantes para definir o score de crédito?
columns = list(x_teste.columns)
importancia = pd.DataFrame(index=columns, data=modelo_arvore.feature_importances_)
importancia = importancia * 100
print(importancia)
