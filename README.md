# Prevendo Pontuações de Crédito com Aprendizado de Máquina

Bem-vindo ao nosso projeto de Previsão de Pontuações de Crédito! Neste repositório, vamos guiá-lo pelo processo de prever pontuações de crédito usando algoritmos populares de aprendizado de máquina. Não se preocupe se você é novo nisso - vamos torná-lo fácil e divertido!

## Sumário
- [Prevendo Pontuações de Crédito com Aprendizado de Máquina](#prevendo-pontuações-de-crédito-com-aprendizado-de-máquina)
  - [Sumário](#sumário)
  - [Introdução](#introdução)
  - [Começando](#começando)
  - [Preparação de Dados](#preparação-de-dados)
  - [Treinamento de Modelos](#treinamento-de-modelos)
  - [Avaliação de Modelos](#avaliação-de-modelos)
  - [Importância das Características](#importância-das-características)

## Introdução

No mundo de hoje, entender as pontuações de crédito é crucial para a tomada de decisões financeiras. Neste projeto, nosso objetivo é prever pontuações de crédito usando técnicas de aprendizado de máquina. Vamos trabalhar com Python e as seguintes bibliotecas:

- Pandas para manipulação de dados
- Scikit-learn para ferramentas de aprendizado de máquina

Vamos direto ao código!

## Começando

Primeiro, importamos as bibliotecas necessárias e carregamos nosso conjunto de dados:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Importe o conjunto de dados
tabela = pd.read_csv("clientes.csv")
display(tabela)
```

## Preparação de Dados

Precisamos garantir que nossos dados estejam no formato certo para aprendizado de máquina. Isso envolve lidar com valores ausentes e converter colunas de texto em representações numéricas:

```python
# Verifique valores ausentes e tipos de dados incorretos
print(tabela.info())
print(tabela.columns)

# Converta colunas de texto em numéricas usando LabelEncoder
codificador = LabelEncoder()

for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

display(tabela)
```

## Treinamento de Modelos

Agora é hora de selecionar nossas características e dividir os dados em conjuntos de treinamento e teste:

```python
# Escolha as características e a variável alvo
x = tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]

# Divida os dados em conjuntos de treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=1
)

# Crie modelos de aprendizado de máquina
modelo_arvore = RandomForestClassifier()  # Modelo de Árvore de Decisão
modelo_knn = KNeighborsClassifier()  # Modelo de Vizinhos Mais Próximos

# Treine os modelos
modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

## Avaliação de Modelos

Vamos avaliar o desempenho dos nossos modelos comparando as previsões com as pontuações de crédito reais:

```python
# Faça previsões
previsao_arvore = modelo_arvore.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste.to_numpy())

# Compare as previsões com os dados de teste
print("Acurácia da Árvore de Decisão:", accuracy_score(y_teste, previsao_arvore))
print("Acurácia dos Vizinhos Mais Próximos:", accuracy_score(y_teste, previsao_knn))

# E se nosso modelo sempre adivinhasse 'Padrão'? Qual seria a acurácia?
contagem_scores = tabela["score_credito"].value_counts()
print("Acurácia se sempre adivinhasse 'Padrão':", contagem_scores["Padrão"] / sum(contagem_scores))
```

## Importância das Características

Por fim, vamos explorar quais características são mais importantes na determinação das pontuações de crédito:

```python
# Determine a importância das características para o modelo de Árvore de Decisão
colunas = list(x_teste.columns)
importancia = pd.DataFrame(index=colunas, data=modelo_arvore.feature_importances_)
importancia = importancia * 100
print("Importância das Características (%):")
print(importancia)
```

É isso aí! Você acaba de explorar o mundo da previsão de pontuações de crédito com aprendizado de máquina. Sinta-se à vontade para experimentar diferentes algoritmos, hiperparâmetros e técnicas de engenharia de características para melhorar suas previsões.

Feliz codificação! 🚀📈