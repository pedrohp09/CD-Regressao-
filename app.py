# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Carregando o dataset Boston Housing
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame

# Visualizando as primeiras linhas do dataset
print(df.head())

# Informações sobre o dataset
print("\nInformações do dataset:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Visualizando a distribuição da variável target (MEDV - valor médio das casas)
plt.figure(figsize=(8, 6))
sns.histplot(df['MEDV'], kde=True)
plt.title('Distribuição dos valores médios das casas (MEDV)')
plt.xlabel('Valor médio (em $1000)')
plt.ylabel('Frequência')
plt.show()

# Matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()

# Relação entre LSTAT (% de população de baixa renda) e MEDV
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['LSTAT'], y=df['MEDV'])
plt.title('Relação entre LSTAT e MEDV')
plt.xlabel('% de população de baixa renda')
plt.ylabel('Valor médio das casas (em $1000)')
plt.show()

# Separando features e target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados (opcional, mas pode ajudar em alguns casos)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criando e treinando o modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Coeficientes do modelo
print("\nCoeficientes do modelo:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
    
print(f"\nIntercept: {model.intercept_:.4f}")

# Fazendo previsões
y_pred = model.predict(X_test_scaled)

# Métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMétricas de avaliação:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualizando previsões vs valores reais
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Valores Reais vs Previsões')
plt.show()

# Resíduos
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Previsões')
plt.ylabel('Resíduos')
plt.title('Análise de Resíduos')
plt.show()

# Criando um dataframe com os coeficientes para análise
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nImportância das features:")
print(coef_df)

# Plotando a importância das features
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Importância das Features no Modelo')
plt.xlabel('Coeficiente')
plt.ylabel('Feature')
plt.show()

