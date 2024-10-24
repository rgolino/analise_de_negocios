
# Importar as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

def run_analysis():
    # Passo 2: Coleta de Dados
    data = {
        'mes': pd.date_range(start='2022-01-01', periods=24, freq='M'),
        'custo_producao': np.random.normal(10000, 2000, 24),  # Custo de produção
        'vendas': np.random.normal(12000, 3000, 24),  # Receita de vendas
        'horas_trabalhadas': np.random.normal(160, 20, 24),  # Horas trabalhadas por funcionário
        'numero_funcionarios': np.random.randint(50, 70, 24),  # Número de funcionários
        'despesas_administrativas': np.random.normal(5000, 1000, 24)  # Despesas fixas
    }
    
    df = pd.DataFrame(data)

    # Passo 3: Análise Exploratória de Dados (EDA)
    print(df.head())
    print(df.describe())

    sns.pairplot(df)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['mes'], df['custo_producao'], label='Custo de Produção')
    plt.plot(df['mes'], df['vendas'], label='Receita de Vendas')
    plt.xlabel('Meses')
    plt.ylabel('Valores')
    plt.title('Tendência de Custos e Vendas')
    plt.legend()
    plt.show()

    # Passo 4: Análise de Custo-Volume-Lucro (CVL)
    df['lucro'] = df['vendas'] - df['custo_producao'] - df['despesas_administrativas']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='horas_trabalhadas', y='lucro', data=df)
    plt.title('Relação entre Horas Trabalhadas e Lucro')
    plt.show()

    # Passo 5: Modelagem Preditiva
    X = df[['custo_producao', 'horas_trabalhadas', 'numero_funcionarios', 'despesas_administrativas']]
    y = df['lucro']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regressão Linear
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    y_pred_lr = model_lr.predict(X_test)

    print("Erro Quadrático Médio (MSE) - Regressão Linear:", mean_squared_error(y_test, y_pred_lr))
    print("Coeficiente de Determinação (R^2) - Regressão Linear:", r2_score(y_test, y_pred_lr))

    # Modelo de Árvore de Decisão
    model_tree = DecisionTreeRegressor(random_state=42)
    model_tree.fit(X_train, y_train)

    y_pred_tree = model_tree.predict(X_test)

    print("Erro Quadrático Médio (MSE) - Árvore de Decisão:", mean_squared_error(y_test, y_pred_tree))
    print("Coeficiente de Determinação (R^2) - Árvore de Decisão:", r2_score(y_test, y_pred_tree))

    # Passo 6: Previsão de Séries Temporais (ARIMA)
    df.set_index('mes', inplace=True)
    model_arima = ARIMA(df['custo_producao'], order=(1, 1, 1))
    model_fit = model_arima.fit()

    forecast = model_fit.forecast(steps=6)
    print("Previsão de Custos de Produção para os próximos 6 meses:", forecast)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['custo_producao'], label='Custo de Produção Histórico')
    plt.plot(pd.date_range(start=df.index[-1], periods=6, freq='M'), forecast, label='Previsão de Custos', color='red')
    plt.xlabel('Meses')
    plt.ylabel('Custo de Produção')
    plt.title('Previsão de Custo de Produção')
    plt.legend()
    plt.show()

    # Passo 7: Criação de Dashboards
    df.reset_index(inplace=True)
    df['mes_str'] = df['mes'].dt.strftime('%Y-%m')
    df.set_index('mes_str', inplace=True)

    df[['custo_producao', 'vendas', 'lucro']].plot(kind='bar', figsize=(12, 8))
    plt.title('Comparação de Custo, Vendas e Lucro')
    plt.xlabel('Mês')
    plt.ylabel('Valores')
    plt.xticks(rotation=45)
    plt.show()

# Chamada da função principal
if __name__ == '__main__':
    run_analysis()
