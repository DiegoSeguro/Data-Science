#!/usr/bin/env python
# coding: utf-8

#Importando bibliotecas.
import numpy as np
import seaborn as sns
import pandas as pd

# Lendo e atribuindo variável df aos dados.
df = pd.read_json('results-20210820-192643.json')

# #### 2.2 UTILIZANDO SOMENTE AS COLUNAS FUNDAMENTAIS (MODELO 01): #### #

# Criando a coluna Conversion_Output_I (int) com base na coluna Conversion.
df['Conversion_Output_I'] = df['Conversion'].astype(int)

# Criando uma tabela com os dados das conversões positivas.
Conversion_win = df[
    df[
        'Conversion_Output_I'] == 
        1][[
            'Price', 
            'DealerETAGroup',
            'Quality',
            'Position',
            'RecommendationType'
            ]]

# Imprimindo a tabela com os dados das conversões positivas. 
print(Conversion_win)

# Imprimindo a contagem de valores da nova coluna Conversion_Output_I.
print(df['Conversion_Output_I'].value_counts())

# Importando o modelo linear generalizado de regressão logística.
import statsmodels.formula.api as smf

# Atribuindo as variáveis ao modelo
model = smf.logit('Conversion_Output_I ~ Price + Quality + Position',
data = df)

# Atribuindo a variável resultados à saída do modelo.
resultados = model.fit()

# Exibindo os resultados, saída do modelo.
print(resultados.summary())

# Importando recursos do Sklearn.
from sklearn import linear_model
lr = linear_model.LogisticRegression()

# Criando uma variável para as variáveis preditoras.
preditoras = df[[
                'Price', 
                'Quality', 
                'Position'
                ]]

# Executando a adequação do modelo.
adequa_model = lr.fit(X = preditoras, y = df['Conversion_Output_I'])

# Criando variáveis para a exibição dos resultados.
valores = np.append(adequa_model.intercept_, adequa_model.coef_)
nomeclatura = np.append('intercept', preditoras.columns)

# Colocando os resultados no formado DataFrame com rótulos.
resultados_adequa = pd.DataFrame(valores, index = nomeclatura, columns=['coef'])

# Exponenciando os resultados.
resultados_adequa['exp'] = np.exp(resultados_adequa['coef'])

# Exibindo os resultados.
print(resultados_adequa)

# #### UTILIZANDO AS COLUNAS COMPLEMENTARES (MODELO 02): #### #

# Lendo e atribuindo variável dataframe aos dados.
dados = pd.read_json('results-20210820-192643 (2).json')

# Agrupando pela coluna 'Quality' e definido médias para 'Conversion', 'Position' e 'Price'.
dados_2 = dados.groupby('Quality').agg('mean'.split(',')).reset_index()

# Configurando as integração das novas colunas: "Conversion_mean", "Position_mean", "Price_mean" 
dados_2.columns = ['_'.join(c).strip('_')for c in dados_2.columns.values] 

# Mesclando as colunas existente com as novas colunas. 
df2 = dados.merge(dados_2, on='Quality')  

# Criando a coluna Conversion_Output_I2 (int) com base na coluna Conversion.
df2['Conversion_Output_I2'] = df2['Conversion'].astype(int)

# Atribuindo as variáveis ao modelo
model2 = smf.logit(
                    'Conversion_Output_I2 ~ \
                    DealerETAGroup + \
                    RecommendationType + \
                    Conversion_mean + \
                    Position_mean + \
                    Price_mean',
            data = df2)

# Atribuindo a variável resultados à saída do modelo.
resultados2 = model2.fit()

# Exibindo os resultados, saída do modelo.
print(resultados2.summary())

# Exponenciando os resultadoa
result_exp = np.exp(resultados2.params)
print(result_exp)
