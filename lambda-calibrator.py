
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


import pandas as pd

#reading .csv 
df = pd.read_csv("teste3TITAN_SIGNALS_2024-10-04_11-03-11.csv", delimiter=";", encoding='utf-8')

#data length
startCol = 0
endCol = 830
df['lambdaA'] = df['lambdaA'][startCol:endCol]
df['pgbkA_In'] = df['pgbkA_In'][startCol:endCol]


# Linear regression
df['A'] = df['lambdaA'].rolling(window=10).mean()
df['B'] = df['pgbkA_In'].rolling(window=10).mean()
# Remover possíveis valores NaN resultantes da média móvel
df_filtrado = df.dropna(subset=['A', 'B'])

# Redefinir X e Y com os dados filtrados
X = df_filtrado[['A']]  # Variável independente (formato 2D)
Y = df_filtrado['B']   # Variável dependente

model = LinearRegression()

# fitting to the model
model.fit(X, Y)

# Obtendo os coeficientes alfa e beta
alfa = model.coef_[0]  # Coeficiente de inclinação (alfa)
beta = model.intercept_  # Intercepto (beta)

print(f'Alfa: {alfa}, Beta: {beta}')

#plot comparison
fig, axs = plt.subplots(2,1,figsize=(10,6))   
axs[0].set_title("Experimental data (.csv)")
axs[0].plot(df["RPM"][startCol:endCol],label="RPM")
axs[0].legend()    
axs[0].grid(True)

axs[1].plot(df["pgbkA_In"].rolling(window=10).mean(),label="Lambda FT")
axs[1].legend()
axs[1].grid(True)

axs[1].plot(df["lambdaA"].rolling(window=10).mean()*alfa + beta,label="Lambda Calibr.")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
