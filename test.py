import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exemplo de probabilidades simuladas (substitua pelas suas previsões)
np.random.seed(42)
# Simulando probabilidades para classe 0 e 1 (substitua por suas previsões reais)
probs_classe_0 = np.random.beta(2, 5, size=200)  # Probabilidades simuladas para a classe 0
probs_classe_1 = np.random.beta(5, 2, size=200)  # Probabilidades simuladas para a classe 1

# Criando um DataFrame com as probabilidades e classes correspondentes
df_probs = pd.DataFrame({
    'Probabilidades': np.concatenate([probs_classe_0, probs_classe_1]),
    'Classe': [0] * 200 + [1] * 200
})

# Criando o gráfico de densidade
plt.figure(figsize=(10, 6))
sns.kdeplot(df_probs[df_probs['Classe'] == 0]['Probabilidades'], shade=True, color='blue', label='Classe 0')
sns.kdeplot(df_probs[df_probs['Classe'] == 1]['Probabilidades'], shade=True, color='red', label='Classe 1')

# Detalhes do gráfico
plt.title('Distribuição das Probabilidades por Classe')
plt.xlabel('Probabilidade')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)

# Exibir o gráfico
plt.show()