import pandas as pd
import numpy as np

# Simulação de dados
np.random.seed(42)

# Criando um DataFrame com simulação de fraude
data = {
    'order_amount': np.random.randint(10, 500, 100),  # Valores aleatórios de quantia de pedidos
    'category': np.random.choice(['electronics', 'fashion', 'groceries', 'health'], 100),  # Categorias
    'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'cash'], 100),  # Método de pagamento
    'fraud_flag': np.random.choice([0, 1], 100, p=[0.85, 0.15])  # 15% de fraude, 85% não fraude
}

df = pd.DataFrame(data)

# Classe para cálculo de WA
def Woe_IV_Dis(df, features, target):
    aux = features + [target] 
    
    df = df[aux].copy()
    
    # Empty dataframe
    df_woe_iv = pd.DataFrame({},index=[])
    
    for feature in features:
        df_woe_iv_aux = pd.crosstab(df[feature], df[target], normalize='columns') \
                        .assign(WoE=lambda i: np.log(i[0] / i[1])) \
                        .assign(IV=lambda i: (i['WoE']*(i[0]-i[1]))) \
                        .assign(IV_total=lambda i: np.sum(i['IV']))

        df_woe_iv = pd.concat([df_woe_iv, df_woe_iv_aux])
    
    return df_woe_iv   

# Calculate information value
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum() 
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))
    iv = data['IV'].sum()
    # print(iv)

    return iv, data

if __name__ == '__main__':
    print(df.head())
    # woe = ln((% de não fraude) / (% de fraude))

    # feature = payment_method and target = fraud_flag
    # Calculando a taxa de não fraude e fraude para cada método de pagamento
    paymennts_methods = df['payment_method'].unique()
    for method in paymennts_methods:
        mask = (df['payment_method'] == method)
        total = sum(mask)
        fraud_count = sum(df[mask]['fraud_flag'])
        non_fraud_count = total - fraud_count
        fraud_rate = fraud_count / total
        non_fraud_rate = non_fraud_count / total
        woe = np.log(non_fraud_rate / fraud_rate)
        print(f'{method} - WOE: {woe}')

    # Calculando o IV
    woe_iv = Woe_IV_Dis(df, ['payment_method'], 'fraud_flag')
    print(woe_iv)
    
    iv, data = calc_iv(df, 'payment_method', 'fraud_flag', pr=True)
    print(data)