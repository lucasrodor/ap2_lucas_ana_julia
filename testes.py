import pandas as pd
import seaborn as sns
import statsmodels.api as sm 
import pandas as pd
import matplotlib.pyplot as plt
from apis import df_final


#REGRESSAO HOMICÍDIO X POPULAÇÃO

#variavel independente
X= df_final["População"]

#variavel dependente
y= df_final["Qtd_Homicidios"]

#Constante
x = sm.add_constant(X)
modelo = sm.OLS(y,X).fit()
print(modelo.summary())