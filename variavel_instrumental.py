import pandas as pd
import seaborn as sns
import statsmodels.api as sm 
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.iv import IV2SLS
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

### Testes 

df= pd.read_excel(r'data\dados_completos_final.xlsx')
df

#Primeira analise dos dados
df.shape
df.columns

# Variável independente (X)
X = df["Gasto_Seguranca"]

# Variável dependente (y)
y = df["Qtd_Homicidios"]

# Adicionar constante (intercepto)
X = sm.add_constant(X)

# Estimar o modelo MQO (OLS)
modelo = sm.OLS(y, X).fit()

# Exibir o resumo dos resultados
print(modelo.summary())

# Primeira etapa: regredir Gasto_Seguranca no instrumento e controles
X_first = sm.add_constant(df[['valor_icms', 'População', 'PIB_per_capita']])
y_first = df['Gasto_Seguranca']

first_stage = sm.OLS(y_first, X_first).fit()
print(first_stage.summary())

# Variáveis
y = df['Qtd_Homicidios']            # dependente
endog = df['Gasto_Seguranca']       # endógena
instr = df['valor_icms']        # instrumental
controls = df[['População', 'PIB_per_capita']]

# Adiciona constante
exog = sm.add_constant(controls)


# Modelo 2SLS
iv_model = IV2SLS(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instr
).fit(cov_type='robust')

print(iv_model.summary)

#Correlação
df2 = df[["Gasto_Seguranca","PIB_per_capita","Qtd_Homicidios","População"]]
df_numerico = df2.select_dtypes(include = "number")
df_corr = df_numerico.corr()
sns.heatmap(df_corr, annot=True)


# --- 1) Calcular MDE aproximado a partir do SE do coeficiente IV ---
coef = -4.851e-07   # seu coef IV
se = 5.135e-07      # seu SE do coef (do output)
alpha = 0.05
z = 1.96            # z para 95% CI
MDE = z * se
print(f"MDE (95% CI) ≈ {MDE:.3e}")
print(f"Coef estimado = {coef:.3e} (|coef| < MDE? -> {abs(coef) < MDE})")

# --- 2) Rodar IV com LOGs (ajuda escala/heterocedasticidade) ---
df['ln_homicidios'] = np.log(df['Qtd_Homicidios'] + 1)
df['ln_gasto_seg'] = np.log(df['Gasto_Seguranca'] + 1)
df['ln_pop'] = np.log(df['População'])
df['ln_pibpc'] = np.log(df['PIB_per_capita'] + 1)

# instrumento (pode usar log também se fizer sentido)
df['ln_valor_icms'] = np.log(df['valor_icms'] + 1)

y = df['ln_homicidios']
endog = df['ln_gasto_seg']
instr = df['ln_valor_icms']            # ou lista de instrumentos
controls = df[['ln_pop','ln_pibpc']]
exog = sm.add_constant(controls)

iv_log = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instr).fit(cov_type='robust')
print(iv_log.summary)

# Criar variáveis ao quadrado
df["ln_gasto_seg_quadrado"] = df["ln_gasto_seg"] ** 2
df["ln_valor_icms_quadrado"] = df["ln_valor_icms"] ** 2

# Variáveis
y = df["ln_homicidios"]
endog = df[["ln_gasto_seg", "ln_gasto_seg_quadrado"]]   # duas endógenas
instr = df[["ln_valor_icms", "ln_valor_icms_quadrado"]] # dois instrumentos
controls = df[["ln_pop", "ln_pibpc"]]
exog = sm.add_constant(controls)

# Modelo IV com termos quadráticos
iv_quad = IV2SLS(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instr
).fit(cov_type="robust")

print(iv_quad.summary)

# Criar variáveis ao quadrado
df["ln_gasto_seg_quadrado"] = df["ln_gasto_seg"] ** 2
df["ln_valor_icms_quadrado"] = df["ln_valor_icms"] ** 2

# Variáveis
y = df["ln_homicidios"]
endog = df[["ln_gasto_seg", "ln_gasto_seg_quadrado"]]   # duas endógenas
instr = df[["ln_valor_icms", "ln_valor_icms_quadrado"]] # dois instrumentos
controls = df[["ln_pop"]]
exog = sm.add_constant(controls)

# Modelo IV com termos quadráticos
iv_quad = IV2SLS(
    dependent=y,
    exog=exog,
    endog=endog,
    instruments=instr
).fit(cov_type="robust")

iv_quad.summary

# ---------------------------------------------------------
# GRÁFICOS 
# ---------------------------------------------------------

# 1) Resíduos x valores ajustados – OLS
ols_fitted = modelo.fittedvalues
ols_resid = modelo.resid

plt.figure(figsize=(6,4))
plt.scatter(ols_fitted, ols_resid, alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Valores ajustados (OLS)')
plt.ylabel('Resíduos')
plt.title('Resíduos vs. valores ajustados – OLS')
plt.tight_layout()
plt.savefig(r'figs\ols_residuos_vs_ajustados.png', dpi=300)
plt.show()

# 2) Resíduos x valores ajustados – IV (2SLS)
# o objeto do linearmodels tem fitted_values e resids
iv_fitted = iv_model.fitted_values
iv_resid = iv_model.resids

plt.figure(figsize=(6,4))
plt.scatter(iv_fitted, iv_resid, alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Valores ajustados (IV-2SLS)')
plt.ylabel('Resíduos')
plt.title('Resíduos vs. valores ajustados – IV-2SLS')
plt.tight_layout()
plt.savefig(r'figs\iv_residuos_vs_ajustados.png', dpi=300)
plt.show()

# 3) QQ-plot dos resíduos – OLS
plt.figure(figsize=(5,5))
qqplot(ols_resid, line='s')
plt.title('QQ-plot dos resíduos – OLS')
plt.tight_layout()
plt.savefig(r'figs\ols_qqplot.png', dpi=300)
plt.show()

# 4) QQ-plot dos resíduos – IV
plt.figure(figsize=(5,5))
qqplot(iv_resid, line='s')
plt.title('QQ-plot dos resíduos – IV-2SLS')
plt.tight_layout()
plt.savefig(r'figs\iv_qqplot.png', dpi=300)
plt.show()

# 5) Comparação visual dos coeficientes OLS x IV
# do OLS simples você estimou só gasto, então pegamos esse coef.
coef_ols = modelo.params['Gasto_Seguranca']

# do IV pegamos o coeficiente da variável endógena
coef_iv = iv_model.params['Gasto_Seguranca']

plt.figure(figsize=(5,4))
plt.bar(['OLS', 'IV-2SLS'], [coef_ols, coef_iv])
plt.title('Coeficiente de Gasto_Seguranca – OLS vs IV')
plt.ylabel('Coeficiente')
plt.tight_layout()
plt.savefig(r'figs\coef_ols_vs_iv.png', dpi=300)
plt.show()
