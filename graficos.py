import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from apis import df_go, df_final, df_uf_2023, df_mun_go_2023

# pasta de saída
os.makedirs("figs", exist_ok=True)
pdf = PdfPages("figs/graficos_seguranca_go.pdf")

# ===== 0) Preparos rápidos =====
# df_go: série histórica de GO
df_go = df_go.copy()
df_go["periodo"] = pd.to_datetime(df_go["periodo"], errors="coerce")
df_go = df_go.sort_values("periodo")
df_go["valor"] = pd.to_numeric(df_go["valor"], errors="coerce")
if "media_movel_3a" not in df_go.columns:
    df_go["media_movel_3a"] = df_go["valor"].rolling(3).mean()
if "taxa_var" not in df_go.columns:
    df_go["taxa_var"] = df_go["valor"].pct_change()*100

# df_uf_2023: homicídios por UF (2023)
df_uf_2023 = df_uf_2023.copy()
df_uf_2023["valor"] = pd.to_numeric(df_uf_2023["valor"], errors="coerce")
df_uf_2023_ord = df_uf_2023.sort_values("valor", ascending=False).reset_index(drop=True)

# df_mun_go_2023: homicídios por município de GO (2023)
df_mun_go_2023 = df_mun_go_2023.copy()
# na sua tabela o nome é Qtd_Homicidios
df_mun_go_2023["Qtd_Homicidios"] = pd.to_numeric(df_mun_go_2023["Qtd_Homicidios"], errors="coerce").fillna(0)

# df_final: dataset enriquecido
df_final = df_final.copy()
df_final["Gasto_pc"] = df_final["Gasto_Seguranca"] / df_final["População"]
df_final["Taxa_1000hab"] = pd.to_numeric(df_final["taxa/1000hab"], errors="coerce")
df_final["Qtd_Homicidios"] = pd.to_numeric(df_final["Qtd_Homicidios"], errors="coerce")
df_final["PIB_per_capita"] = pd.to_numeric(df_final["PIB_per_capita"], errors="coerce")
df_final["Gasto_pc"] = pd.to_numeric(df_final["Gasto_pc"], errors="coerce")

# ==============================================
# 1) Série temporal GO + média móvel
# ==============================================
fig = plt.figure(figsize=(10,4.2))
x = df_go["periodo"].dt.year
plt.plot(x, df_go["valor"], label="Homicídios (GO)")
plt.plot(x, df_go["media_movel_3a"], label="Média móvel (3 anos)")
plt.title("Homicídios em Goiás (série histórica)")
plt.xlabel("Ano")
plt.ylabel("Quantidade")
plt.grid(True, alpha=0.3)
plt.legend()
fig.tight_layout()
fig.savefig("figs/01_serie_historica_go.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 2) Variação % ano a ano (GO)
# ==============================================
fig = plt.figure(figsize=(10,4.2))
plt.bar(x, df_go["taxa_var"])
plt.axhline(0, linewidth=1)
plt.title("Variação percentual anual de homicídios – GO")
plt.xlabel("Ano")
plt.ylabel("% vs. ano anterior")
plt.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figs/02_variacao_yoy_go.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 3) Estados em 2023 (ordenado) e destaque para GO
# ==============================================
pos_go = df_uf_2023_ord.index[df_uf_2023_ord["sigla"].eq("GO")][0] + 1
titulo = f"Homicídios por UF em 2023 (ordem decrescente) — GO na posição {pos_go}"
fig = plt.figure(figsize=(10,5))
plt.bar(df_uf_2023_ord["sigla"], df_uf_2023_ord["valor"])
plt.title(titulo)
plt.xlabel("UF")
plt.ylabel("Homicídios (2023)")
plt.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figs/03_ufs_2023.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 4) Top 10 municípios de GO (2023) por número absoluto
# ==============================================
top10_abs = df_mun_go_2023.sort_values("Qtd_Homicidios", ascending=False).head(10)
fig = plt.figure(figsize=(10,5))
plt.barh(top10_abs["Municipio"], top10_abs["Qtd_Homicidios"])
plt.gca().invert_yaxis()
plt.title("Top 10 municípios de GO por homicídios absolutos — 2023")
plt.xlabel("Homicídios (2023)")
plt.ylabel("Município")
plt.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("figs/04_top10_abs.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 5) Distribuição (histograma) homicídios municípios GO (2023)
# ==============================================
fig = plt.figure(figsize=(10,4.2))
vals = df_mun_go_2023["Qtd_Homicidios"].dropna()
bins = max(8, int(np.sqrt(len(vals))))
plt.hist(vals, bins=bins)
plt.title("Distribuição de homicídios por município – GO (2023)")
plt.xlabel("Homicídios no ano")
plt.ylabel("Número de municípios")
plt.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figs/05_hist_municipios_2023.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 6) Maiores taxas por 1.000 hab (Top 10)
# ==============================================
top10_taxa = df_final.sort_values("Taxa_1000hab", ascending=False).head(10)
labels = top10_taxa["Municipio"].str.replace(" - GO","", regex=False)
fig = plt.figure(figsize=(10,5))
plt.barh(labels, top10_taxa["Taxa_1000hab"])
plt.gca().invert_yaxis()
plt.title("Top 10 maiores taxas de homicídios (por 1.000 hab) — 2023")
plt.xlabel("Taxa por 1.000 habitantes")
plt.ylabel("Município")
plt.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("figs/06_top10_taxa.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 7) Boxplot, correlação e histogramas (com df_final)
# ==============================================
import seaborn as sns

vars_modelo = ["Qtd_Homicidios", "Gasto_Seguranca", "valor_icms", "População", "PIB_per_capita"]
df_modelo = df_final[vars_modelo].copy()

# boxplot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_modelo, orient="h", ax=ax)
ax.set_title("Boxplot das variáveis do modelo de regressão")
ax.set_xlabel("Valor")
fig.tight_layout()
fig.savefig("figs/09_boxplot.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# matriz de correlação
corr = df_modelo.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Matriz de correlação das variáveis do modelo")
fig.tight_layout()
fig.savefig("figs/10_matriz_correlacao.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# histogramas
fig = plt.figure(figsize=(12, 8))
df_modelo.hist(bins=20, edgecolor="black", figsize=(12, 8))
plt.suptitle("Distribuição das variáveis utilizadas no modelo", y=1.02)
plt.tight_layout()
fig.savefig("figs/11_histogramas_individuais.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# fecha PDF no final de tudo
pdf.close()
print("✅ Gráficos salvos em: figs/")
