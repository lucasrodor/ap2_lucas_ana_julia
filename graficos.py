import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from apis import df_go, df_final, df_filtrado, df_uf, df_uf_2023, df_mun_go_2023, df_pop_go_2022, df_pib
from apis import total_homicidios_estado, media_homicidios_estado, total_homicidios_2023, media_homicidios_2023


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
df_mun_go_2023["qtd_homicidio"] = pd.to_numeric(df_mun_go_2023["qtd_homicidio"], errors="coerce")

# df_final: dataset enriquecido
df_final = df_final.copy()
# nomes de colunas conforme você gerou:
# ['Codigo IBGE','Municipio','População','Gasto_Seguranca','Qtd_Homicidios',
#  'PIB','PIB_per_capita','taxa/1000hab','Ano','Ano_PIB']
# cria métricas derivadas úteis
df_final["Gasto_pc"] = df_final["Gasto_Seguranca"] / df_final["População"]
df_final["Taxa_1000hab"] = df_final["taxa/1000hab"]
# garante tipos
df_final["Qtd_Homicidios"] = pd.to_numeric(df_final["Qtd_Homicidios"], errors="coerce")
df_final["Taxa_1000hab"] = pd.to_numeric(df_final["Taxa_1000hab"], errors="coerce")
df_final["PIB_per_capita"] = pd.to_numeric(df_final["PIB_per_capita"], errors="coerce")
df_final["Gasto_pc"] = pd.to_numeric(df_final["Gasto_pc"], errors="coerce")

# ==============================================
# 1) Série temporal GO + média móvel (1989–2023)
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
# posição de GO
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
top10_abs = df_mun_go_2023.sort_values("qtd_homicidio", ascending=False).head(10)
fig = plt.figure(figsize=(10,5))
plt.barh(top10_abs["sigla"], top10_abs["qtd_homicidio"])
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
vals = df_mun_go_2023["qtd_homicidio"].dropna()
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
# 7) Dispersão: PIB per capita x Taxa por 1.000 hab
# ==============================================
base_disp = df_final.dropna(subset=["PIB_per_capita","Taxa_1000hab"]).copy()
fig = plt.figure(figsize=(8,6))
plt.scatter(base_disp["PIB_per_capita"], base_disp["Taxa_1000hab"])
plt.title("PIB per capita × Taxa de homicídios (por 1.000 hab)")
plt.xlabel("PIB per capita (R$)")
plt.ylabel("Taxa por 1.000 hab")
plt.grid(True, alpha=0.3)

# linha de tendência (OLS simples)
xv = base_disp["PIB_per_capita"].values
yv = base_disp["Taxa_1000hab"].values
if len(xv) > 1:
    b1, b0 = np.polyfit(xv, yv, 1)
    xx = np.linspace(xv.min(), xv.max(), 100)
    plt.plot(xx, b1*xx + b0, linewidth=2)
fig.tight_layout()
fig.savefig("figs/07_scatter_pibpc_taxa.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# ==============================================
# 8) Dispersão: Gasto em segurança per capita x Taxa por 1.000 hab
# ==============================================
base_gasto = df_final.dropna(subset=["Gasto_pc","Taxa_1000hab"]).copy()
fig = plt.figure(figsize=(8,6))
plt.scatter(base_gasto["Gasto_pc"], base_gasto["Taxa_1000hab"])
plt.title("Gasto em segurança per capita × Taxa de homicídios (por 1.000 hab)")
plt.xlabel("Gasto em segurança per capita (R$)")
plt.ylabel("Taxa por 1.000 hab")
plt.grid(True, alpha=0.3)

# tendência
xg = base_gasto["Gasto_pc"].values
yg = base_gasto["Taxa_1000hab"].values
if len(xg) > 1:
    b1g, b0g = np.polyfit(xg, yg, 1)
    xxg = np.linspace(xg.min(), xg.max(), 100)
    plt.plot(xxg, b1g*xxg + b0g, linewidth=2)
fig.tight_layout()
fig.savefig("figs/08_scatter_gastopc_taxa.png", dpi=220, bbox_inches="tight")
pdf.savefig(fig); plt.close(fig)

# fecha PDF
pdf.close()

print("✅ Gráficos salvos em: figs/")



