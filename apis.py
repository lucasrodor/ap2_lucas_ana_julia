import requests
import pandas as pd

# ==============================
# Config
# ==============================
BASE = "https://www.ipea.gov.br/atlasviolencia/api/v1"
SERIE_ID = 328  # Homicídios
ARQ_SEG = r"data\despesas_seguranca.xlsx"
ARQ_ICMS = r"data\finbra_icms.xlsx"

# ==============================
# 1) Séries por UF (para medidas)
# ==============================
df_uf = pd.DataFrame(requests.get(f"{BASE}/valores-series/{SERIE_ID}/3", timeout=60).json())
df_go = df_uf[df_uf['sigla'] == "GO"].copy()

# Datas e tipos
df_uf["periodo"] = pd.to_datetime(df_uf["periodo"], errors="coerce")
df_uf["ano"] = df_uf["periodo"].dt.year
df_uf["valor"] = pd.to_numeric(df_uf["valor"], errors="coerce")

# Medidas gerais
df_uf_2023 = df_uf[df_uf["ano"] == 2023].copy()
df_uf_2023["valor"] = pd.to_numeric(df_uf_2023["valor"], errors="coerce")

total_homicidios_estado = df_uf.groupby('sigla', as_index=False)['valor'].sum()
media_homicidios_estado = df_uf.groupby('sigla', as_index=False)['valor'].mean().round(2)
total_homicidios_2023 = df_uf_2023['valor'].sum()
media_homicidios_2023 = df_uf_2023['valor'].mean().round(2)

# Medidas de GO
df_go["valor"] = pd.to_numeric(df_go["valor"], errors="coerce")
df_go['taxa_var'] = df_go['valor'].pct_change() * 100
df_go['media_movel_3a'] = df_go['valor'].rolling(3).mean()

# ==============================
# 2) Códigos dos municípios de GO
# ==============================
df_cod = pd.DataFrame(requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/estados/GO/municipios", timeout=60).json())
codigos = [str(c) for c in df_cod["id"].tolist()]  # sempre string

# ==============================
# 3) Homicídios por município (pega 2023)
# ==============================
df_mun = pd.DataFrame(requests.get(f"{BASE}/valores-series/{SERIE_ID}/4", timeout=60).json())
df_mun_go = df_mun[df_mun["cod"].isin(codigos)].copy()

df_mun_go["periodo"] = pd.to_datetime(df_mun_go["periodo"], errors="coerce")
df_mun_go["ano"] = df_mun_go["periodo"].dt.year
df_mun_go_2023 = df_mun_go[df_mun_go["ano"] == 2023].copy()

df_mun_go_2023.rename(columns={'valor': 'Qtd_Homicidios', 'sigla': 'Municipio'}, inplace=True)
df_mun_go_2023["Qtd_Homicidios"] = pd.to_numeric(df_mun_go_2023["Qtd_Homicidios"], errors="coerce")
df_mun_go_2023["cod"] = df_mun_go_2023["cod"].astype(str).str.zfill(7)

# Top 10 homicídios 2023
top10 = df_mun_go_2023.sort_values('Qtd_Homicidios', ascending=False).head(10)

# ==============================
# 4) Despesa com segurança (planilha)
# ==============================
df_seg = pd.read_excel(ARQ_SEG)
df_seg["Cod.IBGE"] = df_seg["Cod.IBGE"].astype(str).str.zfill(7)

# ==============================
# 5) Merge base: homicídios + segurança
# ==============================
df_final = (
    df_mun_go_2023
      .merge(df_seg, left_on="cod", right_on="Cod.IBGE", how="inner")
      .loc[:, ["cod", "População", "Valor", "UF", "Municipio", "Qtd_Homicidios", "ano"]]
      .rename(columns={"cod": "Codigo IBGE", "ano": "Ano"})
      .copy()
)

# Taxa por 1000 hab (a População será substituída pelo ICMS mais abaixo,
# então recalcularemos depois também, por segurança)
df_final['taxa/1000hab'] = df_final['Qtd_Homicidios'] / df_final['População'] * 1000

# ==============================
# 6) ICMS (trazer valor_icms e POPULAÇÃO do df_icms)
# ==============================
df_icms = pd.read_excel(ARQ_ICMS)
df_icms["Cod.IBGE"] = df_icms["Cod.IBGE"].astype(str).str.zfill(7)

# Use só as colunas necessárias; renomeia Valor -> valor_icms
df_icms_use = df_icms.loc[:, ["Cod.IBGE", "População", "Valor"]].rename(columns={"Valor": "valor_icms"})

# Merge e substituição de população
df_final = (
    df_final
      .merge(df_icms_use, left_on="Codigo IBGE", right_on="Cod.IBGE", how="left")
      .drop(columns=["Cod.IBGE"])
      .rename(columns={"População_x": "População_df_final", "População_y": "População_icms"})
      .copy()
)

# Se existir população do ICMS, substitui
df_final["População"] = df_final["População_icms"].fillna(df_final["População_df_final"])
df_final.drop(columns=["População_df_final", "População_icms"], inplace=True)

# Recalcula taxa por 1000 hab com a população final
df_final['taxa/1000hab'] = df_final['Qtd_Homicidios'] / df_final['População'] * 1000

# ==============================
# 7) PIB municipal (último ano disponível) e per capita
# ==============================
URL_PIB = "https://apisidra.ibge.gov.br/values/t/5938/n6/all/n3/52/v/37/p/last?formato=json"
df_raw = pd.DataFrame(requests.get(URL_PIB, timeout=60).json()[1:])

df_pib = (
    df_raw.rename(columns={"D1C": "codigo_municipio", "D1N": "municipio", "D3N": "ano", "V": "valor_pib"})
          .assign(codigo_municipio=lambda d: d["codigo_municipio"].astype(str).str.zfill(7),
                  ano=lambda d: d["ano"].astype(int),
                  valor_pib=lambda d: pd.to_numeric(d["valor_pib"], errors="coerce") * 1000)  # mil R$ -> R$
          .loc[:, ["codigo_municipio", "valor_pib", "ano"]]
          .rename(columns={"ano": "Ano_PIB"})
          .copy()
)

# Um registro por município (já é "last" na URL, mas garantimos)
df_pib_latest = (
    df_pib.sort_values(["codigo_municipio", "Ano_PIB"])
          .drop_duplicates("codigo_municipio", keep="last")
)

# Merge PIB
df_final = (
    df_final.merge(df_pib_latest, left_on="Codigo IBGE", right_on="codigo_municipio", how="left")
            .drop(columns=["codigo_municipio"])
            .copy()
)

# PIB per capita
df_final["PIB_per_capita"] = (df_final["valor_pib"] / df_final["População"]).round(2)

# ==============================
# 8) Limpeza e ordem final de colunas
# ==============================
df_final.rename(columns={"Valor": "Gasto_Seguranca", "valor_pib": "PIB"}, inplace=True)
df_final = df_final[
    ["Codigo IBGE", "Municipio", "População", "Gasto_Seguranca", "valor_icms",
     "Qtd_Homicidios", "taxa/1000hab", "PIB", "PIB_per_capita", "Ano", "Ano_PIB"]
].copy()

# ==============================
# 9) (Opcional) salvar resultados
# ==============================
# df_final.to_excel("dados_completos.xlsx", index=False)

# ==============================
# 10) Objetos finais (os mesmos que você lista no fim)
# ==============================
# Séries/medidas e dataframes pedidos
_ = (
    df_go,                 # série GO com taxa_var e média móvel
    codigos,               # lista de códigos
    df_mun_go_2023,        # homicídios municipais 2023
    df_uf_2023,            # homicídios UF 2023
    total_homicidios_estado,
    media_homicidios_estado,
    total_homicidios_2023,
    media_homicidios_2023,
    df_go['taxa_var'],
    df_go['media_movel_3a'],
    top10,
    df_final               # -> TABELA FINAL LIMPA
)

print("df_final pronto!")
print(df_final.head())

# ==============================
# 11) Salvar o resultado final em Excel
# ==============================

# Caminho onde o arquivo será salvo (você pode mudar o nome ou a pasta)
caminho_saida = r"data\dados_completos_final.xlsx"

# Salva em Excel (sem o índice)
df_final.to_excel(caminho_saida, index=False)