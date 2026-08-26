# Entrega 1 — Fundamentos de Manipulação e Visualização de Dados

Compilado das três entregas · Disciplina de Automação e IA · Google Colab

**Tese.** Convenções e proveniência determinam conclusões tanto quanto os dados.
Três conjuntos de confiabilidade decrescente demonstram sete modos de falha, nenhum
dos quais produz erro de execução.

---

## 1. Arquivos e ordem de execução

| Arquivo | Papel |
|---|---|
| `nucleo.py` | Convenções, carga com cache, auditoria, paleta. Importado pelas três |
| `Entrega_1.1_Titanic_Calibracao.ipynb` | Calibração e auditoria de convenções |
| `Entrega_1.2_IoT_Proveniencia.ipynb` | Perícia de proveniência, com validação |
| `Entrega_1.3_RT_IoT2022_Baseline.ipynb` | Análise substantiva e síntese das três |
| `resultados/*.json` | Achados registrados por cada notebook |
| `titanic.csv`, `d2.csv`, `d3.csv` | Conjuntos de dados |

**Ordem obrigatória:** 1.1 → 1.2 → 1.3. A Seção 11 da Entrega 1.3 lê os três JSONs e
interrompe a execução se algum estiver ausente ou se as convenções divergirem.

---

## 2. Estrutura de cada notebook

Formato de relatório técnico, uniforme nas três:

1. Identificação do conjunto
2. Escopo — questão de pesquisa, papel na entrega, pré-requisitos
3. Matriz de rastreabilidade — requisito do Desafio Final → seção → evidência
4. Convenções e premissas declaradas
5. Preparação do ambiente e calibração
6. Auditoria de entrada
7. Análises, cada uma no ciclo **Objetivo / Método / Formalização / Resultado e interpretação**
8. Registro dos achados em JSON
9. Conclusões e recomendações
10. Apêndice A — registro de decisões (decisão, alternativa descartada, justificativa)
11. Apêndice B — referência rápida

---

## 3. Quadro comparativo

| | Entrega 1.1 | Entrega 1.2 | Entrega 1.3 |
|---|---|---|---|
| Conjunto | Titanic | IoT Security & Energy | RT-IoT2022 |
| Fonte | seaborn-data | Kaggle (`zara2099`) | UCI ML Repository #942 |
| Dimensões | 891 × 15 | 4,000 × 24 | 123,117 × 85 |
| Papel | calibração | controle | objeto de estudo |
| Proveniência | canônica | declarada, sem lastro | documentada |
| Veredito | real | **sintetico** | real ou inconclusivo |
| Vars. informativas | — | **1** de 19 | **66** de 82 |
| md5 | `56f29cc0b807` | `cd09503e4bec` | `87ec8763e33f` |

---

## 4. Sete modos de falha demonstrados

| # | Entrega | Modo de falha | Evidência | Efeito |
|---|---|---|---|---|
| 1 | 1.1 | Convenção silenciosa de quantil | setosa: **4** outliers (tipo 7) contra **1** (tipo 1) | critério de exclusão muda sem aviso |
| 2 | 1.1 | Imputação anterior à análise | dispersão de `age` contrai **10.49%** | associação com o desfecho é atenuada |
| 3 | 1.1 | Agregação sobre grupos heterogêneos | r = **-0.1176** agregado contra **[0.7425, 0.5259, 0.4572]** | sinal da relação se inverte |
| 4 | 1.1 | Remoção acrítica de duplicatas | **107** duplicatas legítimas, 125/160 na 3ª classe | `drop_duplicates` eliminaria entidades distintas |
| 5 | 1.2 | Proveniência não verificada | \\|r\\| máx **0.0461**; **15/20** colunas uniformes | EDA produz ruído apresentado como achado |
| 6 | 1.3 | Acurácia global sob desbalanceamento | **3380.7:1**; regra constante acerta **76.89%** | modelo sem capacidade preditiva parece adequado |
| 7 | 1.3 | Assinatura tomada por aprendizado | `fwd_URG_flag_count` → `NMAP_XMAS_TREE_SCAN` (η² = 0.998) | modelo redescobre a definição do ataque |

---

## 5. Resultado substantivo (Entrega 1.3)

**Os dispositivos IoT em operação normal possuem assinatura comportamental própria.**
**37 das 80** variáveis avaliadas os separam entre si, com
predominância das temporais (`idle.*`, `*_iat.*`). A assinatura reside no comportamento
temporal, não no conteúdo do pacote — conteúdo cifrado não impede o perfilamento.

### Estabilidade da linha de base

| Dispositivo | QCD de `flow_duration` | Zeros em `idle.min` | Regularidade |
|---|---|---|---|
| MQTT_Publish | **0.347** | 5.0% | alta — publicação periódica |
| Thing_Speak | **0.934** | 98.9% | baixa |
| Wipro_bulb | **0.998** | 68.4% | muito baixa — acionamento humano |

**Recomendação operacional:** limiar de anomalia calibrado por dispositivo. Um limiar
ajustado ao MQTT (QCD 0,35) aplicado ao Wipro_bulb (QCD 1,00) produziria alarme
praticamente contínuo.

---

## 6. Sobre cruzamento e balanceamento entre conjuntos

**Não há junção possível entre as três entregas.** Não existe chave, entidade nem
unidade de observação comum: a unidade de uma linha é, respectivamente, um passageiro,
um registro fabricado e um fluxo de rede. Uma junção fabricaria relação inexistente —
o mesmo erro que a Entrega 1.2 documenta.

O que é compartilhado é o **método**, não os registros. A comparação da Seção 11 da
Entrega 1.3 é metodológica, e o `assert` de convenções é o que lhe dá base.

**Cruzamento interno** (`pivot_table`, `crosstab`, `groupby` multichave) é aplicado em
1.1 (sexo × classe, faixa etária × sexo) e 1.3 (classe × protocolo, faixa de duração ×
natureza do tráfego).

**Balanceamento de classes não é aplicado.** Reamostrar antes da análise exploratória
descreveria a amostra sintética, não a rede — mesmo mecanismo do modo de falha 2.
Balanceamento pertence ao estágio de modelagem, dentro do fold de treino.

**Comparação entre conjuntos de tamanhos distintos** foi validada: a Seção 8.2 da
Entrega 1.2 mostra que o veredito do RT-IoT2022 é invariante à subamostragem para
n = 4.000 e n = 1.000.

---

## 7. Reprodutibilidade

- Convenções idênticas nas três, fixadas em `nucleo.CONVENCOES`: `{'quantil': 'linear', 'ddof': 1, 'outlier': '1.5*IQR', 'corr': 'pearson'}`
- A Seção 11.1 da Entrega 1.3 interrompe a execução se as convenções divergirem
- `nu.calibrar()` executa no início das três e falha se `ddof` ou o método de quantil mudarem
- Conjuntos identificados por md5 em cada JSON
- A síntese não recalcula nenhum valor — lê os JSONs

---

## 8. Execução no Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q scipy seaborn pyarrow
# ajustar DADOS na primeira célula de cada notebook:
DADOS = "/content/drive/MyDrive/Entrega1"
```

`nucleo.carregar()` gera cache `.parquet` na primeira leitura — medido em 10× a 106×
mais rápido que `read_csv` nas releituras, relevante para o `d3.csv` de 55 MB.

---

## 9. Avisos

**Versão do Titanic.** A calibração usa os valores conhecidos do `titanic.csv` do
seaborn-data. O `train.csv` da competição Kaggle tem esquema distinto (`Survived`,
`Pclass` 1/2/3) e a calibração falhará — corretamente, porque o contrato não corresponde.
Para utilizá-lo, ajustar `OURO_TITANIC` em `nucleo.py`.

**Entrega 1.2.** Nenhum resultado substantivo do conjunto sintético entra na conclusão
final. A redação adotada é "conjunto sintético cuja documentação não declara a natureza
sintética" — é o que a evidência sustenta.

**Correções registradas.** Três interpretações iniciais foram contrariadas pela execução
e estão documentadas nos Apêndices A, não suprimidas: as 107 duplicatas do Titanic são
legítimas; a interação sexo × classe tem direção oposta conforme a métrica (absoluta ou
relativa); e a métrica de dispersão IQR/mediana é indefinida sob inflação de zeros,
tendo sido substituída pelo QCD.
