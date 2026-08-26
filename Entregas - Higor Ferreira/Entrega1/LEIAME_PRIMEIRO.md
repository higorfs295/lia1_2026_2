# 🚀 Entrega 1 — NumPy, Pandas e confiabilidade de dados

Três notebooks, três datasets e uma mesma pergunta:

> **Como evitar conclusões convincentes produzidas por dados ou convenções pouco confiáveis?**

O projeto foi preparado para o Google Colab, está executado e mantém a sequência **1.1 → 1.2 → 1.3**.

---

## ⚡ Comece aqui

1. Envie a pasta `Entrega1` para `Meu Drive`.
2. Abra `Entrega_1.1_Titanic_Calibracao.ipynb` no Colab.
3. Execute `Ambiente de execução → Executar tudo`.
4. Repita com os notebooks 1.2 e 1.3, nessa ordem.

> ✅ A primeira célula de cada notebook monta o Drive, usa `/content/drive/MyDrive/Entrega1`, instala apenas pacotes ausentes, confere os arquivos e mostra o que está pronto ou pendente.

Se a pasta tiver outro nome, altere o campo visual `PASTA_PROJETO` na primeira célula.

---

## 📁 Estrutura esperada

```text
MyDrive/Entrega1/
├── Entrega_1.1_Titanic_Calibracao.ipynb
├── Entrega_1.2_IoT_Proveniencia.ipynb
├── Entrega_1.3_RT_IoT2022_Baseline.ipynb
├── nucleo.py
├── titanic.csv
├── iris.csv
├── d2.csv
├── d3.csv
├── LEIAME_PRIMEIRO.md
├── RELATORIO_VALIDACAO.md
└── resultados/
    ├── entrega_1_1.json
    ├── entrega_1_2.json
    └── entrega_1_3.json
```

Os CSVs pequenos já estão incluídos. Se `titanic.csv` ou `iris.csv` forem removidos, a célula inicial tenta baixá-los novamente do repositório `seaborn-data`.

---

## 🧭 O papel de cada notebook

| Ordem | Notebook | Pergunta central | Resultado |
|---:|---|---|---|
| 1 | **Titanic** | As convenções e transformações alteram conclusões silenciosamente? | Calibra o pipeline e demonstra três modos de falha |
| 2 | **IoT Security & Energy** | O dataset sustenta a proveniência que declara? | Identifica sinais fortes de geração sintética |
| 3 | **RT-IoT2022** | Dispositivos normais possuem assinatura comportamental própria? | Encontra 37 de 80 variáveis separadoras e recomenda limiar por dispositivo |

### 🔗 Encadeamento

```text
1.1 calibra convenções
        ↓
1.2 testa proveniência
        ↓
1.3 registra seus achados e consolida os 3 JSONs
```

O notebook 1.3 agora grava `entrega_1_3.json` **antes** de abrir a síntese. Isso elimina a dependência circular que existia na versão anterior.

---

## 🎨 Como ler os notebooks

- **⚡ Em 30 segundos:** resultado principal logo no início.
- **🗺️ Roteiro:** mostra o caminho antes dos detalhes.
- **🎯 Pergunta:** explica o objetivo de cada análise.
- **🧭 Como a análise foi feita:** bloco recolhível para reduzir ruído visual.
- **✅ Leitura principal:** interpretação imediatamente após a evidência.
- **🧰 Apêndices:** decisões e referência rápida ficam recolhíveis.

O visual segue a linguagem do material da disciplina: títulos com emojis, hierarquia clara, dicas curtas, exemplos executáveis e paleta azul-marinho/dourado.

---

## 📊 Resultados em uma página

| Evidência | Resultado |
|---|---:|
| Redução da dispersão de `age` após imputação pela média | **10,49%** |
| Outliers da setosa conforme a convenção de quantil | **4 vs. 1** |
| Correlação agregada no Iris | **−0,1176** |
| Variáveis uniformes no dataset 1.2 | **15 de 20** |
| Correlação máxima entre variáveis no dataset 1.2 | **0,0461** |
| Desbalanceamento máximo no RT-IoT2022 | **3380,7 : 1** |
| Acurácia da classe majoritária | **76,89%** |
| Variáveis que separam dispositivos normais | **37 de 80** |

---

## 🗂️ Fontes dos dados

- [Titanic — competição Kaggle](https://www.kaggle.com/competitions/titanic/data)
- [IoT Network Security and Energy Dataset — Kaggle](https://www.kaggle.com/datasets/zara2099/iot-network-security-and-energy-dataset)
- [RT-IoT2022 — Kaggle](https://www.kaggle.com/datasets/joebeachcapital/real-time-internet-of-things-rt-iot2022)
- [RT-IoT2022 — fonte primária UCI, dataset 942](https://archive.ics.uci.edu/dataset/942/rt-iot2022)

### ⚠️ Nota sobre o Titanic

A seleção do projeto é a competição Titanic do Kaggle, mas o notebook de calibração usa o espelho `seaborn-data`. Ele contém os mesmos passageiros clássicos e acrescenta campos didáticos como `alive`, `class` e `who`.

O arquivo `train.csv` do Kaggle possui outro esquema (`Survived`, `Pclass`, etc.) e não deve substituir `titanic.csv` sem adaptação do notebook e dos valores de calibração.

---

## 🛠️ Soluções rápidas

### A pasta não foi encontrada

Confirme o caminho:

```text
/content/drive/MyDrive/Entrega1
```

Se necessário, edite `PASTA_PROJETO` na primeira célula.

### `d2.csv` ou `d3.csv` está pendente

Confirme os nomes exatos:

- `IoT_Security_Dataset_Ultimate.csv` → `d2.csv`
- `RT_IOT2022.csv` → `d3.csv`

### Apareceram arquivos `.parquet`

É esperado. Eles são caches de leitura criados ao lado dos CSVs. Podem ser apagados; serão recriados na próxima execução.

### O módulo foi editado durante a sessão

Use `Ambiente de execução → Reiniciar sessão` antes de executar tudo novamente.

---

## ✅ Estado da entrega

Os três notebooks foram executados integralmente, na ordem, sem erros. Os 13 gráficos foram revisados e os três JSONs foram regenerados a partir dos dados incluídos.

Detalhes: consulte `RELATORIO_VALIDACAO.md`.
