# Modelagem Supervisionada para Predição de Pagamento de Mensalidades

Este projeto tem como objetivo construir um modelo preditivo capaz de **estimar a probabilidade** de pagamento de uma mensalidade por parte de um aluno após a realização de uma ação de cobrança. Os dados utilizados compreendem um histórico de mensalidades e as ações de cobrança realizadas por uma empresa responsável por esse processo em certas faculdades privadas.

## 1. Objetivo do Projeto

O objetivo central é oferecer modelos que auxiliem na tomada de decisão, permitindo à empresa:

- Identificar alunos com maior ou menor probabilidade de efetuar o pagamento;
- Priorizar os casos com maior risco de inadimplência;
- Evitar ações de cobrança desnecessárias com alunos que provavelmente irão pagar;
- Otimizar os recursos de cobrança de forma mais estratégica.

## 2. Associação entre Ações de Cobrança e Mensalidades

Como os dados de cobrança não continham um identificador direto de vínculo com as mensalidades, foi desenvolvida uma regra de associação temporal. Para cada ação de cobrança, buscou-se a mensalidade do mesmo aluno com data de vencimento mais próxima da data da ação, dentro de uma janela de até 10 dias. Em caso de múltiplas possibilidades, foi escolhida a com menor diferença absoluta.

Essa associação permitiu criar um novo conjunto de dados, no qual cada linha representa uma ação de cobrança associada a uma mensalidade específica, contendo variáveis como:

- `acao_cobranca` (tipo de ação realizada);
- `dias_dif` (diferença em dias entre a ação e o vencimento da mensalidade);
- `foi_pago` (variável-alvo que indica se a mensalidade foi paga, sendo 1 para pagamento, 0 para não pagamento).

## 3. Resultados

Foram avaliados os seguintes modelos de classificação:

- Regressão Logística  
- Decision Tree  
- Random Forest  
- XGBoost  
- Gradient Boosting  
- LightGBM  
- Naive Bayes  
- KNN  

Os modelos foram avaliados com base em métricas como **AUC**, **recall**, **F1-score** e **acurácia balanceada**, considerando diferentes objetivos estratégicos.

| Objetivo                                 | Modelo               | Destaques                                      |
|------------------------------------------|----------------------|-----------------------------------------------|
| Prever quem irá pagar (`foi_pago = 1`)    | Regressão Logística / Naive Bayes | Recall: 0.83, F1-score: 0.70       |
| Estimar probabilidades                   | XGBoost / LightGBM   | AUC ≈ 0.65, bom equilíbrio geral              |
| Identificar inadimplência (`foi_pago = 0`) | Random Forest        | Recall: 0.68, F1-score: 0.60                 |

## 4. Estratégias de Modelagem

- Análise exploratória dos dados;
- Regra de associação entre ações e mensalidades;
- Criação de features a partir de atributos temporais e de ação;
- Treinamento e comparação de modelos preditivos;
- Interpretação dos resultados conforme o foco da empresa (pagamento ou inadimplência).

## 5. Ferramentas Utilizadas

- Python 3.10;
- Jupyter Notebook, com execução via VSCode;
- Bibliotecas:
  - `pandas` e `numpy` para manipulação e análise de dados;
  - `matplotlib` e `seaborn` para visualização;
  - `scikit-learn`, `xgboost` e `lightgbm` para modelagem preditiva.

## 6. Possíveis Melhorias

- Aplicação de validação cruzada para maior robustez;
- Uso de técnicas como SMOTE ou ADASYN para balancear as classes;
- Ajuste de limiares de decisão com base em métricas específicas;
- Calibração das probabilidades previstas pelos modelos;
- Testes com novos atributos e algoritmos como SVM ou CatBoost.

## Autor

**Ricardo Luís Bertolucci Filho**  

- [LinkedIn](https://www.linkedin.com/in/ricardo-lu%C3%ADs-bertolucci-filho/)
- [GitHub](https://github.com/ric-rky/ric-rky)
- E-mail: bertolucci.rl@gmail.com

##

Notebook principal: [`teste_principia.ipynb`](https://github.com/ric-rky/Predicao-de-pagamento-de-mensalidades/blob/main/predicao_mensalidades.ipynb)
