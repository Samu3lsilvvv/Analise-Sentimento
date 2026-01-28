# Classificação de Sentimentos em Reviews de Produtos (Positivo, Negativo, Neutro)

Este repositório apresenta um projeto completo de **Machine Learning** para automatizar a análise de sentimentos em reviews de produtos de e-commerce, classificando-os em **positivo** e **negativo**.

## Índice
- [Visão Geral](#visão-geral)
- [Problema de Negócio](#problema-de-negócio)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Metodologia](#metodologia)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Código Implementado](#código-implementado)
- [Instalação e Uso](#instalação-e-uso)
- [Deploy e Produção](#deploy-e-produção)
- [Contribuição](#contribuição)

## Visão Geral

Este projeto implementa uma solução de **Processamento de Linguagem Natural (NLP)** e **Machine Learning** para classificar automaticamente o sentimento de reviews de produtos em três categorias. A solução percorre todo o fluxo de trabalho de um projeto de Data Science, desde a definição do problema até o deploy do modelo em produção.

**Principais características:**
-  Classificação multiclasse (positivo, negativo, neutro)
-  Pipeline completo de pré-processamento de texto
-  Múltiplos algoritmos testados e comparados
-  Sistema de deploy pronto para produção
-  Métricas de avaliação robustas para problemas multiclasse
-  Balanceamento automático de classes
-  Visualizações detalhadas dos resultados

##  Problema de Negócio

### Contexto
Uma empresa de e-commerce precisa classificar automaticamente reviews em três categorias para:
- **Identificar clientes satisfeitos** (positivo) - potencial para depoimentos e marketing
- **Priorizar reclamações** (negativo) - ação rápida da equipe de suporte
- **Analisar feedback neutro** (neutro) - identificar oportunidades de melhoria

### Objetivo
Desenvolver um modelo de ML que classifique automaticamente reviews como:
-  **Positivo** (2) - Cliente satisfeito
-  **Neutro** (1) - Cliente neutro/indiferente
-  **Negativo** (0) - Cliente insatisfeito

### Benefícios Esperados
- **Eficiência operacional**: Redução de 85% no tempo de análise manual
- **Priorização inteligente**: Reviews negativos direcionados automaticamente ao suporte
- **Insights estratégicos**: Identificação de padrões em feedbacks neutros
- **Satisfação do cliente**: Resposta mais rápida a problemas
- **Tomada de decisão**: Dados estruturados para melhorias de produto

##  Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|---------|-----|
| Python | 3.9+ | Análise e modelagem |
| Pandas | 2.3.1 | Manipulação de dados |
| numpy  | 2.3.2 | Computação numérica |
| Matplotlib/Seaborn | 3.10/0.13.2 | Visualizações estáticas |
| joblib | 1.4.2 | Serialização de modelos |
| imbalanced-learn  | 5.24.1 | Visualizações interativas |
| re  | 2.2.1 | Expressões regulares |
| scikit-learn | 1.6.1 | Machine Learning |

#  Pacotes Específicos

## Bibliotecas Principais para Machine Learning

### **TfidfVectorizer** 
- **Função**: Vetorização de texto com TF-IDF (Term Frequency-Inverse Document Frequency)
- **Uso**: Converte texto em representação numérica ponderada

### **LogisticRegression** 
- **Função**: Regressão logística multiclasse
- **Uso**: Classificação probabilística para múltiplas categorias

### **RandomForestClassifier** 
- **Função**: Ensemble com árvores de decisão
- **Uso**: Combina múltiplas árvores para melhor generalização

### **SVC** 
- **Função**: Máquinas de vetores de suporte (Support Vector Classifier)
- **Uso**: Classificação com margens ótimas entre classes

### **XGBoost** 
- **Função**: Gradient boosting otimizado
- **Uso**: Ensemble sequencial com boosting

### **SMOTE** 
- **Função**: Oversampling para balanceamento de classes
- **Uso**: Gera amostras sintéticas de classes minoritárias

### **GridSearchCV** 
- **Função**: Otimização de hiperparâmetros
- **Uso**: Busca sistemática das melhores configurações

---

# Metodologia

O projeto segue um fluxo de trabalho estruturado em **10 etapas**:

## 1. Definição do Problema
- **Análise do contexto empresarial**
- **Definição de métricas de sucesso multiclasse**
- **Estabelecimento de objetivos claros** para três categorias

## 2. Coleta e Expansão de Dados
- **Dataset original**: 500 reviews
- **Expansão com 150 exemplos neutros** (gerados automaticamente)
- **Dataset final**: 650 reviews balanceados

## 3. Pré-processamento Avançado
- **Remoção de acentos e caracteres especiais**
- **Normalização de texto** (minúsculas)
- **Remoção de URLs, menções, hashtags**
- **Filtro de stopwords em português**
- **Preservação de contexto emocional** (!, ?)

## 4. Análise Exploratória (EDA)
- **Distribuição das três classes**
- **Comprimento médio dos reviews por sentimento**
- **Palavras mais frequentes por categoria**
- **Identificação de padrões linguísticos**

## 5. Engenharia de Features
- **Vetorização TF-IDF com n-grams (1,2)**
- **Seleção de 2000 features mais relevantes**
- **Normalização com StandardScaler**

## 6. Balanceamento de Classes
- **Uso de SMOTE** (Synthetic Minority Oversampling Technique)
- **Garantia de distribuição equilibrada no treino**
- **Prevenção de viés para classes majoritárias**

## 7. Divisão dos Dados
- **75% treino / 25% teste**
- **Estratificação por três classes**
- **Random state para reprodutibilidade**

## 8. Modelagem Multiclasse
- **Pipeline integrado**: pré-processamento + modelo
- **4 algoritmos testados**: **Regressão Logística Multinomial, Random Forest, Support Vector Machines, XGBoost** 

## 9. Avaliação Multiclasse
- **Métricas por classe e agregadas**
- **Matriz de confusão 3x3**
- **F1-Score weighted** (métrica principal)
- **Relatórios de classificação detalhados**

## 10. Deploy e Produção
- **Serialização do melhor modelo**
- **Função de predição para novos dados**
- **Sistema de recomendação baseado no sentimento**

## Métricas do Melhor Modelo

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **Acurácia** | 84.2% | Porcentagem total de acertos |
| **F1-Score (Weighted)** | 83.8% | Média harmônica balanceada |
| **Precisão (Positivo)** | 86.1% | Acertos entre previsões positivas |
| **Recall (Negativo)** | 82.3% | Negativos reais identificados |
| **AUC-ROC (Macro)** | 0.891 | Área sob a curva ROC |

# Instalação Passo a Passo

## 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/classificacao-sentimentos-multiclasse.git
cd classificacao-sentimentos-multiclasse
```

## 2. Criar ambiente virtual (recomendado)
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate


# Windows
python -m venv venv
venv\Scripts\activate
```

## 3. Executar o projeto
```bash
# Modo completo (todas as etapas)
python main.py

# Modo específico
python main.py --ajuda  # Ver opções disponíveis


```
