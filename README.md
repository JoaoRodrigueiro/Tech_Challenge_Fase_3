Tech Challenge — Fase 3

Este projeto foi desenvolvido como parte da pós-graduação, atendendo aos requisitos da Fase 4.
A aplicação realiza geração automática de descrições oficiais de produtos da Amazon a partir do título fornecido, utilizando técnicas de Fine-Tuning leve (LoRA) e RAG (Retrieval-Augmented Generation) sobre o dataset LF-AmazonTitles-1.3MM.
________________________________________
Objetivo

Dado o título de um produto, o sistema deve:
•	Gerar a descrição oficial em português (PT-BR)
•	Utilizar apenas evidências recuperadas do dataset
•	Informar o UID da fonte usada como contexto
•	Garantir consistência factual via RAG
________________________________________
Abordagem Utilizada

A solução combina duas estratégias principais:
•	Fine-tuning supervisionado (SFT) de um foundation model usando LoRA
•	RAG denso com busca vetorial utilizando FAISS

Fluxo geral:

1.	O título é convertido em embedding.
2.	O índice FAISS recupera os k títulos mais similares.
3.	A descrição correspondente é utilizada como contexto com UID.
4.	O modelo fine-tunado gera a descrição final em PT-BR.
5.	Aplicação de pós-processamento e fallback quando necessário.
________________________________________
Funcionalidades

•	Geração automática de descrição oficial em PT-BR
•	Recuperação de evidência via busca vetorial densa
•	Citação do UID como fonte da resposta
•	Tradução automática na inferência
•	Avaliação automática via ROUGE-L
•	Execução via CLI
________________________________________
Tecnologias Utilizadas

•	Python 3.11
•	Hugging Face Transformers
•	LoRA (Low-Rank Adaptation)
•	FAISS (IndexFlatIP)
•	Google FLAN-T5 Small
•	Multilingual E5 (embeddings densos)
•	Helsinki-NLP Opus MT (EN → PT)
•	NumPy
•	Pandas
________________________________________
Stack do Projeto

Base Model:
•	google/flan-t5-small

Fine-Tuning:
•	LoRA (r=8, alpha=16, dropout=0.05)
•	target_modules = ["q","k","v","o"]

RAG:
•	intfloat/multilingual-e5-small
•	FAISS (IndexFlatIP)

Tradução:
•	Helsinki-NLP/opus-mt-tc-big-en-pt

Scripts principais:
•	prepare_dataset.py
•	fine_tune.py
•	build_dense_index_full.py
•	cli.py
•	eval.py
________________________________________
Dataset

Fonte:
•	LF-AmazonTitles-1.3MM (data/trn.json.gz)

Campos utilizados:
•	title (entrada)
•	content (descrição oficial)

Amostra SFT:
•	Train: 1.900 exemplos
•	Val: 100 exemplos

Arquivos gerados:

•	data/processed/sft_train.jsonl
•	data/processed/sft_val.jsonl
________________________________________
Resultados

Avaliação realizada em 100 exemplos utilizando ROUGE-L (F1):
Baseline (sem contexto):
0.0462
Fine-Tuning + RAG:
0.4413
Ganho aproximado de 9,5 vezes ao combinar Fine-Tuning com Recuperação de Contexto.
Isso mostra que o modelo isolado não possui capacidade suficiente para reconstruir descrições específicas, mas com evidência recuperada o desempenho se aproxima significativamente do conteúdo oficial.
________________________________________
Arquitetura Resumida

1.	Busca vetorial densa no índice FAISS.
2.	Recuperação de descrição + UID como evidência.
3.	Prompt controlado instruindo o modelo a usar apenas o contexto.
4.	Geração da descrição em PT-BR.
5.	Pós-processamento e fallback.
________________________________________
Modo de Executar o Projeto (Windows / CPU)

Requisitos:
•	Python 3.11
•	Ambiente virtual (venv)
No PowerShell, na raiz do projeto:

py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

Depois: 

python prepare_dataset.py
python fine_tune.py
python build_dense_index_full.py
python cli.py

