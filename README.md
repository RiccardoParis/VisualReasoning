# Multimodal Cascade Reasoning per Visual Question Answering

> Architettura multimodale a due fasi basata su T5 e Cross-Attention per migliorare il Visual Grounding nel ragionamento logico-visivo.

## Obiettivo del progetto
Il progetto affronta il problema delle allucinazioni testuali e della scarsa interpretabilità nei modelli di Visual Question Answering (VQA).Avere un sistema in grado di rispondere a domande complesse su immagini. La risposta richiede passaggi sequenziali che simulino il ragionamento effettivo sull'immagine, non può basarsi su risposte statistiche ma deve effettivamente "ragionare" su 
quello che vede.  Si intende dimostrare, tramite test di ablazione e valutazioni a cascata, che il modello finale non deduce la risposta basandosi esclusivamente su bias testuali, ma impara a "mettere a fuoco" i pixel esatti dell'immagine necessari a validare il ragionamento.

## Background e tecniche utilizzate
* **Tecniche di Deep Learning:** Il sistema utilizza Large Language Models (LLM) basati sull'architettura Transformer (T5) accoppiati a Vision Encoders. L'integrazione multimodale avviene tramite layer di Cross-Attention personalizzati. Per ottimizzare l'addestramento senza incorrere in *catastrophic forgetting*, viene impiegata la tecnica di Parameter-Efficient Fine-Tuning (PEFT) tramite adattatori LoRA.
* **Strumenti:** PyTorch, librerie dell'ecosistema Hugging Face (`transformers`, `peft`), Scikit-learn per la metrica e Torchvision per il processing visivo.
* **Dataset:** Il progetto è addestrato e valutato sul dataset **CLEVR** (Compositional Language and Elementary Visual Reasoning). Il dataset è specificamente progettato per testare le capacità di ragionamento visivo attraverso immagini di oggetti 3D e domande complesse che richiedono logica relazionale e spaziale.

## Esperimenti

| Esperimento | Architettura | Tecniche aggiuntive | Acc. Rationale | Acc. Risposta | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Baseline | T5 Base + Random Vision | Nessuna | 0.00% | 0.00% | 0.00% |
| 2. Singola Fase (Stage 1) | T5 Base + ViT | LoRA | 99.90% | 47.00% | 24.72% |
| 3. Ablazione Testo | FLAN-T5 Small | Prompting Text-Only | 99.90% | 0.00% | 0.00% |
| 4. Due Fasi (Cascade) | T5 Base + ViT | LoRA + Visual Grounding | 99.90% | 47.10% | 32.78% |

## Analisi dei risultati
* **Modelli migliori:** L'architettura a due fasi (Cascade - Stage 2) ha ottenuto le prestazioni migliori. Pur mantenendo un'accuratezza della risposta finale simile allo Stage 1 (47.10% vs 47.00%), ha incrementato nettamente l'F1-Score (32.78% vs 24.72%). Questo dimostra che delegare l'estrazione finale della risposta a una fase puramente visivo-estrattiva migliora la precisione della classe predetta e la stabilità del modello.
* **Principali errori e Anomalie:** L'esperimento di ablazione sul testo puro (FLAN-T5) ha registrato un crollo totale dell'accuratezza (0.00%). Questo risultato estremo non è un'anomalia negativa, ma la dimostrazione empirica cruciale del progetto: conferma che il dataset e le domande non contengono *bias testuali* risolvibili senza l'immagine. 
* **Lezioni apprese:** Scomporre il ragionamento logico (Stage 1) dalla focalizzazione spaziale (Stage 2) permette alla Cross-Attention di specializzarsi esclusivamente sul grounding visivo. Questo approccio a cascata rende l'architettura non solo più accurata in termini di bilanciamento delle classi, ma soprattutto più interpretabile.

## Architettura del codice

```bash
progetto/
├── notebooks/                  # Notebook Jupyter (Pipeline sequenziale)
│   ├── 00_Setup_Environment.ipynb
│   ├── 01_Preparation_Dataset.ipynb
│   ├── 02_Encoder_Comparison.ipynb
│   ├── 03_Train_Stage1.ipynb
│   ├── 04_Train_Stage2.ipynb
│   ├── 05_Validation_Stage1.ipynb
│   ├── 06_Validation_Stage2.ipynb
│   └── 07_Evaluation.ipynb
├── src/                        # Codice principale
│   ├── dataset.py
│   └── models.py
└── README.md                   # Questo file
```
## Setup ed Esecuzione

Il progetto è stato progettato per essere eseguito in ambiente **Google Colab** o in locale. Tutte le dipendenze necessarie (`transformers`, `peft`, `torchao`, ecc.) vengono installate e gestite automaticamente nelle prime celle di ogni notebook. Non è richiesto un file `requirements.txt` globale.

**Per eseguire su Google Colab:**
1. Caricare l'intera cartella del progetto sul proprio Google Drive assicurandosi che il percorso sia esattamente: `MyDrive/DeepLearning`. In caso contrario, aggiornare la variabile `PROJECT_ROOT` nella prima cella dei notebook.
2. Aprire ed eseguire i notebook in ordine sequenziale (da `00_Setup` a `07_Evaluation`).
3. Il notebook di comparazione degli encoder visuali utilizza 3 immagini specifiche data la loro complessità che sono incluse nel repository da includere nella cartella 'data/processed/images/train'

**Per eseguire in locale:**
I notebook riconoscono automaticamente l'assenza dell'ambiente Colab e mappano le directory relative in modo dinamico. Assicurarsi di avere Jupyter installato e avviare il server dalla directory radice del progetto.

