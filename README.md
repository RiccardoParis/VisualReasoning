Multimodal Cascade Reasoning per Visual Question Answering

Architettura multimodale a due fasi basata su T5 e Cross-Attention per migliorare il Visual Grounding nel ragionamento logico-visivo.

Obiettivo del progetto:
Il progetto affronta il problema delle allucinazioni testuali e della scarsa interpretabilità nei modelli di Visual Question Answering (VQA).
L'obiettivo è dimostrare che scomporre il task in due fasi distinte — la generazione testuale del ragionamento logico (Rationale) e la successiva estrazione visiva della risposta (Visual Grounding) — migliora l'affidabilità del sistema.
Si intende dimostrare, tramite specifici test di ablazione, che il modello finale non deduce la risposta basandosi esclusivamente sul pattern testuale, ma impara a "mettere a fuoco" i pixel esatti dell'immagine necessari a validare il ragionamento per produrre la risposta corretta.

Background e tecniche utilizzate
Tecniche di Deep Learning:
Il sistema utilizza Large Language Models (LLM) basati sull'architettura Transformer (T5) accoppiati a Vision Encoders.
L'integrazione multimodale avviene tramite layer di Cross-Attention personalizzati.
Per ottimizzare l'addestramento testuale senza incorrere in catastrophic forgetting, viene impiegata la tecnica di Parameter-Efficient Fine-Tuning (PEFT) tramite adattatori LoRA. 

Strumenti: L'intera pipeline è sviluppata in PyTorch, sfruttando l'ecosistema Hugging Face (librerie transformers e peft) per la gestione dei modelli pre-addestrati e la tokenizzazione.
Per la valutazione metrica è stato utilizzato Scikit-learn.

Dataset: 
Il progetto è addestrato e valutato sul dataset CLEVR (Compositional Language and Elementary Visual Reasoning).
Questo dataset è specificamente progettato per testare le capacità di ragionamento visivo attraverso immagini di oggetti 3D (cilindri, sfere, cubi) e domande complesse che richiedono conteggio, confronto e logica relazionale o spaziale.
