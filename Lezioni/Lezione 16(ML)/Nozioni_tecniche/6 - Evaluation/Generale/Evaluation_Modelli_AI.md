# BLOCCO 1 --- Valutazione Universale dei Modelli di AI

(Indipendente da sklearn, PyTorch, TensorFlow)

------------------------------------------------------------------------

## 0) Checklist mentale: cosa significa "modello addestrato bene"

Un modello è addestrato bene se soddisfa **tutte** queste condizioni:

1.  Performance buona su dati **mai visti** (test / scenario reale)
2.  Generalizzazione corretta (gap train--validation contenuto)
3.  Stabilità (risultati simili su split / fold diversi)
4.  Affidabilità (errori coerenti con il costo reale FP vs FN)
5.  Robustezza (non crolla con rumore o data shift ragionevoli)
6.  Correttezza metodologica (niente data leakage, split corretti)

Accuracy alta **da sola non basta**.

------------------------------------------------------------------------

## 1) Train / Validation / Test split

### Perché serve

-   **Train**: apprendimento dei parametri
-   **Validation**: tuning iperparametri, soglia, early stopping
-   **Test**: valutazione finale (una sola volta)

### Errori critici

-   Data leakage diretto o indiretto
-   Preprocessing fittato su tutto il dataset
-   Duplicati o stesso utente in train e test
-   Split temporale errato nelle time series

📌 Tutto ciò che "impara dai dati" va fittato **solo su train**.

------------------------------------------------------------------------

## 2) Loss function: interpretazione corretta

La **loss** è ciò che il modello ottimizza.\
Le **metriche** sono ciò che conta nel mondo reale.

### Curve di loss

-   Train ↓ e Val ↓ → apprendimento corretto
-   Train ↓ e Val ↑ → overfitting
-   Train alta e piatta → underfitting
-   Oscillazioni → learning rate troppo alto / training instabile

### Loss comuni

#### Regressione

-   MSE
-   MAE
-   Huber

#### Classificazione

-   Binary Cross-Entropy
-   Categorical Cross-Entropy
-   Focal Loss

------------------------------------------------------------------------

## 3) Accuracy: quando usarla e quando no

Accuracy = corrette / totale.

### Va bene se:

-   classi bilanciate
-   costi simili tra FP e FN

### È fuorviante se:

-   dataset sbilanciato
-   classe positiva rara

------------------------------------------------------------------------

## 4) Confusion Matrix

Mostra **come** il modello sbaglia.

Binaria: - TP, FP, TN, FN

Multiclasse: - matrice NxN

Serve per: - analisi degli errori - valutazione per classe - decisioni
di soglia

------------------------------------------------------------------------

## 5) Precision, Recall, F1

-   **Precision**: affidabilità delle predizioni positive
-   **Recall**: capacità di trovare i veri positivi
-   **F1**: compromesso precision/recall

Varianti: - Fβ - Macro / Micro / Weighted averaging
| Tipo problema | average da usare         |
| ------------- | ------------------------ |
| Binaria       | niente (default)         |
| Multiclasse   | macro / weighted / micro |
| Multi-label   | macro / micro            |


------------------------------------------------------------------------

## 6) Metriche threshold-free

### ROC-AUC

-   misura separabilità globale
-   buona con classi bilanciate

### PR-AUC

-   più informativa con classi sbilanciate

------------------------------------------------------------------------

## 7) Scelta della soglia decisionale

La soglia standard (0.5) quasi mai è ottimale.

Strategie: - massimizzare F1/Fβ su validation - vincoli di recall o
precision - minimizzazione costo atteso

------------------------------------------------------------------------

## 8) Regressione: metriche e diagnosi

-   MAE: errore medio interpretabile
-   RMSE: penalizza errori grandi
-   R²: varianza spiegata (attenzione)

Analisi: - residual plot - errore per sottogruppi

------------------------------------------------------------------------

## 9) Cross-Validation

Serve a valutare **stabilità e affidabilità**.

Tipi: - K-Fold - Stratified K-Fold - GroupKFold - TimeSeriesSplit

Alta varianza → modello instabile o dati insufficienti.

------------------------------------------------------------------------

## 10) Learning Curves

Diagnosi: - Underfitting: train e val scarsi - Overfitting: train
ottimo, val scarso - Migliora con più dati → dataset insufficiente

------------------------------------------------------------------------

## 11) Calibrazione delle probabilità

Un modello può separare bene ma essere mal calibrato.

Strumenti: - Reliability diagram - Brier score

Metodi: - Platt scaling - Isotonic regression

------------------------------------------------------------------------

## 12) Robustezza e validazione realistica

-   Stress test (rumore, missing)
-   Data shift
-   Performance per sottogruppi
-   Out-of-distribution behavior

------------------------------------------------------------------------

## 13) Metriche specifiche per output

### Ranking

-   NDCG, MAP, MRR, Recall@K

### Clustering

-   Silhouette
-   Davies--Bouldin
-   Calinski--Harabasz
-   ARI, NMI (se etichette note)

### NLP generativo

-   Perplexity
-   BLEU
-   ROUGE

### Computer Vision

-   Detection: mAP, IoU
-   Segmentation: IoU, Dice

### Reinforcement Learning

-   Reward medio
-   Varianza reward
-   Stabilità
-   Generalizzazione

------------------------------------------------------------------------

## 14) Segnali pratici di buon addestramento

-   Test simile a validation
-   Gap train/val contenuto
-   Confusion matrix sensata
-   Metriche adeguate al contesto
-   CV stabile
-   Probabilità calibrate
-   Robustezza a shift

------------------------------------------------------------------------

## 15) Errori comuni

-   Accuracy alta ma modello inutile
-   Overfitting mascherato
-   Data leakage
-   Soglia errata
-   Metriche sbagliate
-   Split temporale errato

------------------------------------------------------------------------

## Conclusione

Un buon modello: - generalizza - è stabile - sbaglia nel modo giusto -
usa metriche corrette - è valutato con metodo

Allenare è facile. **Valutare bene è ciò che conta davvero.**
