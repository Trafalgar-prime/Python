# Metriche di Valutazione nei Modelli di Machine Learning (PyTorch)

## Introduzione

Le metriche si dividono principalmente in due famiglie a seconda del
problema:

1)  Classificazione → previsione di categorie
2)  Regressione → previsione di valori continui

Una corretta valutazione è fondamentale per capire se il modello
generalizza oppure soffre di overfitting.

------------------------------------------------------------------------

# 1. Metriche per la Classificazione

## Precision (Precisione)

Definizione: Percentuale di predizioni positive corrette rispetto a
tutte le predizioni positive fatte dal modello.

Formula: Precision = TP / (TP + FP)

Dove: TP = True Positives FP = False Positives

Interpretazione: Misura la qualità delle predizioni positive.

------------------------------------------------------------------------

## Recall (Sensibilità)

Definizione: Percentuale di veri positivi individuati rispetto a tutti i
positivi reali.

Formula: Recall = TP / (TP + FN)

Dove: FN = False Negatives

Interpretazione: Misura la capacità del modello di trovare tutti i casi
positivi.

------------------------------------------------------------------------

## F1-Score

Definizione: Media armonica tra Precision e Recall.

Formula: F1 = 2 \* (Precision \* Recall) / (Precision + Recall)

Interpretazione: Utile quando serve equilibrio tra precisione e
richiamo, specialmente in dataset sbilanciati.

------------------------------------------------------------------------

## Confusion Matrix

Matrice che mostra:

                Predetto Positivo   Predetto Negativo

Reale Positivo TP FN Reale Negativo FP TN

Permette di capire dove il modello commette errori.

------------------------------------------------------------------------

# 2. Metriche per la Regressione

## MSE (Mean Squared Error)

Formula: MSE = (1/n) \* Σ (y_pred - y_true)\^2

Caratteristiche: - Penalizza fortemente errori grandi - Sensibile agli
outlier

------------------------------------------------------------------------

## MAE (Mean Absolute Error)

Formula: MAE = (1/n) \* Σ \|y_pred - y_true\|

Caratteristiche: - Stessa unità di misura del dato - Più robusto
rispetto agli outlier rispetto a MSE

------------------------------------------------------------------------

## R² (Coefficiente di Determinazione)

Formula: R² = 1 - (SS_res / SS_tot)

Dove: SS_res = somma dei quadrati residui SS_tot = varianza totale

Intervallo: 0 → nessuna spiegazione della varianza 1 → modello perfetto

------------------------------------------------------------------------

# 3. Implementazione in PyTorch

Le metriche possono essere:

-   Implementate manualmente
-   Calcolate con torchmetrics
-   Calcolate con scikit-learn

------------------------------------------------------------------------

## Esempio con scikit-learn (Classificazione)

``` python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(targets_reali, predizioni_modello)
report = classification_report(targets_reali, predizioni_modello)

print("Matrice di Confusione:\n", cm)
print("\nReport di Classificazione:\n", report)
```

------------------------------------------------------------------------

## Esempio manuale Accuracy in PyTorch

``` python
corrette = (predizioni == target).sum().item()
accuracy = corrette / len(target)
```

------------------------------------------------------------------------

# 4. Differenza concettuale importante

Loss ≠ Metrica

La Loss: - serve per ottimizzare il modello - è differenziabile

La Metrica: - serve per valutare la performance - non deve essere
necessariamente differenziabile

------------------------------------------------------------------------

# Conclusione

Per una valutazione completa:

Classificazione: - Accuracy - Precision - Recall - F1-score - Confusion
Matrix

Regressione: - MSE - MAE - R²

La scelta della metrica deve sempre essere coerente con l'obiettivo del
problema e la distribuzione dei dati.
