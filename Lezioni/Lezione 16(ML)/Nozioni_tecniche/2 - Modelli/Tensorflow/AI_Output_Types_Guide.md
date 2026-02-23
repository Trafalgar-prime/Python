# Tipologie di Output nei Modelli di AI / Machine Learning

## Tabella riassuntiva

  -----------------------------------------------------------------------------------------------------
  \#   Tipo di output    Cosa rappresenta   Forma           Esempi pratici   Modelli       Metriche
                                            dell'output                      tipici        comuni
  ---- ----------------- ------------------ --------------- ---------------- ------------- ------------
  1    Regressione       Valore continuo    Numero reale    Prezzo casa,     Regressione   MAE, MSE,
                                                            temperatura      lineare, NN   RMSE

  2    Classificazione   2 classi           0/1             Spam detection   Logistic,SVM  Accuracy,
       binaria                                                                             Precision

  3    Classificazione   1 classe su N      Label /         Image            Softmax NN    Accuracy, F1   (un vettore con tutte le classi, ma la sola classe corretta ha 1 mentre tutte le altre 0) ONE-HOT
       multiclasse                          probabilità     classification                                 si applica sia al multiclasse che al multilabel, la differenza è la funzione di attivazione = softmax
                                                                                                           e nella loss che è una BinaryCrossentrpy
  4    Classificazione   Più classi         Vettore binario Tag articoli     NN sigmoid    Hamming loss   
       multilabel                                                                          

  5    Output            Incertezza         Distribuzione   Risk prediction  Bayesian      Log-loss
       probabilistico                                                        models        

  6    Ranking           Ordine elementi    Lista ordinata  Search engines   RankNet       NDCG

  7    Similarità        Somiglianza        Score           Face recognition Siamese NN    Cosine

  8    Clustering        Gruppi             Cluster ID      Segmentazione    K-Means       Silhouette

  9    Output            Oggetti complessi  JSON / tuple    Object detection CNN / YOLO    mAP
       strutturato                                                                         

  10   Output            Sequenze           Lista / stringa Traduzione       Transformer   BLEU
       sequenziale                                                                         

  11   Output            Azione             Azione / policy Robotica         RL            Reward
       decisionale                                                                         

  12   Output generativo Nuovi dati         Testo /         LLM, GAN         GAN, VAE      FID
                                            immagini                                       

  13   Embedding         Rappresentazione   Vettore         Semantic search  BERT          Cosine
                         latente                                                           
  -----------------------------------------------------------------------------------------------------

## Come usare questa guida

1.  Definisci l'output
2.  Seleziona il modello
3.  Usa loss e metriche corrette

| Tipo problema         | Output layer | Attivazione      | Loss                          |
| --------------------- | ------------ | ---------------- | ----------------------------- |
| Binaria               | Dense(1)     | sigmoid          | BinaryCrossentropy            |
| Multiclasse (interi)  | Dense(K)     | softmax o logits | SparseCategoricalCrossentropy |
| Multiclasse (one-hot) | Dense(K)     | softmax o logits | CategoricalCrossentropy       |
| Multi-label           | Dense(K)     | sigmoid          | BinaryCrossentropy            |

softmax e sigmoid come tutte le altre funzioni che limitano i dati in un certo intervallo non hanno bisogno di logits(nella loss), questo perchè i dati sono già traformati in probabilità
invece nel caso in cui non metto la funzione di attivazione o la metto lineare ho bisogno del logits per fare una trasformazione dei dati in una probabilità: questo perchè la loss per
funzione ha necessariamente bisogno che i dati siano sotto forma di propabilità

poi possiamo fare un altro discorso secondo cui sono entrambi metodi validi, ma utilizzare il logits rende tutti i modelli più stabili

| Tipo problema        | Output    | Attivazione finale |
| -------------------- | --------- | ------------------ |
| Binaria              | 1 neurone | Sigmoid            |
| Multiclasse          | K neuroni | Softmax            |
| Multi-label          | K neuroni | Sigmoid            |
| Regressione libera   | 1 neurone | Lineare            |
| Regressione positiva | 1 neurone | Softplus           |
| Output tra -1 e 1    | 1 neurone | Tanh               |
| Policy RL            | K neuroni | Softmax            |


| Tipo problema             | Output             | Attivazione finale | Loss principali                                                  |
| ------------------------- | ------------------ | ------------------ | ---------------------------------------------------------------- |
| **Binaria**               | 1 neurone          | Sigmoid            | `BinaryCrossentropy`                                             |
|                           | 1 neurone (logits) | Nessuna            | `BinaryCrossentropy(from_logits=True)`                           |
| **Multiclasse (interi)**  | K neuroni          | Softmax            | `SparseCategoricalCrossentropy`                                  |
|                           | K neuroni (logits) | Nessuna            | `SparseCategoricalCrossentropy(from_logits=True)`                |
| **Multiclasse (one-hot)** | K neuroni          | Softmax            | `CategoricalCrossentropy`                                        |
|                           | K neuroni (logits) | Nessuna            | `CategoricalCrossentropy(from_logits=True)`                      |
| **Multi-label**           | K neuroni          | Sigmoid            | `BinaryCrossentropy`                                             |
|                           | K neuroni (logits) | Nessuna            | `BinaryCrossentropy(from_logits=True)`                           |
| **Regressione libera**    | 1 neurone          | Lineare            | `MeanSquaredError (MSE)`<br>`MeanAbsoluteError (MAE)`<br>`Huber` |
| **Regressione positiva**  | 1 neurone          | Softplus           | `MSE`, `MAE`, `Huber`                                            |
| **Output tra -1 e 1**     | 1 neurone          | Tanh               | `MSE`, `MAE`                                                     |
| **Poisson regression**    | 1 neurone          | Softplus / exp     | `Poisson`                                                        |
| **Quantile regression**   | 1 neurone          | Lineare            | `Quantile loss` (custom)                                         |
| **Policy RL (discrete)**  | K neuroni          | Softmax            | `CategoricalCrossentropy` (policy gradient)                      |
| **Policy RL (continua)**  | N neuroni          | Lineare / Tanh     | loss custom (es. log-prob gaussiane)                             |
