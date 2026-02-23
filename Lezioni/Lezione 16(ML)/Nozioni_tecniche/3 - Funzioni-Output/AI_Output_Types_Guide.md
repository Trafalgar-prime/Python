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
       multiclasse                          probabilità     classification                 

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


| Problema reale | Architettura corretta |
| -------------- | --------------------- |
| 10 classi      | Dense(10) + softmax   |
| 2 classi       | Dense(1) + sigmoid    |
| Multi-label    | Dense(K) + sigmoid    |
