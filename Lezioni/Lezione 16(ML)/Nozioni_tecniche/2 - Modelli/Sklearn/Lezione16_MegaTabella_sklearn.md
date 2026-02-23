# Lezione 16 — Mega tabella (Parte 1/3): scikit-learn

> Obiettivo: darti una “mappa” dei **modelli più importanti** in scikit-learn e dei loro **parametri/opzioni**, con valori ammessi e vincoli.

---

## I 10 modelli scikit-learn più importanti (pratici, usatissimi)
- Regressione
1. `LinearRegression` (regressione lineare OLS)
2. `Ridge` (regressione lineare con regolarizzazione L2)
3. `Lasso` (regressione lineare con regolarizzazione L1)
- Classificazione
4. `LogisticRegression` (classificazione lineare/logistica)
5. `SVC` (Support Vector Classification)
6. `RandomForestClassifier` (ensemble di alberi: random forest)
7. `GradientBoostingClassifier` (boosting “classico” su alberi)  
   *(nota: per dataset grandi spesso si preferisce `HistGradientBoostingClassifier`)*
8. `KNeighborsClassifier` (k-NN)
9. `MLPClassifier` (rete neurale “classica” feed-forward in sklearn)
- Unsupervises
10. `KMeans` (clustering)

> Nota: `PCA` è super importante, ma è un **trasformatore** (riduzione dimensionale), non un “modello predittivo” in senso stretto. Io lo uso in pratica come “#11 bonus” perché ti serve tantissimo.

---

# Mega tabella — Modelli (Estimator) e opzioni principali

**Legenda valori**
- `bool`: `True/False`
- `int`: intero
- `float`: reale
- `None`: valore nullo
- `str ∈ {...}`: stringa fra scelte possibili
- `callable`: funzione passata dall’utente
- `array-like`: array/lista/np.ndarray
- vincoli: es. `>= 0`, `> 0`, intervalli, ecc.

---

## 1) LinearRegression (OLS)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.linear_model.LinearRegression` |
| Task | Regressione |
| Idea | Trova i coefficienti `w` che minimizzano l’errore quadratico medio (OLS) |
| Parametri principali | `fit_intercept: bool` (default `True`) • `copy_X: bool` • `tol: float` • `n_jobs: int|None` • `positive: bool` |
| Cosa cambia davvero | `fit_intercept=False` se i dati sono già centrati • `positive=True` impone coefficienti non negativi |

---

## 2) Ridge (L2)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.linear_model.Ridge` |
| Task | Regressione |
| Idea | OLS + penalità L2 (riduce overfitting, stabilizza in multicollinearità) |
| Parametri principali | `alpha: float >= 0` (default `1.0`) oppure array per multi-target • `fit_intercept: bool` • `copy_X: bool` • `max_iter: int|None` • `tol: float` • `solver: str` (es. `'auto'`, `'svd'`, `'cholesky'`, `'lsqr'`, `'sparse_cg'`, `'sag'`, `'saga'`, `'lbfgs'` a seconda dei casi) • `positive: bool` • `random_state: int|None` |
| Note | `alpha=0` equivale concettualmente a OLS, ma in pratica è meglio usare `LinearRegression` |

---

## 3) Lasso (L1)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.linear_model.Lasso` |
| Task | Regressione |
| Idea | Penalità L1 → tende a fare “feature selection” (alcuni pesi diventano 0) |
| Parametri principali | `alpha: float >= 0` • `fit_intercept: bool` • `precompute: bool|array-like` • `copy_X: bool` • `max_iter: int` • `tol: float` • `warm_start: bool` • `positive: bool` • `random_state: int|None` • `selection: str ∈ {'cyclic','random'}` |
| Note | Se metti `warm_start=True`, una nuova `fit()` riparte dai pesi precedenti |

---

## 4) LogisticRegression (classificazione lineare)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.linear_model.LogisticRegression` |
| Task | Classificazione (binaria o multiclasse) |
| Idea | Modello lineare che produce probabilità (logistica) con regolarizzazione |
| Parametri principali (molto importanti) | `penalty: str ∈ {'l1','l2','elasticnet'} oppure None` (in base al solver) • `C: float > 0` (forza regolarizzazione: **più piccolo = più regolarizzazione**) • `solver: str ∈ {'lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'}` • `l1_ratio: float in [0,1]` (solo con `penalty='elasticnet'`) • `max_iter: int` • `class_weight: None | 'balanced' | dict` |
| Altri parametri comuni | `fit_intercept: bool` • `tol: float` • `multi_class` (dipende da versione/solver) • `n_jobs: int|None` (solo per alcuni solver) • `random_state: int|None` (soprattutto per `'saga'`) |
| Note pratiche | Se vuoi L1 o ElasticNet in modo robusto, spesso usi `solver='saga'` |

---

## 5) SVC (Support Vector Classification)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.svm.SVC` |
| Task | Classificazione |
| Idea | Massimizza il margine; con kernel non lineari gestisce separazioni complesse |
| Parametri principali | `C: float > 0` (regolarizzazione) • `kernel: str ∈ {'linear','poly','rbf','sigmoid','precomputed'} oppure callable` • `gamma: str ∈ {'scale','auto'} oppure float > 0` (per kernel RBF/poly/sigmoid) • `degree: int` (solo per `'poly'`) • `coef0: float` (poly/sigmoid) |
| Opzioni “di training” | `probability: bool` (se `True` abilita probabilità ma rallenta) • `class_weight: None|'balanced'|dict` • `tol: float` • `max_iter: int` • `shrinking: bool` • `cache_size: float` |
| Note pratiche | Su dataset grandi SVC con kernel può diventare lento; spesso si usa `LinearSVC` o altri modelli |

---

## 6) RandomForestClassifier

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.ensemble.RandomForestClassifier` |
| Task | Classificazione |
| Idea | Molti alberi su sotto-campioni; media/voto riduce varianza |
| Parametri principali | `n_estimators: int` (numero alberi) • `criterion: str ∈ {'gini','entropy','log_loss'}` • `max_depth: int|None` • `min_samples_split: int|float` • `min_samples_leaf: int|float` |
| Parametri di “randomness” | `max_features: str ∈ {'sqrt','log2'} | int | float | None` • `bootstrap: bool` • `max_samples: int|float|None` (solo se `bootstrap=True`) • `random_state: int|None` |
| Altre opzioni utili | `class_weight: None | 'balanced' | 'balanced_subsample' | dict` • `n_jobs: int|None` • `oob_score: bool|callable` (solo se `bootstrap=True`) • `warm_start: bool` • `ccp_alpha: float >= 0` (potatura) |
| Note pratiche | RF spesso funziona bene “out of the box”; controllare overfitting con `max_depth`, `min_samples_*` |

---

## 7) GradientBoostingClassifier (boosting “classico”)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.ensemble.GradientBoostingClassifier` |
| Task | Classificazione |
| Idea | Somma alberi in modo sequenziale: ogni nuovo albero corregge errori precedenti |
| Parametri principali | `loss: str ∈ {'log_loss','exponential'}` • `learning_rate: float >= 0` • `n_estimators: int >= 1` • `max_depth: int` (profondità alberi base) |
| Regularizzazione | `subsample: float in (0,1]` (stochastic gradient boosting) • `min_samples_split` • `min_samples_leaf` • `max_features` • `ccp_alpha` |
| Early stopping | `validation_fraction: float` • `n_iter_no_change: int|None` • `tol: float` |
| Note pratiche | Se `n_samples` è grande (>= ~10k) spesso conviene `HistGradientBoostingClassifier` (più veloce) |

---

## 8) KNeighborsClassifier (k-NN)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.neighbors.KNeighborsClassifier` |
| Task | Classificazione |
| Idea | Classifica guardando i k vicini più prossimi nello spazio delle feature |
| Parametri principali | `n_neighbors: int` (k) • `weights: str ∈ {'uniform','distance'} o callable` • `metric: str` (default `'minkowski'`) • `p: int` (con Minkowski: `p=2` euclidea, `p=1` manhattan) |
| Opzioni implementative | `algorithm: str ∈ {'auto','ball_tree','kd_tree','brute'}` • `leaf_size: int` • `n_jobs: int|None` |
| Note pratiche | k-NN è sensibile alla scala delle feature → spesso serve `StandardScaler` |

---

## 9) MLPClassifier (rete neurale in sklearn)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.neural_network.MLPClassifier` |
| Task | Classificazione |
| Idea | Rete feed-forward (MLP). Qui compaiono parametri tipo **batch**, **learning rate**, **ottimizzatore** (solver). |
| Architettura | `hidden_layer_sizes: tuple` es. `(32, 16)` • `activation: str ∈ {'identity','logistic','tanh','relu'}` |
| Ottimizzazione | `solver: str ∈ {'lbfgs','sgd','adam'}` • `learning_rate: str ∈ {'constant','invscaling','adaptive'}` (per SGD) • `learning_rate_init: float` • `batch_size: int|'auto'` |
| Regolarizzazione | `alpha: float` (L2) |
| Stabilità/convergenza | `max_iter: int` • `tol: float` • `shuffle: bool` • `early_stopping: bool` • `validation_fraction: float` • `n_iter_no_change: int` |
| Momentum (solo SGD) | `momentum: float` • `nesterovs_momentum: bool` |
| Adam params (se solver='adam') | `beta_1: float` • `beta_2: float` • `epsilon: float` |
| Note pratiche | In DL “serio” userai quasi sempre PyTorch/TensorFlow, ma qui capisci bene i concetti |

---

## 10) KMeans (clustering)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.cluster.KMeans` |
| Task | Clustering (non supervisionato) |
| Idea | Trova `k` centroidi minimizzando l’inerzia (distanza intra-cluster) |
| Parametri principali | `n_clusters: int` • `init: str ∈ {'k-means++','random'} o array-like o callable` • `n_init: int|'auto'` • `max_iter: int` • `tol: float` |
| Opzioni utili | `algorithm: str ∈ {'lloyd','elkan'}` • `random_state: int|None` • `verbose: int` • `copy_x: bool` |
| Note pratiche | Spesso si fa `StandardScaler` prima e talvolta `PCA` per ridurre rumore/dimensione |

---

# Bonus fondamentale: PCA (trasformatore)

| Campo | Dettagli |
|---|---|
| Classe | `sklearn.decomposition.PCA` |
| Task | Riduzione dimensionale (trasformazione) |
| Idea | Proietta i dati su componenti principali (SVD) |
| Parametri principali | `n_components: int | float | 'mle' | None` • `whiten: bool` • `svd_solver: str ∈ {'auto','full','randomized','arpack','covariance_eigh'}` • `tol: float` • `random_state: int|None` |
| Note pratiche | PCA **centra** ma non scala: spesso fai `StandardScaler` prima |

---

# Opzioni “trasversali” super importanti in scikit-learn (ti servono sempre)

## 1) `train_test_split`
- `test_size: float|int` (es. `0.2`)
- `train_size: float|int|None`
- `random_state: int|None`
- `shuffle: bool`
- `stratify: array-like|None` (fondamentale in classificazione con classi sbilanciate)

## 2) `StandardScaler`
- `with_mean: bool` (attenzione: sparse → spesso `False`)
- `with_std: bool`

## 3) `Pipeline`
- mette in catena trasformazioni + modello, evita leakage
- es: `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])`

## 4) `GridSearchCV` / `RandomizedSearchCV`
- `estimator`
- `param_grid` / `param_distributions`
- `cv: int | splitter`
- `scoring: str|callable|None`
- `n_jobs: int|None`
- `refit: bool|str`
- `verbose: int`

---

## Perché questa parte (sklearn) è utile se vuoi arrivare a VAE/GAN/Transformer?
- Ti costruisce un istinto su **data leakage**, **scaling**, **split**, **overfitting**, **regolarizzazione**, **tuning**.
- Poi in TensorFlow/PyTorch fai le stesse cose, ma con più controllo (backprop, layer, GPU, ecc.).

