# NGBoost — Resumen del repo de GitHub

Fuente: [stanfordmlgroup/ngboost](https://github.com/stanfordmlgroup/ngboost)

## Qué es NGBoost

NGBoost (*Natural Gradient Boosting*) es una librería de Python que extiende el gradient boosting a **predicción probabilística**: en vez de devolver un único punto, devuelve una distribución de probabilidad completa para cada predicción (por ejemplo, media y varianza de una Normal, o los parámetros de una Poisson, Weibull, etc.).

Lo logra haciendo boosting sobre los parámetros de una distribución elegida, usando **gradientes naturales** (que corrigen la geometría/parametrización de la distribución mediante la métrica de información de Fisher). El diseño es modular: se puede combinar cualquier

- **base learner** (por defecto, árboles de decisión),
- **distribución** (Normal, LogNormal, Exponential, Poisson, Gamma, Beta, Weibull, Cauchy, Laplace, Student-t, Categorical, Multivariate Normal, mixturas, etc.),
- **scoring rule** (LogScore/MLE o CRPScore).

Está construida sobre scikit-learn, así que sigue la API conocida `.fit()` / `.predict()`.

## Instalación

```bash
pip install --upgrade ngboost
# o
conda install -c conda-forge ngboost
```

## API clave y ejemplo mínimo

Clases principales:

- `NGBRegressor` — regresión probabilística.
- `NGBClassifier` — clasificación probabilística.
- `NGBSurvival` — análisis de supervivencia (datos censurados).

Ejemplo del README (California housing):

```python
from ngboost import NGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, Y = fetch_california_housing(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)        # predicción puntual (ej. la media)
Y_dists = ngb.pred_dist(X_test)      # distribución completa predicha (media, std, etc.)
```

## Notebooks y ejemplos disponibles

- **`notebooks/`** (los más fáciles de adaptar a Colab): un notebook por distribución — `example_normal.ipynb`, `example_poisson.ipynb`, `example_weibull.ipynb`, `example_gamma.ipynb`, `example_t.ipynb`, `example_laplace.ipynb`, `example_beta.ipynb`, `example_betabinomial.ipynb`, `example_halfnormal.ipynb`, `example_logitnormal.ipynb`, notebooks de mixturas (`example_mixture_normal.ipynb`, `example_mixture_lognormal.ipynb`), `example_multivariate_normal.ipynb`, y `sympy_normal_demo.ipynb` (deriva simbólicamente el gradiente natural con sympy).
  - **Recomendado para la demo en Colab: `example_normal.ipynb`** — es el más simple y coincide con el flujo básico del README.
- **`examples/`** (scripts `.py`, uno por tipo de tarea): `regression.py`, `classification.py`, `multiclass_classification.py`, `survival.py`, `poisson.py`, `multivariate_normal.py`, `sklearn_cv.py` (integración con cross-validation de sklearn). También hay subcarpetas de investigación: `experiments/`, `model_interpretation/`, `simulations/`, `tuning/`.

## Estructura del repo

- `ngboost/` — código fuente:
  - `ngboost.py` — algoritmo principal.
  - `api.py` — wrappers `NGBRegressor` / `NGBClassifier` / `NGBSurvival`.
  - `scores.py` — reglas de scoring (LogScore/MLE, CRPScore).
  - `learners.py` — base learners por defecto.
  - `distns/` — ~20 módulos de distribuciones.
  - `manifold.py`, `evaluation.py`, `helpers.py`.
- `examples/`, `notebooks/`, `data/`, `figures/`, `results/`, `scripts/`, `docs/`, `tests/`.
- Archivos raíz: `README.md`, `RELEASE_NOTES.md`, `LICENSE`, `setup.cfg`, `pyproject.toml`.

## Licencia y dependencias

- **Licencia:** Apache 2.0.
- **Actividad:** ~1.9k stars, 256 forks, 740 commits en `master` (historial activo).
- **Dependencias principales:** scikit-learn, numpy, scipy, tqdm, lifelines (supervivencia), sympy (derivación simbólica de gradientes), joblib, matplotlib. Soporta Python 3.9–3.15. Tooling de desarrollo con Poetry (pytest, black, pre-commit).
- **Cita / paper original:** Duan, Avati, Ding, Thai, Basu, Ng, Schuler (2019), *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*, arXiv:1910.03225 — coincide con `ngboostpaper.pdf`.

## Tip para la presentación

Usar el snippet del README como demo en vivo: mostrar primero `ngb.predict(X_test)` (predicción puntual, como cualquier GBM) y después `ngb.pred_dist(X_test)` (la distribución completa), para que la diferencia con XGBoost/LightGBM quede clara en código, no solo en teoría.

Después, abrir `notebooks/example_normal.ipynb` en Colab para graficar la banda de incertidumbre (media ± desvío estándar) sobre las predicciones — es la forma más simple de mostrar visualmente la ventaja central de NGBoost: no solo "cuánto", sino "cuán seguro estoy de cuánto".
