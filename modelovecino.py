"""
Modelo KNN (vecinos más cercanos) para regresión.

PARA RELLENAR: solo la sección CONFIGURACIÓN de abajo.
El resto del script no hace falta tocarlo.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

# =============================================================================
# CONFIGURACIÓN  <-- RELLENAR AQUÍ
# =============================================================================

RUTA_EXCEL = "datos.xlsx"   # ruta al Excel (o .csv)
HOJA = 0                    # nombre o índice de la hoja del Excel

TARGET = ""                 # columna a predecir, p. ej. "precio"

# Columnas numéricas (se escalan con StandardScaler)
NUMERIC_FEATURES = [
    # "superficie", "habitaciones", ...
]

# Columnas categóricas SIN orden (One-Hot)
NOMINAL_FEATURES = [
    # "tipo_vivienda", ...
]

# Columnas categóricas CON orden: {columna: [categorías de menor a mayor]}
ORDINAL_FEATURES = {
    # "energy": ["G", "F", "E", "D", "C", "B", "A"],
    # "condition": ["mala", "regular", "buena", "nueva"],
}

# Columnas categóricas con MUCHAS categorías (p. ej. barrio, código postal).
# Las categorías poco frecuentes se agrupan (min_frequency se ajusta en el grid).
HIGH_CARDINALITY_FEATURES = [
    # "barrio", ...
]

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def cargar_datos(ruta, hoja=0):
    if str(ruta).lower().endswith(".csv"):
        return pd.read_csv(ruta)
    return pd.read_excel(ruta, sheet_name=hoja)


def comprobar_columnas(df):
    """Avisa con un mensaje claro si falta algo en la configuración."""
    if not TARGET:
        raise ValueError("Falta rellenar TARGET en la configuración.")
    todas = ([TARGET] + NUMERIC_FEATURES + NOMINAL_FEATURES
             + list(ORDINAL_FEATURES) + HIGH_CARDINALITY_FEATURES)
    if len(todas) == 1:
        raise ValueError("No hay ninguna columna de entrada en la configuración.")
    faltan = [c for c in todas if c not in df.columns]
    if faltan:
        raise ValueError(
            f"Estas columnas no están en el Excel: {faltan}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )


df = cargar_datos(RUTA_EXCEL, HOJA)
comprobar_columnas(df)

feature_cols = (NUMERIC_FEATURES + NOMINAL_FEATURES
                + list(ORDINAL_FEATURES) + HIGH_CARDINALITY_FEATURES)

df = df.dropna(subset=[TARGET])     # sin target no se puede entrenar
X = df[feature_cols]
y = df[TARGET]

# =============================================================================
# 2. PARTICIÓN train / validation / test  (60 / 20 / 20)
# =============================================================================

# Primero se separa el test final.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
)

# El resto se divide en train y validation (0.25 * 0.8 = 0.2 del total).
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE,
)

print(pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(X_train), len(X_val), len(X_test)],
    "percentage": [len(X_train) / len(X), len(X_val) / len(X), len(X_test) / len(X)],
}))

# =============================================================================
# 3. PREPROCESADO PARA KNN
# =============================================================================

transformers = []

if NUMERIC_FEATURES:
    knn_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers.append(("numeric", knn_numeric, NUMERIC_FEATURES))

if NOMINAL_FEATURES:
    knn_nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers.append(("nominal", knn_nominal, NOMINAL_FEATURES))

if ORDINAL_FEATURES:
    knn_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=list(ORDINAL_FEATURES.values()),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scaler", StandardScaler()),
    ])
    transformers.append(("ordinal", knn_ordinal, list(ORDINAL_FEATURES)))

if HIGH_CARDINALITY_FEATURES:
    knn_high_card = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
        )),
    ])
    transformers.append(("high_card", knn_high_card, HIGH_CARDINALITY_FEATURES))

knn_preprocessor = ColumnTransformer(transformers)

knn_pipeline = Pipeline([
    ("preprocessing", knn_preprocessor),
    ("model", KNeighborsRegressor()),
])

# =============================================================================
# 4. BÚSQUEDA DE HIPERPARÁMETROS (GridSearchCV 5-fold, RMSE)
# =============================================================================

knn_param_grid = {
    "model__n_neighbors": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    "model__weights": ["uniform", "distance"],
    "model__p": [1],    # 1 = Manhattan, 2 = Euclídea
}
# min_frequency solo tiene sentido si hay columnas de alta cardinalidad.
if HIGH_CARDINALITY_FEATURES:
    knn_param_grid["preprocessing__high_card__encoder__min_frequency"] = [2, 3, 4, 5, 6]

# n_neighbors no puede superar las filas de cada fold de entrenamiento.
max_k = int(len(X_train) * 4 / 5)
knn_param_grid["model__n_neighbors"] = [
    k for k in knn_param_grid["model__n_neighbors"] if k <= max_k
]

knn_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    return_train_score=True,
)

knn_search.fit(X_train, y_train)


def search_results_table(search, top=10):
    """Tabla resumen del GridSearchCV, ordenada por RMSE de validación cruzada."""
    res = pd.DataFrame(search.cv_results_)
    tabla = pd.DataFrame({
        "params": res["params"],
        "cv_rmse": -res["mean_test_score"],
        "cv_rmse_std": res["std_test_score"],
        "train_rmse": -res["mean_train_score"],
        "rank": res["rank_test_score"],
    }).sort_values("rank")
    return tabla.head(top).reset_index(drop=True)


print("\nMejores combinaciones (CV):")
print(search_results_table(knn_search).to_string())
print("\nMejores parámetros:", knn_search.best_params_)

# =============================================================================
# 5. EVALUACIÓN
# =============================================================================

def metricas(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# Validation: el mejor modelo (entrenado en train) sobre datos no vistos.
best_knn = knn_search.best_estimator_
print("\nValidation:", metricas(y_val, best_knn.predict(X_val)))

# Test final: se reentrena con train + validation y se evalúa una sola vez.
final_knn = clone(best_knn)
final_knn.fit(X_temp, y_temp)
print("Test:", metricas(y_test, final_knn.predict(X_test)))
