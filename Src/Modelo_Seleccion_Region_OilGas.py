import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 12345
BUDGET = 10_000_000  # presupuesto disponible para desarrollar pozos
REVENUE_PER_THOUSAND_BARRELS = 4_500  # ingreso por cada mil barriles producidos
N_WELLS_PER_REGION = 500
N_TOP_WELLS = 200
N_BOOTSTRAP_ITER = 1000


def load_region_data(
    path_region_0: str = "Data/geo_data_0.csv",
    path_region_1: str = "Data/geo_data_1.csv",
    path_region_2: str = "Data/geo_data_2.csv",
) -> dict:
    """
    Carga los datos de las tres regiones y devuelve
    un diccionario {nombre_region: DataFrame}.
    """
    df0 = pd.read_csv(path_region_0)
    df1 = pd.read_csv(path_region_1)
    df2 = pd.read_csv(path_region_2)

    return {"0": df0, "1": df1, "2": df2}


def train_region(df: pd.DataFrame, region_name: str, random_state: int = RANDOM_STATE) -> dict:
    """
    Entrena un modelo de regresión lineal para una región.

    Devuelve un diccionario con:
    - model: modelo entrenado
    - features_valid: features del conjunto de validación
    - target_valid: objetivo real del conjunto de validación
    - preds_valid: predicciones en el conjunto de validación
    - rmse, r2, mean_pred: métricas básicas
    """
    features = df[["f0", "f1", "f2"]]
    target = df["product"]

    features_train, features_valid, target_train, target_valid = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(features_train, target_train)

    preds_valid = pd.Series(model.predict(features_valid), index=target_valid.index)

    rmse = mean_squared_error(target_valid, preds_valid, squared=False)
    r2 = r2_score(target_valid, preds_valid)
    mean_pred = preds_valid.mean()

    return {
        "region": region_name,
        "model": model,
        "features_valid": features_valid,
        "target_valid": target_valid,
        "preds_valid": preds_valid,
        "rmse": rmse,
        "r2": r2,
        "mean_pred": mean_pred,
    }


def calculate_profit(
    target_valid: pd.Series,
    predictions_valid: pd.Series,
    budget: float = BUDGET,
    revenue_per_unit: float = REVENUE_PER_THOUSAND_BARRELS,
    n_wells_in_sample: int = N_WELLS_PER_REGION,
    n_top_wells: int = N_TOP_WELLS,
) -> float:
    """
    Calcula el beneficio esperado para una muestra de pozos.

    Pasos:
    - toma una muestra de n_wells_in_sample pozos (ya muestreada fuera)
    - selecciona los n_top_wells con mayor predicción
    - calcula el ingreso usando los valores reales (target_valid)
    - resta el presupuesto invertido
    """
    # Ordenamos índices de mayor a menor predicción
    order = predictions_valid.sort_values(ascending=False)
    selected_idx = order.head(n_top_wells).index

    # Volumen real de producto (en miles de barriles)
    total_product = target_valid.loc[selected_idx].sum()

    revenue = total_product * revenue_per_unit
    profit = revenue - budget

    return profit


def bootstrap_profit(
    target_valid: pd.Series,
    predictions_valid: pd.Series,
    n_iter: int = N_BOOTSTRAP_ITER,
    n_wells_in_sample: int = N_WELLS_PER_REGION,
    n_top_wells: int = N_TOP_WELLS,
    budget: float = BUDGET,
    revenue_per_unit: float = REVENUE_PER_THOUSAND_BARRELS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Aplica bootstrapping para estimar la distribución del beneficio.

    Devuelve un diccionario con:
    - mean_profit: beneficio promedio
    - ci_low, ci_high: intervalo de confianza al 95%
    - risk: probabilidad de obtener beneficio negativo
    """
    rng = np.random.default_rng(random_state)
    indices = target_valid.index.to_numpy()

    profits = []

    for _ in range(n_iter):
        sample_idx = rng.choice(indices, size=n_wells_in_sample, replace=True)
        sample_target = target_valid.loc[sample_idx]
        sample_pred = predictions_valid.loc[sample_idx]

        profit = calculate_profit(
            target_valid=sample_target,
            predictions_valid=sample_pred,
            budget=budget,
            revenue_per_unit=revenue_per_unit,
            n_wells_in_sample=n_wells_in_sample,
            n_top_wells=n_top_wells,
        )
        profits.append(profit)

    profits = np.array(profits)
    mean_profit = profits.mean()
    ci_low, ci_high = np.percentile(profits, [2.5, 97.5])
    risk = (profits < 0).mean()

    return {
        "mean_profit": mean_profit,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "risk": risk,
        "profits": profits,
    }


def main():
    # 1. Cargar datos
    data_by_region = load_region_data()

    # 2. Entrenar modelo por región y mostrar métricas
    region_results = []

    print("=== Resultados de entrenamiento por región ===")
    for region_name, df in data_by_region.items():
        res = train_region(df, region_name)
        region_results.append(res)

        print(f"Región {region_name}")
        print(f"  RMSE:       {res['rmse']:.2f}")
        print(f"  R²:         {res['r2']:.3f}")
        print(f"  Media pred: {res['mean_pred']:.2f}")
        print("-" * 40)

    # 3. Análisis de beneficio y riesgo por región
    print("\n=== Beneficio y riesgo estimados (bootstrapping) ===")
    summary_rows = []

    for res in region_results:
        stats = bootstrap_profit(
            target_valid=res["target_valid"],
            predictions_valid=res["preds_valid"],
        )

        summary_rows.append(
            {
                "Región": res["region"],
                "Beneficio promedio (USD)": stats["mean_profit"],
                "IC95% low (USD)": stats["ci_low"],
                "IC95% high (USD)": stats["ci_high"],
                "Riesgo de pérdida (%)": stats["risk"] * 100,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.round(2).sort_values("Beneficio promedio (USD)", ascending=False))


if __name__ == "__main__":
    main()
