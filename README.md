# ⛽ Selección de Región Petrolera Óptima

[![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.10-blue.svg)](https://www.python.org/)
[![Data Science Bootcamp](https://img.shields.io/badge/Proyecto-Bootcamp-green)](#)
[![Status](https://img.shields.io/badge/Status-Completo-brightgreen)](#)

---

Este proyecto pertenece al sector **Oil & Gas** y tiene como objetivo elegir la **mejor región** para desarrollar nuevos pozos petroleros.  

Se analizan tres regiones utilizando datos geológicos (`f0`, `f1`, `f2`) y la variable objetivo `product` (volumen de producto en miles de barriles).  
A través de un modelo de **regresión lineal** y simulaciones de **bootstrapping**, se estiman el beneficio esperado y el riesgo de pérdida en cada región, bajo un presupuesto fijo de inversión.

Este proyecto conecta directamente mi perfil de **Ingeniero Petrolero** con **Ciencia de Datos aplicada al negocio energético**.

---

## 📌 Objetivo

Determinar en qué región conviene invertir para:

- Maximizar el **beneficio promedio esperado**.
- Mantener bajo control el **riesgo de obtener pérdidas**.
- Utilizar un modelo de ML (regresión lineal) entrenado con datos geológicos para estimar la producción futura.

---

## 🛠️ Herramientas utilizadas

- `Python`
- `Pandas`, `NumPy`
- `scikit-learn` (LinearRegression, train_test_split, métricas)
- `Jupyter Notebook`
- Estadística aplicada: **bootstrapping** e **intervalos de confianza**

---

## 📊 Contenido del análisis

- ✔ Carga y revisión inicial de las tres regiones (`geo_data_0`, `geo_data_1`, `geo_data_2`).
- ✔ Verificación de:
  - Valores nulos.
  - Filas duplicadas.
  - Distribución de la variable objetivo (`product`).
- ✔ Entrenamiento de un modelo de **regresión lineal** por región:
  - División en entrenamiento y validación (75/25).
  - Cálculo de métricas:
    - `RMSE` (error cuadrático medio raíz).
    - `R²` (coeficiente de determinación).
    - Media de las reservas predichas.
- ✔ Simulación del escenario de negocio:
  - Presupuesto fijo para perforar **500 pozos** por región.
  - Selección de los **200 pozos con mayor producción estimada**.
  - Cálculo de ingreso con los valores reales y estimación de beneficio.
- ✔ **Bootstrapping (1000 iteraciones)**:
  - Muestreo repetido de 500 pozos.
  - Cálculo del beneficio usando los 200 mejores pozos en cada iteración.
  - Cálculo de:
    - Beneficio medio.
    - Intervalo de confianza al 95%.
    - Riesgo de obtener beneficio negativo.

---

## 📈 Resultados clave

- El análisis muestra que ninguna región alcanza, en promedio, el volumen requerido para garantizar beneficio si se consideran todos los pozos por igual.  
- Sin embargo, al seleccionar solo los **200 mejores pozos** en cada muestra:
  - Una de las regiones (Región 1) presenta:
    - **Beneficio promedio más alto**.
    - **Intervalo de confianza** que se mantiene principalmente en valores positivos.
    - **Riesgo de pérdida** sensiblemente menor que en las otras dos regiones.
- Las otras regiones (0 y 2) muestran buenos pozos individuales, pero:
  - El riesgo de cerrar en pérdidas es más alto.
  - Los intervalos de confianza presentan mayor probabilidad en zona negativa.

---

## 🧠 Conclusión

- La región recomendada para inversión adicional es la **Región 1**, ya que:
  - Ofrece el **mejor equilibrio** entre beneficio esperado y riesgo.
  - Su distribución de beneficios se concentra por encima de cero.
- Las regiones 0 y 2 podrían considerarse solamente bajo escenarios de mayor tolerancia al riesgo o como complemento secundario.
- Este enfoque permite:
  - Traducir **variables geológicas** a **decisiones económicas**.
  - Conectar análisis técnico de pozos con métricas de negocio (beneficio, riesgo, probabilidad de pérdida).

Este proyecto demuestra cómo la **Ciencia de Datos** puede apoyar decisiones estratégicas en **Oil & Gas**, seleccionando áreas de explotación con base en modelos de ML y análisis estadístico.

---

## 📁 Estructura del proyecto

```text
Seleccion-Region-Petrolera/

├── Src/
│   └── Modelo_Seleccion_Region_OilGas.py      # Código fuente limpio con el pipeline de ML
│
├── Notebooks/
│   └── Proyecto_Sprint11_Seleccion_Region.ipynb   # Notebook con el desarrollo paso a paso
│
├── Data/
│   ├── geo_data_0.csv
│   ├── geo_data_1.csv
│   └── geo_data_2.csv
│
├── requirements.txt                         # Librerías necesarias
├── .gitignore
└── README.md
Asegúrate de que las rutas del script (Data/geo_data_0.csv, etc.) coincidan con los nombres reales de tus archivos en el repo.
```

## 👨‍💻 Autor

Axel López

🔗 LinkedIn https://www.linkedin.com/in/axel-l%C3%B3pez-linares/

✉️ axellpzlin@gmail.com

🎯 Proyecto de portafolio - Bootcamp de Ciencia de Datos (Oil & Gas)
