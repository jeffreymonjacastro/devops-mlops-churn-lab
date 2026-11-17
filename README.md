# MLOps Churn Lab

## Laboratorio 14.1: resultados locales de las fases 3 y 4

Ejecución del 2026-11-11 con 240 clientes de entrenamiento, 60 de test y
validación cruzada estratificada de 5 folds. El test no participa en la selección
del modelo ni del umbral.

| Candidato | F1 CV | ROC-AUC CV |
|---|---|---|
| logistic | 0.6666 | 0.7315 |
| random_forest | 0.6900 | 0.7326 |
| xgboost | 0.6829 | 0.7100 |
| gradient_boosting (reto obligatorio) | 0.6476 | 0.6858 |

El cuarto candidato no cambia el ganador: se elige random_forest por F1 CV.
El baseline constante positivo obtiene F1 CV 0.6629.

El umbral elegido por costo con predicciones out-of-fold de train es 0.30.
En test: F1 0.7013, precision 0.5745, recall 0.9000, ROC-AUC 0.7989 y costo
S/ 43.33 por cliente, frente a S/ 50.00 al contactar a todos. Los cuatro checks
del quality gate pasan. Las métricas describen este dataset pequeño, no una
validación de resultados en producción.

Para reproducir desde la raíz del repositorio:

```powershell
conda activate churn-lab
python src/validate_data.py
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
pytest tests/ -q
mlflow ui --backend-store-uri mlruns --port 5001
```

Para las capturas, abre http://localhost:5001:

1. En **Experiments**, abre `churn-prediction-fintech`. Selecciona los cinco runs
   (baseline y cuatro candidatos) y abre la comparación o vista de gráficos.
   Muestra las métricas `cv_f1_mean`, `cv_f1_std` y `cv_roc_auc_mean`.
2. En **Models**, abre `churn-predictor`. Captura las versiones y sus tags
   `quality_gate`, `test_f1`, `threshold`, junto con los aliases `champion` y
   `challenger`. Los números de versión dependen de las evaluaciones ejecutadas.

El entorno del laboratorio fija MLflow 2.22.1, como las capturas de referencia.
Cada nueva evaluación crea
una versión real; no hace falta repetir entrenamientos para llenar la tabla.

Los resultados detallados están en `reports/evaluation_report.json`,
`reports/model_card.md` y `data/production_model_info.json`.

Punto de partida de los laboratorios 14.1 y 14.2: predicción de churn para una fintech siguiendo CRISP-ML(Q).

El proyecto funciona, sus tests pasan y tiene un bug que solo aparece en producción. Encontrarlo y eliminarlo es el primer paso del laboratorio.

## El caso

**Predicción de Churn para una Fintech**

Una fintech detectó un incremento en la pérdida de clientes (churn). El área de analítica ya construyó una primera solución: entrena un modelo, registra experimentos en MLflow, lo publica como API y tiene un pipeline en GitHub Actions. Todo corre y los tests pasan, pero el negocio desconfía de las predicciones y nadie puede explicar por qué se eligió ese modelo ni cuándo habría que reemplazarlo.

El reto del equipo es llevar esa solución a un nivel confiable siguiendo CRISP-ML(Q):

- Validar los datos antes de entrenar.
- Garantizar que la API prediga lo mismo que el modelo evaluado.
- Elegir el modelo con evidencia y decidir con criterios de negocio.
- Promover a producción solo lo que pasa un quality gate.
- Detectar cuándo el modelo deja de funcionar y reentrenarlo.
- Documentar riesgos, controles y decisiones.

## Cómo empezar

1. Haz **fork** de este repositorio a tu cuenta.
2. Clona tu fork (reemplaza `TU-USUARIO` por tu usuario de GitHub) y crea el entorno:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
git checkout -b lab/14.1

conda create -n churn-lab python=3.10 -y
conda activate churn-lab
pip install -r requirements.txt "setuptools<81"
```

3. Comprueba que todo corre:

```bash
pytest tests/ -q                      # 18 passed
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
uvicorn app.main:app --port 8000      # http://localhost:8000/docs
```

4. Sigue la guía: [`docs/11. mlops-churn-crisp-mlq-parte-1.md`](docs/11.%20mlops-churn-crisp-mlq-parte-1.md).

## Estructura

```
.
├── app/main.py                     # API FastAPI: /predict, /predict/batch, /health
├── src/
│   ├── data_prep.py                # limpieza, encoding, escalado y split
│   ├── train.py                    # 3 modelos con MLflow tracking
│   └── evaluate_and_register.py    # registro en MLflow Model Registry
├── tests/                          # tests de la API con el modelo mockeado
├── data/
│   └── telco_customer_churn_mlops.csv   # 300 clientes, 7 features, target churn
├── docs/                           # guía del laboratorio
├── .github/workflows/mlops.yml     # test → data-prep → train → register → smoke-test
├── Dockerfile
└── requirements.txt
```

## Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `customer_id` | texto | Identificador del cliente |
| `gender` | categoría | Male, Female |
| `senior_citizen` | 0/1 | Adulto mayor |
| `tenure_months` | entero | Meses como cliente |
| `contract_type` | categoría | Month-to-Month, One Year, Two Year |
| `internet_service` | categoría | Fiber, DSL, None |
| `monthly_charges` | decimal | Cargo mensual |
| `support_tickets_last_6m` | entero | Tickets de soporte en 6 meses |
| `churn` | Yes/No | El cliente abandonó (target) |

## Qué vas a construir

| Laboratorio | Fases CRISP-ML(Q) | Resultado |
|---|---|---|
| 14.1 | 1 a 4: negocio y datos, preparación, modelado, evaluación | Contrato de datos, Pipeline de scikit-learn, validación cruzada, umbral por costo, modelo `@champion` |
| 14.2 | 5 y 6: despliegue, monitoreo | API sin skew, imagen en Docker Hub, drift, reentrenamiento automático |
