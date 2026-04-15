import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
from scipy import stats



df = pd.read_csv('datos_farmacias_origen_segmentado.csv', encoding='utf-8', sep=";")

CATEGORIA_MAP = {
    # Analgésicos / Antiinflamatorios
    "acetaminofen":     "analgesico",
    "ibuprofeno":       "analgesico",

    # Antibióticos / Antiinfecciosos
    "amoxicilina":      "antibiotico",
    "metronidazol":     "antibiotico",
    "nitrofurantoina":  "antibiotico",

    #Antifúngico
    "fluconazol": "antifúngico",

    # Cardiovascular / Metabólico
    "amlodipina":       "cardiovascular",
    "losartan potasico":"cardiovascular",
    "atorvastatina":    "cardiovascular",
    "metformina":       "cardiovascular",  # antidiabético, pero comparte perfil de precio

    # Antihistamínicos
    "cetirizina":       "antihistaminico",
    "loratadina":       "antihistaminico",

    # Neurológico
    "acido valproico":  "neurológico",

    #Antidepresivo

    "sertralina": "antidepresivo",

    # Gastrointestinal
    "omeprazol":        "gastrointestinal",
}

def clasificar(nombre):
    n = str(nombre).lower().strip()
    # Busca coincidencia parcial para cubrir variantes de nombre
    for clave, categoria in CATEGORIA_MAP.items():
        if clave in n:
            return categoria
    return "otro"


df['Categoria'] = df['Principio_Activo'].apply(clasificar)


# Se generan las variables dummies. Drop_first=True elimina la primera categoría definida con CategoricalDType


cat_Segmento = pd.CategoricalDtype(categories=['Medio', 'Alto', 'Bajo'], ordered=True)
df['Segmento'] = df['Segmento'].astype(cat_Segmento)

print(df['Categoria'].value_counts())

df = pd.get_dummies(df, columns=['Categoria'], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=['Cadena_Farmacia'], drop_first=True, prefix='Cadena', dtype=int)
df = pd.get_dummies(df, columns=['Segmento'], drop_first=False, dtype=int)


# Muestra solo las columnas nuevas
nuevas = [c for c in df.columns if c.startswith('Categoria_') or c.startswith('Cadena_')]
print(nuevas)


#Aleatorizar la muestra para evitar un sesgo de ordenamiento sistemático

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


features = [
    'Dosis_mg',
    'Unidades_por_Caja',
    'Es_Marca',
    'Categoria_antibiotico',
    'Categoria_antidepresivo',
    'Categoria_antifúngico',
    'Categoria_cardiovascular',
    'Categoria_gastrointestinal',
    'Categoria_neurológico',
    'Segmento_Alto',
    'Segmento_Bajo'
]

X = df[features]
y_log = np.log1p(df['Precio_USD'])
X_with_const = sm.add_constant(X)
model = sm.OLS(y_log, X_with_const).fit()


print("\n=== MODEL RESULTS ===")
print(model.summary())

#residuos

residuos = model.resid

# residuos

df['residuos'] = model.resid
df['y_hat'] = model.fittedvalues

# Muestra las observaciones con residuo > 1.0
outliers = df[df['residuos'] > 1.0][['ID_Medicamento', 'Principio_Activo',
                                      'Nombre_Comercial', 'Laboratorio',
                                      'Precio_USD', 'Es_Marca', 'residuos']]
print(outliers)

#Prueba de Homocedasticidad Breusch-Pagan
bp_final = het_breuschpagan(residuos, model.model.exog)
print(f"--- 3. p-value Breusch-Pagan (Final): {bp_final[1]}")

#Prueba de Autocorrelación Breusch-Godfrey
bg_test = acorr_breusch_godfrey(model, nlags=1)

print(f"Breusch-Godfrey Lagrange Multiplier statistic: {bg_test[0]:.4f}")
print(f"Breusch-Godfrey p-value: {bg_test[1]:.4f}")

# VIF

X_vif = X
vif_final = pd.DataFrame()
vif_final["Feature"] = features
vif_final["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(features))]
print(vif_final)

#Prueba de Normalidad Shapiro-Wilk
stat_shapiro, p_valor_shapiro = stats.shapiro(residuos)

print("\n=== PRUEBA DE NORMALIDAD (SHAPIRO-WILK) ===")
print(f"Estadístico W: {stat_shapiro:.4f}")
print(f"p-valor: {p_valor_shapiro:.4e}")

#Gráfico Q-Q Plot

#    1. Extraer las estadísticas de influencia del modelo
influencia = model.get_influence()

#    2. Obtener los residuos studentizados (internos)
residuos_studentizados = influencia.resid_studentized_internal

#    3. Crear el Q-Q Plot
# line='45' dibuja la línea diagonal de referencia
# fit=True ajusta los datos estándar para la comparación
fig = sm.qqplot(residuos_studentizados, line='45', fit=True)

#    4. Ajustes visuales
plt.title('Q-Q Plot de Residuos Studentizados')
plt.xlabel('Cuantiles Teóricos (Distribución Normal)')
plt.ylabel('Residuos Studentizados')
plt.grid(True, linestyle='--', alpha=0.7)


#    5. Mostrar el gráfico
plt.show()


# Grafica residuos vs valores predichos
plt.scatter(model.fittedvalues, residuos, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores ajustados (log)')
plt.ylabel('Residuos')
plt.title('Residuos vs Valores Ajustados')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Regresión Lineal con Scikit-Learn
model = LinearRegression()
model.fit(X_train, y_train)

#Coeficientes de cada variable

for i, name in enumerate(X):
    print(f'{name:>10}: {model.coef_[i]}')
# Revisión de precisión en el set de prueba

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"Model Accuracy (R2 Score): {r2:.2f} (Closer to 1.0 is better)")

# Hacer predicciones para el dataset entero
#Esto agrega una columna llamada "Precio_USD_Estimado"

log_preds = model.predict(X)
df['Precio_USD_Estimado'] = np.expm1(log_preds)
df['Precio_USD_Estimado'] = df['Precio_USD_Estimado'].round(2)

# Let's see the difference for the first 5 players
print(df[['ID_Medicamento', 'Precio_USD', 'Precio_USD_Estimado']].head())

# Convertir el conjunto de prueba de Y a dinero real (no logarítmico)
y_test_real = np.expm1(y_test)
# Convertir el conjunto de prueba de X a dinero real (no logarítmico)
predictions_test_real = np.expm1(model.predict(X_test))

mae_usd = mean_absolute_error(y_test_real, predictions_test_real)
print(f"En promedio, el modelo se equivoca por: ${mae_usd:,.2f}")

mape = np.mean(np.abs((y_test_real - predictions_test_real) / y_test_real)) * 100
print(f"Error Porcentual Absoluto Promedio: {mape:.2f}%")
df.to_csv('datos_farmacias_con_estimaciones.csv', index=False, encoding='utf-8-sig', sep=',', decimal='.')
print("Archivo guardado")



