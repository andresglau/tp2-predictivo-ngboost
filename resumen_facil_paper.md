# NGBoost — Resumen fácil (pre-lectura del paper)

Guía para leer antes de meterse con el paper original (`ngboostpaper.pdf`). Está explicado bien simple, con un ejemplo que se repite en todo el documento: **adivinar cuántos caramelos hay en un frasco**.

## 1. El problema: adivinar "un número" no alcanza

Si decís "hay 47 caramelos", eso es una **adivinanza puntual**. Lo más honesto sería decir "creo que 47, pero podrían ser entre 40 y 54". Eso ya no es un número, es un **rango con una forma** — como cuando el pronóstico del clima dice "20°, pero podría variar entre 18° y 22°".

Eso es una **predicción probabilística**: en vez de devolver un número, devolvés una **distribución de probabilidad completa**. Importa mucho en medicina o clima, donde saber "cuán seguro estoy" es tan importante como la respuesta misma.

## 2. Por qué esto era difícil antes

- **Métodos bayesianos**: potentes pero lentos y difíciles de usar.
- **GAMLSS**: permite variar la incertidumbre, pero con una fórmula rígida elegida de antemano.
- **Redes neuronales bayesianas**: funcionan, pero son lentas y difíciles de calibrar.
- **Gradient boosting normal (XGBoost, etc.)**: rápido y fácil, pero en regresión solo da un número, no un rango.

NGBoost junta lo mejor: la facilidad y velocidad del boosting + la capacidad de dar un rango de confianza, sin ser experto.

## 3. La idea central de NGBoost

En vez de aprender a predecir "un número" (la media), el modelo aprende a predecir **los parámetros de una distribución completa**.

Ejemplo del frasco: en vez de aprender a decir "47", aprende dos cosas a la vez:
- **μ (mu)** — el centro de la estimación (47).
- **σ (sigma)** — cuán ancho es el margen de error (±7).

Esos dos números definen una "campana" (Normal) que dice "creo que está centrado en 47, y la mayoría de las veces entre 40 y 54". No tiene que ser una campana: puede ser Poisson (contar cosas), Weibull o Gamma (tiempo hasta que algo ocurre, común en medicina), etc.

NGBoost se arma con **tres piezas intercambiables** (como Lego):

1. **Base learner** — el "ayudante" que predice en cada paso (normalmente un arbolito de decisión).
2. **Distribución** — la forma del rango de confianza (Normal, Poisson, Weibull...).
3. **Scoring rule** — la regla para calificar qué tan bien adivinó (ver siguiente punto).

## 4. ¿Cómo sabe el modelo si adivinó bien? (Scoring rules)

Para un solo número se usa un "error" (ej. error al cuadrado). Para una distribución entera hace falta una **proper scoring rule**: una regla de puntaje donde la mejor forma de ganar puntos es **decir la verdad** — ni exagerar la seguridad ni fingir más incertidumbre de la que hay.

El paper usa dos:
- **Log Score (MLE)** — castiga fuerte si le asignó poca probabilidad al valor real. La más común.
- **CRPS** — alternativa más robusta ante datos raros (outliers).

## 5. El problema del "paso equivocado" (por qué hace falta el Gradiente Natural)

Es la parte más importante del paper. Para aprender, el modelo corrige sus parámetros (μ y σ) de a poco, dando "pasos" en una dirección (el **gradiente**). El problema: la "distancia" entre dos valores de parámetros no siempre significa lo mismo (cambiar σ de 1 a 2 duplica la campana; cambiar μ de 1 a 2 puede ser un cambio chico si la campana es ancha).

**Analogía:** caminar con el piso mitad hielo, mitad arena pegajosa. Si das pasos del mismo tamaño sin importar el piso, en el hielo casi no avanzás y en la arena te hundís gastando energía de más. Eso le pasa al **gradiente común (ordinario)**.

El **gradiente natural** mide la distancia *real* entre distribuciones (no entre números de parámetros) y ajusta el tamaño del paso según el "terreno" en cada punto (usa la matriz de información de Fisher para corregirlo). Resultado: aprendizaje más parejo y estable — en el paper se ve en la Figura 4: con gradiente común, los ejemplos "con suerte" dominan el aprendizaje; con gradiente natural, todos aprenden a un ritmo parecido.

## 6. El algoritmo, paso a paso (en criollo)

1. Arranca adivinando algo simple: el promedio general, sin mirar pistas (θ⁽⁰⁾).
2. En cada ronda (de M rondas totales):
   - Mide cuánto se equivocó con cada ejemplo, usando el gradiente natural (el paso "bien medido").
   - Entrena un arbolito que aprenda a predecir esa corrección.
   - Decide cuánto peso darle a la corrección (búsqueda simple: probar, si no mejora, reducir a la mitad).
   - Suma esa corrección (escalada por una tasa de aprendizaje chica) a la predicción actual.
3. Al final, la suma de todas las correcciones da la distribución final para cada x.

Es boosting normal (sumar arbolitos que corrigen al anterior), pero corrigiendo **todos los parámetros de la distribución a la vez**, no solo un número.

## 7. ¿Funciona en la práctica? (Experimentos)

Probado en 10 datasets típicos (UCI: precios de casas, energía, vino, etc.) contra: redes con dropout (MC Dropout, Concrete Dropout), ensembles de redes (Deep Ensembles), Procesos Gaussianos, GAMLSS y Bosques distribucionales.

- **Resultado:** NGBoost empata o gana contra casi todos, sobre todo en datasets chicos, siendo mucho más simple de usar.
- Incluso el número puntual (la media de la distribución predicha) es competitivo contra Random Forest, Gradient Boosting normal y Elastic Net — casi "gratis", como premio extra por predecir la distribución completa.
- **Ablation (prueba de que cada pieza importa):** sin el gradiente natural (usando gradiente común o de "segundo orden"), el modelo anda mucho peor. Confirma que el gradiente natural no es un detalle cosmético, es la pieza clave.

## 8. Ventajas y desventajas

**Ventajas**
- No hace falta ser experto en estadística bayesiana ni en redes neuronales — funciona "de fábrica".
- Modular: cambiás la distribución según el problema (conteos, supervivencia, clasificación, regresión) sin cambiar de herramienta.
- Da incertidumbre real, no solo un número.
- Escala bien con muchos datos, igual que cualquier boosting (el costo extra depende de p, la cantidad de parámetros de la distribución, que suele ser 1 o 2).
- API familiar de scikit-learn (`.fit()`, `.predict()`, `.pred_dist()`).

**Desventajas / limitaciones**
- Más lento que un boosting normal: entrena un arbolito por cada parámetro de la distribución, y además invierte una matriz p×p por cada dato en cada ronda.
- Si la distribución elegida no encaja con los datos reales, la incertidumbre predicha puede ser mala.
- El paper reconoce que todavía no hay garantías teóricas rigurosas de convergencia — pregunta abierta para trabajo futuro.
- Para predicción puntual pura (sin importar la incertidumbre), un gradient boosting normal bien afinado le sigue ganando un poco en RMSE.

## 9. La frase de cierre (de 7 años)

> Antes, las máquinas de adivinar números (boosting) solo decían *un* número. NGBoost les enseñó a decir *un número y cuán seguras están*, dando pasos "medidos con la regla correcta" (gradiente natural) en vez de pasos a lo loco. Y funciona casi tan rápido y fácil como las máquinas de antes.

---

*Fuente: Duan, Avati, Ding, Thai, Basu, Ng, Schuler (2019), "NGBoost: Natural Gradient Boosting for Probabilistic Prediction", arXiv:1910.03225 — ver `ngboostpaper.pdf` para el texto completo y las tablas de resultados.*
