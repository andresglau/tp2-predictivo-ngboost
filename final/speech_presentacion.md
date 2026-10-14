# Speech — NGBoost (Examen 2, Análisis Predictivo)

Guión completo para presentar `NGBoost — Análisis Predictivo.pptx`, slide por slide. Pensado para **20-30 minutos** hablando a ritmo tranquilo (la consigna pide explícitamente "preferible hablar más lento y cubrir menos que hablar más rápido y cubrir mucho"). Los tiempos son orientativos; súmalos y da ~26-28 min, dejando margen para preguntas.

**Cómo usarlo:** es un guión para decir con tus palabras, no para leer textual. Está cortado en bloques naturales por si el grupo se quiere repartir la presentación — los cortes lógicos son después de la slide 5 (termina la parte de "contexto/comparación"), después de la slide 10 (termina la teoría, arranca la demo) y después de la slide 14 (termina el balance, arranca el cierre).

---

## Slide 1 — Portada (≈1 min)

Buenas, somos el grupo [nombres] y venimos a presentar **NGBoost**, Natural Gradient Boosting for Probabilistic Prediction. Es un método de boosting publicado en 2019 por un equipo de Stanford, y la idea central que lo distingue de un boosting clásico como XGBoost es esta: en vez de predecir un solo número para cada observación, predice una **distribución de probabilidad completa**.

La estructura de la charla va a ser así: primero el problema que motiva esto, después la idea y cómo funciona por dentro — incluyendo la parte matemática, que vamos a tratar de explicar bien y no solo mostrar la fórmula —, después un par de demos corriendo el código en Python, la evidencia que trae el paper, y cerramos con ventajas, desventajas y cuándo conviene usarlo.

## Slide 2 — El problema (≈2 min)

Arrancamos con un ejemplo simple. Supongamos que queremos predecir el precio de una casa, y tenemos dos casas distintas, A y B. Un modelo de regresión clásico — una regresión lineal, un Random Forest, un XGBoost — le va a asignar a las dos **exactamente la misma predicción**: 150 mil dólares.

El problema es que puede estar pasando algo muy distinto por detrás. Para la casa A, el modelo está prácticamente seguro: el desvío es de solo 8 mil, así que el 95% de las veces el precio real cae entre 134 y 166 mil. Para la casa B, con un desvío de 40 mil, esa misma confianza del 95% cubre un rango brutal: entre 72 y 228 mil dólares. Son niveles de certeza completamente distintos, y un modelo de predicción puntual — que solo estima la esperanza condicional, E de y dado x — no tiene ninguna forma de decírnoslo. Las dos casas quedan indistinguibles.

Y esto importa en la práctica: si estás fijando un precio, aprobando un crédito, o prometiendo un plazo de entrega, no es lo mismo apostar fuerte a un número que estar adivinando a ciegas.

## Slide 3 — La idea (≈2.5 min)

La propuesta de NGBoost es simple de enunciar: en vez de modelar E de y dado x, un número, modela **toda la distribución de y dado x** — lo que se llama regresión probabilística. La notación es esta: y dado x se distribuye según una familia D, con parámetros theta que son función de x. Si elegís una Normal, son dos parámetros, mu y sigma; si elegís una Poisson, es un solo parámetro, lambda.

Para que quede bien claro con un ejemplo cotidiano: pensemos en adivinar cuántos caramelos hay en un frasco. Decir "hay 47 caramelos" es una adivinanza puntual. Decir "creo que son 47, pero podrían ser entre 40 y 54" ya no es un número, es una distribución — un centro y un margen de error. Eso es exactamente lo que hace NGBoost: en vez de aprender a decir "47", aprende a decir el centro **y** el margen, a la vez, para cada observación.

Esto funciona en tres momentos. Antes de entrenar, elegís la familia de distribución — Normal es la opción por defecto, pero también hay LogNormal, Gamma, Poisson, y varias más. Durante el entrenamiento, el modelo aprende a predecir **todos** los parámetros de esa distribución como funciones de x, no solo la media. Y al predecir, cada observación nueva recibe su propia distribución completa: de ahí podés sacar intervalos de confianza, cuantiles, o la probabilidad de cualquier evento que te interese.

Un punto importante para no confundirse: esto no es la distribución de todo el dataset. Es la distribución de y para una observación particular, dadas sus características. Todas las observaciones comparten la misma familia — por ejemplo, todas son Normales — pero cada una tiene sus propios parámetros, igual que en clasificación ya estamos acostumbrados a que el modelo devuelva una probabilidad en vez de una etiqueta directa. NGBoost lleva esa misma lógica a los problemas de regresión.

## Slide 4 — Contexto: las alternativas que ya existían (≈2 min)

Esto no es un problema nuevo — ya había formas de atacarlo, pero todas con algún costo grande. Una regresión con sigma constante estima la incertidumbre como el desvío de los residuos, pero asume homocedasticidad: que el margen de error es igual para todas las observaciones, algo que casi nunca se cumple en la realidad. GAMLSS sí permite que mu y sigma varíen en función de x, pero hay que fijar la forma funcional del modelo de antemano, a mano. Los métodos bayesianos dan una distribución posterior completa de los parámetros, pero típicamente necesitan MCMC, que es lento, escala mal con el volumen de datos, y requiere bastante expertise para calibrar bien. Y el deep learning probabilístico — dropout, ensembles de redes — en datos tabulares, que es el tipo de dato más común en la práctica, generalmente no le gana a los métodos clásicos, y necesita mucho tuning.

Mientras tanto, los gradient boosting clásicos —XGBoost, LightGBM— funcionan extremadamente bien en datos tabulares, incluso con pocos datos, son rápidos y fáciles de usar. Pero en regresión, solo dan un número. La propuesta de NGBoost es juntar lo mejor de los dos mundos: la flexibilidad y la facilidad de uso del boosting, con una salida probabilística.

## Slide 5 — XGBoost vs. NGBoost (≈2 min)

Para que quede bien concreto, comparemos los dos lado a lado. Los dos son gradient boosting: suman árboles chicos entrenados en secuencia, cada uno corrigiendo lo que el conjunto anterior hizo mal. La diferencia está en **qué** se está boosteando.

XGBoost en regresión da un número, y_hat; NGBoost da una distribución completa, P de y dado x. XGBoost entrena un árbol por etapa; NGBoost entrena **un árbol por cada parámetro** de la distribución — con una Normal, son dos árboles por etapa, uno para mu y otro para sigma. Lo que aprende cada árbol también cambia: XGBoost aprende el gradiente de la función de pérdida, usando además el Hessiano, información de segundo orden; NGBoost aprende el **gradiente natural** del scoring rule, que es el concepto clave que vamos a explicar en un par de slides. El objetivo de XGBoost suele ser el error cuadrático medio; el de NGBoost es el log score, es decir la verosimilitud negativa, o alternativamente el CRPS. Y en cuanto a velocidad, XGBoost está hiperoptimizado en C++; NGBoost corre en Python sobre scikit-learn, así que es bastante más lento.

Un dato lindo para cerrar esta comparación: si fijás la incertidumbre constante y usás una Normal con log score, NGBoost **se reduce exactamente** al boosting clásico con error cuadrático medio. No es que lo reemplace — lo generaliza.

## Slide 6 — Un método modular, en tres piezas (≈1.5 min)

Esta es la Figura 2 del paper. NGBoost se arma con tres piezas intercambiables, como si fueran piezas de Lego. Primero, el **base learner**: el modelo chico que corrige en cada ronda, normalmente un árbol de decisión de profundidad 3. Segundo, la **distribución**: Normal, LogNormal, Poisson, Exponencial — la familia que mejor describa tus datos. Y tercero, la **scoring rule**: la regla que usa para calificar qué tan buena fue la distribución predicha, comparándola contra lo que realmente pasó.

Las variables x entran a los árboles base, que devuelven los parámetros theta de la distribución para esa observación. Con esos parámetros se arma la distribución predicha, la scoring rule la compara contra el valor real, y esa comparación genera el gradiente natural que entrena la próxima ronda de árboles. Que estas tres piezas sean intercambiables es lo que el paper llama "modular": el mismo esqueleto sirve para regresión, conteos, supervivencia, lo que cambia es qué piezas elegís.

## Slide 7 — Cómo se califica una distribución (≈2 min)

Para entrenar un modelo de regresión normal usamos el error cuadrático. Para entrenar un modelo que predice una distribución entera, hace falta algo distinto: una **scoring rule**, que mira el valor real y se pregunta "¿cuánta probabilidad le había asignado el modelo a lo que efectivamente pasó?".

Pensemos en tres escenarios. Si el modelo predijo una distribución angosta y el valor real cae justo en el centro, le va muy bien — estaba seguro, y acertó. Si predijo una distribución angosta pero el valor real cae lejos, lo castiga fuerte — estaba seguro, y se equivocó, eso es lo peor que puede pasar. Y si predijo una distribución muy ancha, "por las dudas", nunca va a fallar del todo, pero tampoco aporta información — y eso **también** se castiga, aunque más suave. Esa es la gracia de una buena scoring rule: no te conviene ni exagerar la seguridad ni refugiarte en la vaguedad. La mejor estrategia es siempre decir la verdad sobre cuánto sabés.

La más usada es el **log score**: menos el logaritmo de la densidad que el modelo le asignó al valor real, que es lo mismo que la verosimilitud negativa, y es la métrica que usa el paper para comparar métodos. La alternativa es el **CRPS**, más robusto frente a valores extremos. Y como dato de cierre: si fijás la incertidumbre constante, el log score de una Normal se reduce exactamente al error cuadrático — el puente con el boosting de siempre.

## Slide 8 — El problema del gradiente común (≈2.5 min)

Acá entramos en la parte más importante del paper — la "N" de NGBoost. Para entrenar, el modelo tiene que corregir sus parámetros de a poco, dando pasos en la dirección que mejora el score. El problema es que un paso del mismo tamaño en el espacio de los parámetros no significa lo mismo en todos lados.

Miren este ejemplo: si sigma es chico — la distribución es angosta — mover mu en 1 unidad cambia completamente la distribución, las curvas de antes y de después casi no se superponen. Pero si sigma es grande — la distribución ya es ancha — mover mu en esa misma unidad apenas se nota, las dos curvas son casi idénticas. Sin embargo, para el **gradiente común**, el de toda la vida, esos dos pasos "miden" exactamente lo mismo: un paso de tamaño 1 en mu. El gradiente común mide distancias en los números de los parámetros, no en las distribuciones que esos números representan.

Y hay un problema extra: la misma Normal se puede escribir con (mu, sigma), con (mu, sigma al cuadrado), o con (mu, log sigma) — y el gradiente común te da un camino de entrenamiento **distinto** según cuál elijas. No es invariante a cómo parametrizás el problema. Con un solo parámetro esto ya es molesto; cuando tenés que mover varios parámetros a la vez, como acá, el entrenamiento queda totalmente desbalanceado: unos parámetros avanzan de más y otros casi no se mueven.

## Slide 9 — La solución: el gradiente natural (≈3 min)

La solución del paper es el **gradiente natural**. La fórmula es esta: el gradiente natural del score es igual a la inversa de la matriz I de theta, multiplicada por el gradiente común.

¿Qué es I de theta? Es la matriz de información de Fisher. Mide, para cada parámetro, cuán sensible es la distribución a moverlo — básicamente, el "mapa del terreno" del que hablábamos antes: dónde el espacio es elástico y hace falta empujar fuerte para que algo cambie, y dónde es rígido y un empujón chico ya mueve mucho la distribución.

Lo que hace la fórmula es simple pero poderoso: si un parámetro es muy sensible — la matriz de Fisher le asigna un valor alto ahí — el paso se achica. Si es poco sensible, el paso se agranda. El resultado son pasos medidos entre **distribuciones reales**, no entre números de parámetros sueltos: son invariantes a cómo elijas parametrizar, y quedan balanceados entre todos los parámetros a la vez. Para la Normal, si resolvés la cuenta con theta igual a (mu, log sigma), te queda algo muy lindo: el gradiente natural de mu termina siendo literalmente el residuo de toda la vida, predicción menos valor real, sin ningún sigma al cuadrado molestando — exactamente lo que uno querría intuitivamente.

Una pregunta que suele surgir: ¿y por qué no usar el método de Newton, como hace XGBoost con el Hessiano? La respuesta es que Newton usa la curvatura en el espacio de los parámetros, y con una parametrización como (mu, log sigma) ese problema deja de ser convexo — el entrenamiento se rompe. El gradiente natural, en cambio, no tiene ese problema porque mide la curvatura de la distribución, no de los números crudos.

## Slide 10 — El algoritmo, paso a paso (≈3 min)

Juntando todo, así es el algoritmo completo — es el Algoritmo 1 del paper.

**Paso 0, inicializar:** se busca una sola distribución que describa bien todo el target y, ignorando por completo las features. Para una Normal, es tan simple como el promedio y el desvío estándar de todo el dataset de entrenamiento. Todas las observaciones arrancan con exactamente el mismo theta.

Después, se repite lo siguiente en cada una de las M rondas de boosting. **Paso 1, gradientes:** para cada observación, se calcula su gradiente natural — cuánto y en qué dirección conviene corregir sus parámetros, dado su valor real. **Paso 2, árboles:** se entrena un árbol de decisión por cada parámetro — con una Normal, dos árboles — que aprende a predecir esos gradientes en función de las features x. Acá es donde el modelo empieza a generalizar: agrupa observaciones parecidas y aprende patrones, en vez de memorizar corrección por corrección. **Paso 3, el paso rho:** se busca el tamaño de paso con un line search bien simple — arranca en 1, y si no mejora el score, se divide por la mitad, así hasta que mejore. **Paso 4, actualizar:** theta se mueve eta por rho, donde eta es la tasa de aprendizaje — 0.01 en el paper —, y se repite todo el proceso M veces.

Al final, la suma acumulada de todas esas correcciones te da, para cada observación nueva, sus propios parámetros finales. Un detalle que vale la pena remarcar: alcanza con un solo rho por etapa, compartido entre todos los parámetros, porque el gradiente natural ya viene balanceado — si hiciera falta un rho distinto por parámetro, sería señal de que algo está mal calibrado.

El costo extra frente a un boosting común es entrenar p árboles por etapa, en vez de uno, e invertir una matriz p por p para cada observación. Como p suele ser 1 o 2 — Normal o Poisson —, el costo extra es manejable; con datasets muy grandes, se puede entrenar con minibatches.

*(Transición natural a la demo: "Ahora que vimos la teoría completa, vamos a mostrarles esto corriendo en código.")*

## Slide 11 — Demos de NGBoost aplicado (≈4-5 min, incluye mostrar notebook)

Armamos dos demos en Python, con la librería oficial `ngboost`, que tiene la misma API que scikit-learn — `.fit()`, `.predict()`, y además `.pred_dist()` para sacar la distribución completa.

**Demo 1** es un ejemplo de juguete en una dimensión que replica la Figura 1 del paper. Generamos datos sintéticos donde el ruido crece a medida que x aumenta — heterocedasticidad real, a propósito. *(Acá mostrar el notebook: `ngb.fit(X, y)`, y el gráfico con la banda de confianza.)* Miren cómo la banda gris es angosta cerca de x=0, donde el ruido real es chico, y se va ensanchando hacia x=10, donde el ruido real es grande. El modelo aprendió **las dos cosas a la vez** — la forma de la señal y cómo varía la incertidumbre — sin que se lo digamos explícitamente. Un Gradient Boosting clásico solo nos daría la línea negra del medio, sin la banda.

**Demo 2** usa datos reales: el dataset de California Housing, precios de casas, el mismo que usa el repo oficial de NGBoost. Entrenamos NGBoost y, como comparación, un `GradientBoostingRegressor` de scikit-learn clásico. *(Mostrar los números: RMSE de NGBoost 0.607 contra 0.523 del Gradient Boosting clásico.)* NGBoost pierde un poco en RMSE — es esperable, no está optimizado para minimizar el error puntual, sino la verosimilitud de la distribución completa — pero a cambio nos da el NLL, y sobre todo la banda de incertidumbre que el otro modelo directamente no puede dar. *(Mostrar el gráfico de barras de error contra los valores reales.)* Fíjense que la mayoría de los valores reales, las cruces, caen dentro del intervalo del 95% — eso es lo que significa que el modelo esté bien calibrado.

Y para que quede bien claro qué hay "adentro" de cada predicción: `pred_dist()` no devuelve un número, devuelve un objeto distribución, y de ahí podemos sacar mu, sigma, o cualquier percentil que necesitemos — literalmente el theta del que habla el paper, por cada observación de test.

## Slide 12 — Qué muestra el paper (≈2.5 min)

Más allá de nuestra demo, el paper hace experimentos serios: 10 datasets de regresión de UCI, desde 308 hasta más de 500 mil filas, con 20 particiones de train y test, usando Normal con árboles de profundidad 3.

Tres resultados. La **Tabla 1** compara el NLL contra otros métodos probabilísticos — redes bayesianas, procesos gaussianos, GAMLSS — y NGBoost es competitivo, no dominante en todos, pero se destaca especialmente en los datasets chicos. La **Tabla 2** es la más importante de todas: el ablation study, que prueba si cada pieza del método hace falta. Y la respuesta es clara: con el gradiente común en vez del natural, el modelo queda muchas veces peor que ni siquiera asumir sigma constante. El gradiente natural no es un detalle cosmético — es la pieza que hace que todo funcione. La **Tabla 3** mira la predicción puntual pura, usando la media de la distribución como si fuera cualquier y_hat: NGBoost le gana a Elastic Net, queda parecido a Random Forest, y apenas atrás de Gradient Boosting — pero pierde en algunos datasets puntuales, como Kin8nm, Power y Protein. La conclusión de esta parte es que la incertidumbre sale "casi gratis": no se paga casi nada en calidad de la predicción puntual por obtener, además, una distribución completa.

## Slide 13 — Ventajas (≈1.5 min)

Resumiendo las ventajas que respaldan tanto el paper como nuestra propia demo. La más importante: con una distribución completa por observación, podés sacar intervalos, cuantiles, o la probabilidad de cualquier evento que te importe — no solo un punto. La incertidumbre que predice **cambia con x**: no asume varianza constante, la aprende de los datos. Es **modular**: la distribución, la scoring rule y el base learner son todos intercambiables, así que un mismo paquete sirve para regresión, clasificación, conteos, o supervivencia con datos censurados. La predicción puntual es competitiva, muy cerca de un Gradient Boosting clásico, tanto en el paper como en nuestra demo. Y es fácil y estable de usar: misma API de scikit-learn, sin necesidad de MCMC ni de definir priors, y el gradiente natural es justamente lo que estabiliza el entrenamiento.

## Slide 14 — Desventajas (≈2 min)

Pero no es gratis. Primero, **hay que elegir bien la familia de distribución**: si no encaja con los datos reales — por ejemplo, si hay asimetría, dos picos, o un tope — el modelo queda mal especificado. De hecho en nuestra propia demo vimos este problema: la Normal le asigna probabilidad a precios de casas negativos, y los precios de California Housing están topeados en 500 mil dólares, algo que ni la Normal ni la LogNormal capturan bien.

Segundo, **es más lento**: entrena p árboles por etapa en vez de uno, necesita muchas iteraciones — en nuestra demo, con learning rate 0.01, hicieron falta unas 1500 — y al correr en Python sobre scikit-learn, queda bastante más lento que XGBoost en C++. Tercero, **no siempre gana**: en algunos datasets, otros métodos probabilísticos lo superan. Y cuarto, si **solo te importa el número puntual**, un boosting clásico bien tuneado rinde igual o mejor, con mucho menos costo computacional — además hay que sumarle que la interpretabilidad es limitada, porque ahora tenés un ensamble de árboles **por cada parámetro**, no uno solo.

El propio paper reconoce líneas de trabajo futuro: mejores árboles y regularización al estilo XGBoost o LightGBM, predicción conjunta de varias variables a la vez, y todavía faltan garantías teóricas rigurosas de convergencia.

## Slide 15 — Conclusión (≈1.5 min)

Si tuviéramos que resumir todo en una sola frase: **NGBoost no predice mejor el número — predice cuánto confiar en él.**

¿Cuándo conviene usarlo? Cuando la decisión depende de un rango o una probabilidad, no solo de un punto; cuando la incertidumbre cambia de un caso a otro, no es uniforme; y cuando los datos son tabulares, que es justamente donde el boosting brilla. ¿Cuándo conviene quedarse con XGBoost? Cuando solo importa el valor puntual, cuando el volumen de datos es enorme, o cuando la velocidad de entrenamiento es crítica.

## Slide 16 — Referencias (≈0.5 min)

Para cerrar, las fuentes: el paper original es Duan, Avati, Ding, Thai, Basu, Ng y Schuler, 2019, "NGBoost: Natural Gradient Boosting for Probabilistic Prediction", disponible en arXiv. La documentación y la librería oficial están en el repo de GitHub de Stanford ML Group. Y nuestro propio notebook, con el ejemplo de juguete, California Housing, calibración y la comparación Normal contra LogNormal, está en nuestro repo de GitHub, linkeado acá.

Muchas gracias, quedamos abiertos a preguntas.
