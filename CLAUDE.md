# TP2 Predictivo — Examen 2: Presentación de Tema

## Contexto del proyecto

Trabajo de grupo para el Examen 2 de la materia (ver `consigna.md`): elegir una técnica de análisis predictivo fuera del núcleo de la materia, armar una presentación de 20-30 minutos (idea, ventajas, desventajas, ejemplo de código) y entregar por mail links a un repo/carpeta con la presentación y los notebooks/scripts usados.

**Tema elegido:** NGBoost (Natural Gradient Boosting for Probabilistic Prediction).

## Archivos del proyecto

- `consigna.md` — instrucciones del examen (formales + lo dicho en clase).
- `ngboostpaper.pdf` — paper original (Duan et al., 2019, arXiv:1910.03225).
- `ngboost_repo_resumen.md` — resumen del repo oficial [stanfordmlgroup/ngboost](https://github.com/stanfordmlgroup/ngboost): qué es, instalación, API, notebooks de ejemplo, estructura, licencia/dependencias.
- `resumen_facil_paper.md` — resumen del paper en lenguaje simple ("explicado como a un nene de 7 años"), pensado como pre-lectura antes de meterse con el PDF original.
- `ngboost_demo.ipynb` — notebook de demo (punto b de la consigna): ejemplo de juguete 1D que replica la Figura 1 del paper (banda de incertidumbre que crece con el ruido) + ejemplo real con California Housing comparando NGBoost contra `GradientBoostingRegressor` de scikit-learn (RMSE y NLL) + inspección de los parámetros (μ, σ) que devuelve el modelo.
- `CLAUDE.md` — este archivo: contexto del proyecto + bitácora de qué se hizo y por qué.

## Bitácora

Registro cronológico de decisiones y trabajo hecho, con el motivo. Entradas nuevas abajo de todo.

### 2026-10-05
- Se creó `consigna.md` con el enunciado del examen (pasado por el usuario) para tener las instrucciones a mano dentro del proyecto.
- Se dejó `ngboostpaper.pdf` (paper original de NGBoost) sin convertir a Markdown: tiene fórmulas y figuras que se pierden al pasarlo a texto plano: conviene leerlo directamente como PDF cuando haga falta.
- El usuario compartió el repo de GitHub de NGBoost. Se exploró (README, estructura, notebooks, dependencias) y se volcó el resumen en `ngboost_repo_resumen.md`, para tener una referencia rápida al armar las slides y el notebook de ejemplo.
- Se creó este `CLAUDE.md` para llevar registro de las decisiones y el trabajo del proyecto a medida que avanza.
- Se decidió entender bien el tema antes de armar la notebook: la consigna pide poder explicar la idea (incluso "como a un nene de 7 años") y no solo mostrar código, así que se leyó el paper completo y se generó una explicación simple sección por sección, volcada en `resumen_facil_paper.md` para repasar antes de leer el PDF original.
- Se armó `ngboost_demo.ipynb` con dos demos (juguete 1D + California housing), comparando contra `GradientBoostingRegressor` de sklearn para que se vea claramente la ventaja central del paper: dar una banda de incertidumbre que un boosting clásico no puede dar. Se instaló `ngboost` localmente (no estaba) y se verificó corriendo la notebook de punta a punta — no alcanza con que el código "se vea bien", hay que probarlo.
- Se creó el repo público **https://github.com/andresglau/tp2-predictivo-ngboost** con todo el contenido de esta carpeta, para el link que hay que mandar por mail el día del examen (punto a/b de la consigna).

### 2026-10-13
- El usuario agregó la carpeta `final/` con la presentación definitiva (`NGBoost — Análisis Predictivo.pptx`, 16 slides) y la notebook final (`ngboost.ipynb`).
- Se leyó el contenido completo de las slides (texto + notas de orador ya incluidas en el pptx) para armar `final/speech_presentacion.md`: un guión hablado completo, slide por slide, calibrado a los 20-30 minutos que pide la consigna (incluye el pedido explícito de hablar lento y cubrir menos, y la explicación "para un nene de 7 años" en la slide de la idea central).
