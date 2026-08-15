# IA-TAE-entregas

Repositorio de entregables para el curso **TAE-IA** (Transformers, instructor Martín Pérez).
Contiene los notebooks trabajados para los dos módulos de la materia, cada uno con su
versión original (tal como la proporcionó el curso) y la versión modificada con las
respuestas y experimentos correspondientes.

## Estructura

```
IA-TAE-entregas/
├── M1-embeddings/
│   ├── original/       # notebook tal como lo entregó el curso
│   └── modificado/      # con la sección RESULTADOS + Conclusiones agregada
└── M2-transformers/
    ├── original/        # notebook tal como lo entregó el curso
    └── modificado/       # con los ejercicios A1-A4 resueltos
```

## Entregables

- **M1 — Seq2Seq (embeddings)**: exploración de configuraciones de entrenamiento sobre un
  traductor español↔inglés (LSTM encoder-decoder), para encontrar la combinación de
  parámetros que mejor mejora el BLEU sobre el baseline. Ver detalle abajo.
- **M2 — Transformers (variantes)**: clasificación de sentimientos con DistilBERT y GPT-2,
  comparando estrategias de readout, profundidad de fine-tuning, pre-entrenamiento vs.
  entrenamiento desde cero, y evaluación cualitativa entre arquitecturas. *(en progreso)*

Ambos notebooks se entregan directamente al instructor por WhatsApp/correo, según lo
solicitado; este repositorio sirve como registro de versiones y del proceso de trabajo.

---

<details>
<summary><b>M1 — Seq2Seq: hallazgos, entendimiento, decisiones y resultados</b></summary>

### Entendimiento del ejercicio

El notebook trae resuelto un comparativo base (secciones C1-C6: atención, alineación,
degradación por longitud, exposure bias, decoding, comparación contra MarianMT) y un
"upgrade kit" con una función `run_cfg(cfg)` que entrena y evalúa una configuración dada.
Se proponen 9 opciones activables (más datos, tokenización subword, embeddings
pre-entrenados, `pack_padded_sequence`, Q/K/V aprendidos, LSTM apilado, validación/early
stopping, dropout, label smoothing, scheduled sampling). El objetivo era explorar estas
opciones y encontrar la mejor combinación, entendiendo qué aportó y qué no.

### Decisiones de diseño del experimento

El notebook ya trae un ejemplo resuelto (`control`, `more-data`, `all-nine`, `tok+emb`)
donde `all-nine` (todas las opciones activas) pierde contra `more-data` sola. En vez de
repetir esa combinación, se decidió **aislar cada una de las 7 opciones restantes** (3-9)
una por una, sobre el mismo presupuesto de `more-data` (24k pares, 4 épocas), para separar
qué opción individual sí ayuda de las que no — y así entender por qué `all-nine` pierde.

### Hallazgos

Ninguna de las 7 opciones ayudó de forma aislada a 4 épocas:

| Config | BLEU | Δ vs more-data |
|---|---|---|
| more-data (baseline) | 14.45 | — |
| iso-pack | 11.69 | -2.27 |
| iso-qkv | 14.02 | +0.07 (con ~196k params extra, no es ganancia limpia) |
| iso-depth2 | 4.65 | -9.80 |
| iso-earlystop | 12.76 | -1.19 |
| iso-dropout | 12.23 | -2.22 |
| iso-labelsmooth | 12.80 | -1.15 |
| iso-schedsamp | 13.00 | -0.95 |

`iso-depth2` fue el caso más extremo (colapso), con un loss de entrenamiento que no
terminó de bajar — señal de underfitting, no de que la técnica sea mala. Se probó la
misma hipótesis en `iso-dropout`, que tenía el `<unk>` más bajo de la tabla pese a perder
en BLEU.

**Prueba de underfitting** (duplicando épocas de 4 a 8, solo para estos dos casos):

| Config | BLEU (4ep) | BLEU (8ep) | Conclusión |
|---|---|---|---|
| depth2 | 4.65 | 13.60 | Mejora mucho, pero nunca alcanza al baseline; 2.75x más lento y +196k params — se descartó |
| dropout | 12.23 | **17.84** | **Supera al baseline** (+3.39), mejor `<unk>` (57.0%), sin costo extra de parámetros |

### Resultado final

**Combinación ganadora: `dropout=0.1` + `epochs=8`** sobre el presupuesto de `more-data`
(24k pares). BLEU **17.84**, +12.23 sobre `control` y +3.39 sobre `more-data` — mejor que
`all-nine` (12.43) y `tok+emb` (10.12) del ejemplo resuelto del curso.

### Conclusiones

Más "opciones" prendidas a la vez no es más regularización gratis: cada una tiene un costo
de convergencia que solo se paga si se le da suficientes épocas. `dropout` y `depth2`
mostraban la misma señal de underfitting a 4 épocas, pero solo uno de los dos valió la
pena mantener incluso con más presupuesto — `depth2` casi triplica el costo computacional
sin superar al baseline, mientras `dropout` lo supera sin costo adicional. La regla
general: antes de descartar una opción por bajo BLEU, hay que revisar si su loss de
entrenamiento terminó de converger.

</details>

<details>
<summary><b>M2 — Transformers: hallazgos, entendimiento, decisiones y resultados</b></summary>

### Entendimiento del ejercicio

El notebook viene con el "Part 0" completo (dataset SST-2, un clasificador con readout
configurable, y `run_grid` para barrer configuraciones) y con los 4 ejercicios (A1-A4)
**ya resueltos con código de ejemplo** — a diferencia de lo que parecía a primera vista,
no hacía falta escribir los sweeps desde cero. Lo que pide la instrucción de Martin, y lo
que realmente se evalúa, es interpretar esos resultados y responder las preguntas
puntuales de cada sección de conclusiones.

### Decisiones de diseño

- Se corrió cada ejercicio tal como venía (sin modificar el código original), agregando
  solo las respuestas de conclusión marcadas con **🟩 RESPUESTA**.
- En A4 sí se extendió el código (en una celda nueva, sin tocar la original) para incluir
  **DistilBERT** en la comparación, tal como sugería explícitamente la instrucción.

### Hallazgos por ejercicio

**A1 — ¿Desde dónde debe leer el head?** DistilBERT (bidireccional) es casi indiferente
al readout (first/last/mean: 0.83/0.83/0.81). GPT-2 (causal) depende muchísimo de la
posición: `first` cae a nivel de azar (0.52) porque el primer token no puede atender a
nada posterior; `last` es el mejor (0.82) porque acumula todo el contexto.

**A2 — ¿Cuánto del modelo hay que reentrenar?** Para DistilBERT, una sola capa
descongelada (`last1`, 0.84) superó incluso a reentrenar todo el modelo (0.81) — con
pocos datos, más fine-tuning puede sobreajustar. GPT-2 necesitó el modelo completo para
llegar a 0.71; descongelar solo 1-2 capas lo empeoró (0.51-0.52) antes de mejorar.

**A3 — ¿De dónde viene la accuracy: pesos o datos?** Preentrenado vs. aleatorio dio una
diferencia de +0.356 en promedio (0.786 vs. 0.430). Triplicar los datos de entrenamiento
(50→300) solo sumó +0.01. El preentrenamiento domina por completo sobre la cantidad de
datos, al menos en este rango.

**A4 — Evaluación cualitativa con oraciones propias.** El modelo "peor" (GPT-2/first,
acc 0.52) resultó ser casi constante en sus predicciones (std de `p_positive` = 0.0016) —
supera el azar por un sesgo fijo, no porque entienda el texto. Al extender la comparación
con DistilBERT/last (acc 0.83, la más alta), se encontró que falla justo en el caso de
negación ("this movie is *not bad* at all" → lo predice como negativo con alta confianza),
mientras que GPT-2/last (menor accuracy global) sí lo acierta — la accuracy global no
cuenta toda la historia.

### Resultado final

No hay una "mejor configuración" única en este lab (a diferencia de M1) — el objetivo era
comparar arquitecturas y estrategias, no optimizar un solo número. El hallazgo transversal
más importante: **la arquitectura (bidireccional vs. causal) determina qué estrategias de
readout y de fine-tuning funcionan**, y una accuracy alta no garantiza buen manejo de
casos difíciles como la negación.

### Conclusiones

Tres ideas se repiten a lo largo de los 4 ejercicios: (1) el diseño del readout debe
respetar la arquitectura — lo que funciona en un encoder bidireccional puede fallar por
completo en un decoder causal; (2) más fine-tuning no siempre es mejor con pocos datos,
el punto óptimo depende de qué tan alineado esté el preentrenamiento con la tarea; y
(3) la accuracy agregada puede esconder comportamientos problemáticos (predicciones
casi constantes, fallas sistemáticas en negación) que solo se detectan revisando
predicciones individuales, no solo el número final.

</details>
