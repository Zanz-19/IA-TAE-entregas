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

*(pendiente — se completa al terminar los ejercicios A1-A4)*

</details>
