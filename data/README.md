# data/

Aquí va `corpus_indexable.jsonl`, el corpus que produce
[trauma-anonimizacion](https://github.com/hoseStr/trauma-anonimizacion).
Git ignora todo lo que hay en esta carpeta salvo este archivo. Para usar otra
ruta, define la variable de entorno `CORPUS_JSONL`.

Un caso por línea:

```json
{"id": "12345",
 "text": "Motivo de consulta: …\n\nExamen físico: …\n\nEvolución y hospitalización: …",
 "metadata": {"mecanismo_lesion": "arma_de_fuego", "region_anatomica": ["torax"],
              "desenlace": "cirugia", "es_pediatrico": false, "truncado": false,
              "doc_largo": false, "n_chars": 612, "abreviaturas": ["hpaf"]}}
```

`mecanismo_lesion`, `region_anatomica` y `desenlace` alimentan los filtros de
la app y la relevancia de la evaluación.
