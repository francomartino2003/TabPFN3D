# 08 – Experimentos: latente + positional encoding

Generación de series temporales a partir de:
- Un vector latente **m** (sampleado Normal o Uniform).
- **Positional encoding** PE(t) sumado al latente en cada paso t.
- Proyección lineal **x_t = w·m_t + b** para obtener la serie univariada.

## Uso

```bash
python generate_pe_series.py
```

Editar hiperparámetros al inicio del script o vía CLI.
