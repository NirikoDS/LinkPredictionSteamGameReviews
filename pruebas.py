import pickle
import random
import math
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — ajusta estos valores para que coincidan con tu modelo
# ---------------------------------------------------------------------------
MAX_JUEGOS_USUARIO    = 200
MIN_JUEGOS_COLD_START = 2
MIN_POPULARIDAD_JUEGO = 50

# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------
def cargar_grafo(nombre_archivo: str = 'grafo_steam_positivo.pkl'):
    if not os.path.exists(nombre_archivo):
        print(f"[!] No se encontró '{nombre_archivo}'.")
        return None
    print(f"[cache] Cargando grafo desde '{nombre_archivo}'...")
    with open(nombre_archivo, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# 2. MOTOR DE RECOMENDACIÓN (copia exacta del modelo principal)
# ---------------------------------------------------------------------------
def calcular_popularidad(B) -> dict:
    return {n: B.degree(n) for n, d in B.nodes(data=True) if d.get('bipartite') == 1}


def recomendar(user_id, B, popularidad: dict, top_n: int = 50) -> list:
    user_id = str(user_id).strip()
    if not B.has_node(user_id):
        return []

    juegos_usuario = set(B.neighbors(user_id))
    if len(juegos_usuario) < MIN_JUEGOS_COLD_START:
        return []

    todos_los_juegos = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 1}
    candidatos = todos_los_juegos - juegos_usuario
    scores = defaultdict(float)

    for juego_g in candidatos:
        pop_g = popularidad.get(juego_g, 0)
        if pop_g < MIN_POPULARIDAD_JUEGO:
            continue
        for v in B.neighbors(juego_g):
            juegos_v = set(B.neighbors(v))
            if len(juegos_v) > MAX_JUEGOS_USUARIO:
                continue
            comunes = juegos_usuario & juegos_v
            if not comunes:
                continue
            n_comunes = len(comunes)
            jaccard = n_comunes / len(juegos_usuario | juegos_v)
            ra      = n_comunes / len(juegos_v)
            scores[juego_g] += jaccard * ra

    resultado = [
        (juego_g, sc / math.log2(popularidad.get(juego_g, 1) + 2))
        for juego_g, sc in scores.items()
    ]
    resultado.sort(key=lambda x: x[1], reverse=True)
    return resultado[:top_n]

# ---------------------------------------------------------------------------
# 3. MÉTRICAS
# ---------------------------------------------------------------------------
def hit_rate_at_k(preds: list, relevantes: set, k: int) -> bool:
    return bool({r[0] for r in preds[:k]} & relevantes)

def precision_at_k(preds: list, relevantes: set, k: int) -> float:
    hits = sum(1 for r in preds[:k] if r[0] in relevantes)
    return hits / k if k else 0.0

def ndcg_at_k(preds: list, relevantes: set, k: int) -> float:
    dcg  = sum(1.0 / math.log2(i + 2) for i, r in enumerate(preds[:k]) if r[0] in relevantes)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevantes), k)))
    return dcg / idcg if idcg else 0.0

def reciprocal_rank(preds: list, relevantes: set) -> float:
    for i, (juego, _) in enumerate(preds, 1):
        if juego in relevantes:
            return 1.0 / i
    return 0.0

# ---------------------------------------------------------------------------
# 4. EVALUACIÓN LEAVE-ONE-OUT
# ---------------------------------------------------------------------------
def evaluar(B, n_usuarios: int = 200, k_vals: list = None, semilla: int = 42) -> dict:
    """
    Para cada usuario de la muestra:
      1. Oculta un juego real al azar (remove_edge).
      2. Genera recomendaciones con el grafo reducido.
      3. Comprueba si el juego oculto aparece en el Top-k.
      4. Restaura la arista.

    Devuelve métricas agregadas por cada valor de k.
    """
    if k_vals is None:
        k_vals = [50, 100, 250]

    random.seed(semilla)
    popularidad_global = calcular_popularidad(B)

    usuarios_aptos = [
        n for n, d in B.nodes(data=True)
        if d.get('bipartite') == 0
        and 3 <= B.degree(n) <= MAX_JUEGOS_USUARIO
    ]

    if len(usuarios_aptos) < n_usuarios:
        print(f"[!] Solo {len(usuarios_aptos)} usuarios aptos. Usando todos.")
        n_usuarios = len(usuarios_aptos)

    muestra  = random.sample(usuarios_aptos, n_usuarios)
    max_k    = max(k_vals)
    acum     = {k: {'hits': 0, 'prec': [], 'ndcg': [], 'rr': []} for k in k_vals}

    for i, user in enumerate(muestra, 1):
        juego_oculto = random.choice(list(B.neighbors(user)))

        # Ocultar
        B.remove_edge(user, juego_oculto)
        pop_local = {**popularidad_global, juego_oculto: B.degree(juego_oculto)}

        # Predecir
        preds = recomendar(user, B, pop_local, top_n=max_k)

        # Restaurar
        B.add_edge(user, juego_oculto, weight=1.0)

        relevantes = {juego_oculto}
        for k in k_vals:
            acum[k]['hits'] += int(hit_rate_at_k(preds, relevantes, k))
            acum[k]['prec'].append(precision_at_k(preds, relevantes, k))
            acum[k]['ndcg'].append(ndcg_at_k(preds, relevantes, k))
            acum[k]['rr'].append(reciprocal_rank(preds, relevantes))

        if i % 20 == 0:
            hr = acum[k_vals[0]]['hits'] / i * 100
            print(f"  [{i:>3}/{n_usuarios}]  Hit Rate@{k_vals[0]} parcial: {hr:.1f}%")

    n = len(muestra)
    return {
        k: {
            'hit_rate':  acum[k]['hits'] / n,
            'precision': sum(acum[k]['prec']) / n,
            'ndcg':      sum(acum[k]['ndcg']) / n,
            'mrr':       sum(acum[k]['rr'])   / n,
        }
        for k in k_vals
    }, n

# ---------------------------------------------------------------------------
# 5. REPORTE
# ---------------------------------------------------------------------------
def mostrar_reporte(resultados: dict, n_usuarios: int) -> None:
    ks  = sorted(resultados)
    sep = "=" * (24 + 10 * len(ks))

    print(f"\n{sep}")
    print(f"  {'MÉTRICA':<20}", end="")
    for k in ks:
        print(f"  {'@'+str(k):>7}", end="")
    print()
    print(sep.replace("=", "-"))

    filas = [
        ('hit_rate',  'Hit Rate',  '{:.1%}'),
        ('precision', 'Precision', '{:.4f}'),
        ('ndcg',      'nDCG',      '{:.4f}'),
        ('mrr',       'MRR',       '{:.4f}'),
    ]
    for clave, label, fmt in filas:
        print(f"  {label:<20}", end="")
        for k in ks:
            print(f"  {fmt.format(resultados[k][clave]):>7}", end="")
        print()

    print(sep)
    print(f"  Usuarios evaluados: {n_usuarios}  |  Protocolo: Leave-One-Out")
    print(sep)
    print("""
  Hit Rate@k  → ¿El juego oculto apareció en el Top-k?  (métrica principal)
  Precision@k → Fracción del Top-k que era relevante.
                Con 1 item oculto, el máximo posible es 1/k.
  nDCG@k      → Premia aciertos en posiciones más altas. 1.0 = perfecto.
  MRR         → Recíproco del rango del primer acierto.  0.5 = pos. 2 media.
    """)

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    grafo = cargar_grafo('grafo_steam_positivo.pkl')
    if grafo is None:
        raise SystemExit

    N_USUARIOS = 500        # sube a 500+ para resultados más estables
    K_VALS     = [10, 20, 50]

    print(f"\n[>] Evaluación Leave-One-Out  |  {N_USUARIOS} usuarios  |  k = {K_VALS}")
    resultados, n = evaluar(grafo, n_usuarios=N_USUARIOS, k_vals=K_VALS, semilla=42)
    mostrar_reporte(resultados, n)