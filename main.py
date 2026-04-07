import pandas as pd
import networkx as nx
import math
import pickle
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
MAX_JUEGOS_USUARIO = 200   # Umbral para excluir bots/grinders del cálculo
MIN_JUEGOS_COLD_START = 2  # Por debajo de esto, se activa el fallback de popularidad
MIN_POPULARIDAD_JUEGO = 50  # Ignorar juegos con menos de N usuarios (ruido)


# ---------------------------------------------------------------------------
# 1. CARGA Y CONSTRUCCIÓN DEL GRAFO
# ---------------------------------------------------------------------------
def obtener_grafo(ruta_csv: str, nombre_pkl: str = 'grafo_steam_positivo.pkl') -> nx.Graph:
    """
    Construye (o carga desde caché) un grafo bipartito usuario↔juego
    usando únicamente reseñas positivas (voted_up == True).

    Nodos de usuario: bipartite=0
    Nodos de juego:   bipartite=1
    Aristas:          weight=1.0 (afinidad positiva confirmada)
    """
    if os.path.exists(nombre_pkl):
        print(f"[cache] Cargando grafo desde '{nombre_pkl}'...")
        with open(nombre_pkl, 'rb') as f:
            return pickle.load(f)

    print("[build] Procesando CSV y filtrando solo votos POSITIVOS...")
    df = pd.read_csv(ruta_csv, dtype={'author_steamid': str, 'game': str}, low_memory=False)

    df = df.dropna(subset=['author_steamid', 'game', 'voted_up'])
    df_pos = df[df['voted_up'] == True].copy()
    df_clean = df_pos.drop_duplicates(subset=['author_steamid', 'game'])

    B = nx.Graph()
    usuarios = df_clean['author_steamid'].unique()
    juegos   = df_clean['game'].unique()

    B.add_nodes_from(usuarios, bipartite=0)
    B.add_nodes_from(juegos,   bipartite=1)

    print(f"[build] Construyendo red con {len(df_clean):,} conexiones positivas...")
    for _, row in df_clean.iterrows():
        B.add_edge(row['author_steamid'], row['game'], weight=1.0)

    with open(nombre_pkl, 'wb') as f:
        pickle.dump(B, f)

    print(f"[build] Grafo guardado en '{nombre_pkl}'.")
    return B


# ---------------------------------------------------------------------------
# 2. PRECÁLCULO DE POPULARIDAD (OPCIONAL, MEJORA VELOCIDAD)
# ---------------------------------------------------------------------------
def calcular_popularidad(B: nx.Graph) -> dict:
    """Devuelve {juego: num_usuarios} para todos los nodos de juego."""
    return {
        n: B.degree(n)
        for n, d in B.nodes(data=True)
        if d.get('bipartite') == 1
    }


# ---------------------------------------------------------------------------
# 3. MOTOR DE RECOMENDACIÓN MEJORADO
# ---------------------------------------------------------------------------
def recomendar(
    user_id: str,
    B: nx.Graph,
    popularidad: dict | None = None,
    top_n: int = 10,
    max_juegos_usuario: int = MAX_JUEGOS_USUARIO,
    min_juegos_cold_start: int = MIN_JUEGOS_COLD_START,
    min_popularidad_juego: int = MIN_POPULARIDAD_JUEGO,
) -> list[tuple[str, float, int]] | str:
    """
    Recomienda juegos para `user_id` usando un score híbrido:

        score(u, g) = Σ_{v ∈ N(g)} [Jaccard(u, v) × RA(v, g)]
                      ─────────────────────────────────────────
                             log₂(popularidad(g) + 2)

    donde:
      - Jaccard(u, v) = |juegos_u ∩ juegos_v| / |juegos_u ∪ juegos_v|
      - RA(v, g)      = |juegos_u ∩ juegos_v| / |juegos_v|
                        (Resource Allocation: penaliza vecinos con muchos juegos)
      - log₂(pop + 2) penaliza suavemente juegos ultra-populares (más variedad)

    Retorna lista de (juego, score, num_vecinos_contribuyentes) ordenada por score.
    Activa fallback de popularidad si el usuario tiene < min_juegos_cold_start.
    """
    user_id = str(user_id).strip()

    if not B.has_node(user_id):
        return "Usuario no encontrado o no tiene votos positivos registrados."

    if popularidad is None:
        popularidad = calcular_popularidad(B)

    juegos_usuario = set(B.neighbors(user_id))
    n_juegos_usuario = len(juegos_usuario)

    # --- Cold start: usuario con muy pocos juegos ---
    if n_juegos_usuario < min_juegos_cold_start:
        print(f"[cold-start] Usuario con solo {n_juegos_usuario} juego(s). Usando fallback de popularidad.")
        return _fallback_popularidad(juegos_usuario, popularidad, top_n, min_popularidad_juego)

    todos_los_juegos = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 1}
    candidatos = todos_los_juegos - juegos_usuario

    scores: dict[str, float] = defaultdict(float)
    conteo_vecinos: dict[str, int] = defaultdict(int)

    for juego_g in candidatos:
        pop_g = popularidad.get(juego_g, 0)

        # Descartamos juegos con muy poca popularidad (ruido estadístico)
        if pop_g < min_popularidad_juego:
            continue

        for v in B.neighbors(juego_g):
            # Excluimos bots/grinders del cálculo de similitud
            juegos_v = set(B.neighbors(v))
            if len(juegos_v) > max_juegos_usuario:
                continue

            comunes = juegos_usuario.intersection(juegos_v)
            if not comunes:
                continue

            n_comunes = len(comunes)

            # Similitud de Jaccard entre el usuario objetivo y el vecino v
            jaccard = n_comunes / len(juegos_usuario.union(juegos_v))

            # Resource Allocation: cuánto "aporta" v dado que tiene muchos juegos
            ra = n_comunes / len(juegos_v)

            scores[juego_g] += jaccard * ra
            conteo_vecinos[juego_g] += 1

    # Penalización logarítmica de popularidad para favorecer la diversidad
    recomendaciones = []
    for juego_g, sc in scores.items():
        pop_g = popularidad.get(juego_g, 1)
        score_final = sc / math.log2(pop_g + 2)
        recomendaciones.append((juego_g, score_final, conteo_vecinos[juego_g]))

    recomendaciones.sort(key=lambda x: x[1], reverse=True)
    return recomendaciones[:top_n]


# ---------------------------------------------------------------------------
# 4. FALLBACK DE POPULARIDAD (cold start)
# ---------------------------------------------------------------------------
def _fallback_popularidad(
    juegos_vistos: set,
    popularidad: dict,
    top_n: int,
    min_popularidad: int,
) -> list[tuple[str, float, int]]:
    """
    Para usuarios nuevos: devuelve los juegos más populares que no hayan visto.
    El score es simplemente la popularidad normalizada (usuarios/max_usuarios).
    """
    max_pop = max(popularidad.values(), default=1)
    candidatos = [
        (juego, pop / max_pop, 0)
        for juego, pop in popularidad.items()
        if juego not in juegos_vistos and pop >= min_popularidad
    ]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos[:top_n]


# ---------------------------------------------------------------------------
# 5. UTILIDAD: MOSTRAR RESULTADOS
# ---------------------------------------------------------------------------
def mostrar_recomendaciones(user_id: str, resultados) -> None:
    if isinstance(resultados, str):
        print(f"\n[!] {resultados}")
        return

    print(f"\nRecomendaciones para el usuario {user_id}:")
    print(f"{'#':<4} {'Juego':<45} {'Score':>8}  {'Vecinos':>8}")
    print("-" * 70)
    for i, (juego, score, vecinos) in enumerate(resultados, 1):
        etiqueta_vecinos = f"{vecinos}" if vecinos > 0 else "popular"
        print(f"{i:<4} {juego:<45} {score:>8.5f}  {etiqueta_vecinos:>8}")


# ---------------------------------------------------------------------------
# 6. PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    grafo = obtener_grafo('thomas.csv', 'grafo_steam_positivo.pkl')

    # Precalculamos popularidad una sola vez (más eficiente que recalcular por usuario)
    pop = calcular_popularidad(grafo)

    # --- Ejemplo: usuario con historial ---
    user_test = "76561198017550100"
    print(f"\n[>] Generando recomendaciones para {user_test}...")
    resultados = recomendar(user_test, grafo, popularidad=pop, top_n=10)
    mostrar_recomendaciones(user_test, resultados)

    # --- Ejemplo: usuario nuevo (cold start) ---
    user_nuevo = "99999999999999999"
    print(f"\n[>] Probando cold start para {user_nuevo}...")
    resultados_nuevo = recomendar(user_nuevo, grafo, popularidad=pop, top_n=5)
    mostrar_recomendaciones(user_nuevo, resultados_nuevo)