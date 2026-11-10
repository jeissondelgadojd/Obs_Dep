# =============================================================================
# OBSERVATORIO INTEGRADO DE INDICADORES SOCIOECONÓMICOS DE COLOMBIA
# FASE 4 – Aplicación interactiva (Streamlit)
#
# Ejecutar:  streamlit run app.py
# Datos:     ./data/panel_observatorio_fase2.parquet
#            ./data/catalogo_indicadores.csv
#            ./data/colombia_departamentos.geojson   (opcional, recomendado)
# =============================================================================
from __future__ import annotations

import io
import json
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Observatorio Socioeconómico de Colombia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Dependencias: obligatorias y opcionales
# -----------------------------------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    st.error("Falta la librería `plotly`. Instálala con `pip install plotly`.")
    st.stop()

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except Exception:  # pragma: no cover
    st.error("Falta la librería `scikit-learn`. Instálala con `pip install scikit-learn`.")
    st.stop()

try:
    import requests
    HAY_REQUESTS = True
except Exception:  # pragma: no cover
    HAY_REQUESTS = False

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    HAY_STATSMODELS = True
except Exception:
    HAY_STATSMODELS = False

try:
    import shap
    HAY_SHAP = True
except Exception:
    HAY_SHAP = False

try:
    from esda.moran import Moran
    from libpysal.weights import W as PysalW
    HAY_PYSAL = True
except Exception:
    HAY_PYSAL = False


# =============================================================================
# 0. CONSTANTES
# =============================================================================
ARCHIVO_PANEL = "panel_observatorio_fase2.parquet"
ARCHIVO_CATALOGO = "catalogo_indicadores.csv"
TOTAL_NACIONAL = "Total nacional"

# GeoJSON departamental (33 polígonos). Se intenta primero un archivo local.
GEOJSON_URL_DEPARTAMENTOS = (
    "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/"
    "3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
)
# URL indicada en la especificación. Contiene solo el contorno del país
# (1 polígono, sin departamentos); se usa como último recurso y se advierte.
GEOJSON_URL_PAIS = "https://raw.githubusercontent.com/mledoze/countries/master/data/col.geo.json"

DIVIPOLA = {
    "Amazonas": "91", "Antioquia": "05", "Arauca": "81", "Atlántico": "08",
    "Bogotá D.C.": "11", "Bolívar": "13", "Boyacá": "15", "Caldas": "17",
    "Caquetá": "18", "Casanare": "85", "Cauca": "19", "Cesar": "20",
    "Chocó": "27", "Córdoba": "23", "Cundinamarca": "25", "Guainía": "94",
    "Guaviare": "95", "Huila": "41", "La Guajira": "44", "Magdalena": "47",
    "Meta": "50", "Nariño": "52", "Norte de Santander": "54", "Putumayo": "86",
    "Quindío": "63", "Risaralda": "66", "Archipiélago de San Andrés": "88",
    "Santander": "68", "Sucre": "70", "Tolima": "73", "Valle del Cauca": "76",
    "Vaupés": "97", "Vichada": "99",
}
CODIGO_A_CANONICO = {v: k for k, v in DIVIPOLA.items()}

# Coordenadas aproximadas (lat, lon) de la capital de cada departamento.
# Respaldo para el mapa de burbujas y los pesos KNN cuando no hay polígonos.
CAPITALES = {
    "Amazonas": (-4.21, -69.94), "Antioquia": (6.25, -75.56), "Arauca": (7.08, -70.76),
    "Atlántico": (10.97, -74.80), "Bogotá D.C.": (4.71, -74.07), "Bolívar": (10.39, -75.48),
    "Boyacá": (5.54, -73.36), "Caldas": (5.07, -75.52), "Caquetá": (1.61, -75.61),
    "Casanare": (5.34, -72.39), "Cauca": (2.44, -76.61), "Cesar": (10.46, -73.25),
    "Chocó": (5.69, -76.66), "Córdoba": (8.75, -75.88), "Cundinamarca": (5.03, -74.03),
    "Guainía": (3.87, -67.92), "Guaviare": (2.57, -72.64), "Huila": (2.93, -75.28),
    "La Guajira": (11.54, -72.91), "Magdalena": (11.24, -74.20), "Meta": (4.14, -73.63),
    "Nariño": (1.21, -77.28), "Norte de Santander": (7.89, -72.51), "Putumayo": (1.15, -76.65),
    "Quindío": (4.53, -75.68), "Risaralda": (4.81, -75.69),
    "Archipiélago de San Andrés": (12.58, -81.70), "Santander": (7.12, -73.12),
    "Sucre": (9.30, -75.40), "Tolima": (4.44, -75.23), "Valle del Cauca": (3.45, -76.53),
    "Vaupés": (1.25, -70.23), "Vichada": (6.19, -67.49),
}

INDICADORES_CLAVE = [
    "PIB_per_capita", "PIB_tasa_crecimiento", "IDC",
    "incidencia_pobreza_monetaria", "coeficiente_gini", "IPM",
    "Tasa de Desocupación (TD)", "Tasa de Ocupación (TO)",
    "Tasa Global de Participación (TGP)",
]
# Indicadores en los que un valor más alto representa una situación peor.
POLARIDAD_NEGATIVA = {
    "incidencia_pobreza_monetaria", "personas_pobreza_monetaria", "coeficiente_gini",
    "IPM", "Tasa de Desocupación (TD)", "Tasa de Subocupación (TS)",
    "Población desocupada", "Población subocupada",
}
DEPARTAMENTOS_REFERENCIA = [
    "Bogotá D.C.", "Antioquia", "Valle del Cauca", "Atlántico",
    "Chocó", "La Guajira", "Vaupés", "Vichada",
]
ETIQUETAS = {
    "PIB_volumen_encadenado": "PIB (volumen encadenado)",
    "PIB_tasa_crecimiento": "Crecimiento del PIB",
    "PIB_contribucion_crecimiento": "Contribución al crecimiento del PIB",
    "PIB_per_capita": "PIB per cápita",
    "IDC": "Índice Departamental de Competitividad",
    "incidencia_pobreza_monetaria": "Pobreza monetaria (incidencia)",
    "personas_pobreza_monetaria": "Personas en pobreza monetaria",
    "coeficiente_gini": "Coeficiente de Gini",
    "IPM": "Pobreza multidimensional (IPM)",
}
ADVERTENCIAS_METODOLOGICAS = [
    "Pobreza monetaria y Gini solo cubren 2021–2025; la serie no es empalmable con años anteriores.",
    "GEIH, pobreza monetaria y Gini cubren 23 departamentos + Bogotá, no los 33 territorios.",
    "PIB en volumen (encadenado, ref. 2015) y PIB per cápita (pesos corrientes) no son intercambiables.",
    "El IDC proviene del Consejo Privado de Competitividad y usa una escala propia (≈0–10).",
    "IPM 2020 y 2021 tienen notas metodológicas específicas; la subocupación 2020 no está disponible (COVID-19).",
    "Las correlaciones y modelos son ecológicos (nivel departamental): describen asociaciones, no causalidad individual.",
]

COLOR_PRINCIPAL = "#1F5F8B"
COLOR_SECUNDARIO = "#D9822B"
COLOR_NEUTRO = "#8A94A6"
PLANTILLA = "plotly_white"
PAGINAS = [
    "1. Inicio y catálogo técnico",
    "2. Explorador, rankings y evolución",
    "3. EDA y PCA (índice sintético)",
    "4. Econometría espacial y mapas",
    "5. Series de tiempo (VAR / VEC)",
    "6. Machine Learning (Ridge · LASSO · SVR)",
    "7. Simulador de política y XAI",
    "8. Ficha departamental y descargas",
]


# =============================================================================
# 1. ESTILO Y UTILIDADES DE INTERFAZ
# =============================================================================
def inyectar_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
        h1, h2, h3 {letter-spacing: -0.01em;}
        .obs-hero {
            padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;
            background: linear-gradient(120deg, #17456B 0%, #1F5F8B 55%, #2E86AB 100%);
            color: #ffffff;
        }
        .obs-hero h1 {color: #ffffff; font-size: 1.75rem; margin: 0 0 .35rem 0; padding: 0;}
        .obs-hero p {color: rgba(255,255,255,.88); margin: 0; font-size: .98rem;}
        .obs-nota {
            border-left: 4px solid #D9822B; background: rgba(217,130,43,.09);
            padding: .65rem .9rem; border-radius: 6px; font-size: .9rem; margin: .4rem 0 1rem 0;
        }
        .obs-chip {
            display: inline-block; padding: .15rem .6rem; border-radius: 999px;
            background: rgba(31,95,139,.12); color: #1F5F8B; font-size: .78rem;
            margin-right: .35rem; font-weight: 600;
        }
        div[data-testid="stMetric"] {
            background: rgba(31,95,139,.06); border: 1px solid rgba(31,95,139,.16);
            border-radius: 12px; padding: .75rem .9rem;
        }
        div[data-testid="stMetricLabel"] p {font-size: .82rem;}
        section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
        .stTabs [data-baseweb="tab"] {font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _version_streamlit() -> tuple:
    try:
        partes = re.findall(r"\d+", st.__version__)[:2]
        return tuple(int(p) for p in partes)
    except Exception:
        return (1, 30)


_ANCHO_NUEVO = _version_streamlit() >= (1, 50)


def mostrar_fig(fig, alto: int | None = None) -> None:
    """Dibuja una figura de Plotly a todo el ancho, compatible entre versiones de Streamlit."""
    fig.update_layout(template=PLANTILLA, margin=dict(l=10, r=10, t=50, b=10))
    if alto:
        fig.update_layout(height=alto)
    if _ANCHO_NUEVO:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def mostrar_tabla(df: pd.DataFrame, alto: int | None = None, indice: bool = False) -> None:
    kwargs = {"hide_index": not indice}
    if alto:
        kwargs["height"] = alto
    if _ANCHO_NUEVO:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


def nota(texto: str) -> None:
    st.markdown(f'<div class="obs-nota">{texto}</div>', unsafe_allow_html=True)


def etiqueta(indicador: str) -> str:
    return ETIQUETAS.get(indicador, indicador)


def fmt_num(v, decimales: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if abs(v) >= 1_000_000:
        return f"{v:,.0f}"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:,.{decimales}f}"


def a_csv(df: pd.DataFrame, indice: bool = False) -> bytes:
    return df.to_csv(index=indice).encode("utf-8-sig")


def norm_txt(txt) -> str:
    """Mayúsculas, sin tildes ni puntuación; para emparejar nombres territoriales."""
    if txt is None:
        return ""
    s = unicodedata.normalize("NFKD", str(txt))
    s = "".join(c for c in s if not unicodedata.combining(c)).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


NORM_A_CANONICO = {norm_txt(k): k for k in DIVIPOLA}


def nombre_a_canonico(txt) -> str | None:
    n = norm_txt(txt)
    if not n:
        return None
    if n in NORM_A_CANONICO:
        return NORM_A_CANONICO[n]
    if "BOGOTA" in n:
        return "Bogotá D.C."
    if "SAN ANDRES" in n:
        return "Archipiélago de San Andrés"
    if "NORTE" in n and "SANTANDER" in n:
        return "Norte de Santander"
    if "VALLE" in n and "CAUCA" in n:
        return "Valle del Cauca"
    if "GUAJIRA" in n:
        return "La Guajira"
    return None


# =============================================================================
# 2. CARGA DE DATOS (cacheada)
# =============================================================================
def _buscar_archivo(nombre: str) -> Path | None:
    base = Path(__file__).resolve().parent
    for carpeta in (base / "data", Path.cwd() / "data", base, Path.cwd(), Path("/data")):
        ruta = carpeta / nombre
        if ruta.exists():
            return ruta
    return None


def _limpiar_panel(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = ["departamento_canonico", "anio", "indicador", "valor"]
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise ValueError(f"El panel no contiene las columnas requeridas: {faltan}")
    df = df.copy()
    for col, defecto in [("desagregacion", "Total"), ("unidad", ""), ("fuente", ""),
                         ("marca_calidad", None), ("codigo_divipola", None)]:
        if col not in df.columns:
            df[col] = defecto
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["departamento_canonico", "indicador", "anio"])
    df["anio"] = df["anio"].astype(int)
    df["desagregacion"] = df["desagregacion"].fillna("Total").astype(str)
    df["indicador"] = df["indicador"].astype(str).str.strip()
    df["departamento_canonico"] = df["departamento_canonico"].astype(str).str.strip()
    df["unidad"] = df["unidad"].fillna("").astype(str)
    df["fuente"] = df["fuente"].fillna("").astype(str)

    def _codigo(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s.zfill(2) if s.isdigit() else s

    df["codigo_divipola"] = df["codigo_divipola"].map(_codigo)
    sin_codigo = df["codigo_divipola"].isna()
    df.loc[sin_codigo, "codigo_divipola"] = df.loc[sin_codigo, "departamento_canonico"].map(DIVIPOLA)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Cargando panel del observatorio…")
def cargar_panel(ruta: str | None, contenido: bytes | None = None) -> pd.DataFrame:
    if contenido is not None:
        df = pd.read_parquet(io.BytesIO(contenido))
    else:
        df = pd.read_parquet(ruta)
    return _limpiar_panel(df)


def _catalogo_desde_panel(panel: pd.DataFrame) -> pd.DataFrame:
    cat = panel.groupby("indicador").agg(
        unidad=("unidad", "first"),
        fuente=("fuente", "first"),
        anio_min=("anio", "min"),
        anio_max=("anio", "max"),
        n_territorios=("departamento_canonico", "nunique"),
    ).reset_index()
    cat["descripcion"] = cat["indicador"].map(etiqueta)
    return cat[["indicador", "descripcion", "unidad", "fuente", "anio_min", "anio_max", "n_territorios"]]


@st.cache_data(show_spinner=False)
def cargar_catalogo(ruta: str | None, panel: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Devuelve (catálogo, es_reconstruido)."""
    base = _catalogo_desde_panel(panel)
    if ruta is None:
        return base, True
    try:
        cat = pd.read_csv(ruta, encoding="utf-8-sig")
        if "indicador" not in cat.columns:
            return base, True
        for col in ["descripcion", "unidad", "fuente", "anio_min", "anio_max", "n_territorios"]:
            if col not in cat.columns:
                cat[col] = cat["indicador"].map(base.set_index("indicador")[col])
        cat["descripcion"] = cat["descripcion"].fillna(cat["indicador"].map(etiqueta))
        return cat, False
    except Exception:
        return base, True


def _iterar_anillos(geometria: dict):
    if not geometria:
        return
    tipo, coords = geometria.get("type"), geometria.get("coordinates", [])
    if tipo == "Polygon":
        for anillo in coords:
            yield anillo
    elif tipo == "MultiPolygon":
        for poligono in coords:
            for anillo in poligono:
                yield anillo


def _leer_geojson_crudo(url_departamental: str) -> tuple[dict | None, str]:
    base = Path(__file__).resolve().parent
    locales = []
    for carpeta in (base / "data", Path.cwd() / "data"):
        if carpeta.exists():
            locales += sorted(carpeta.glob("*.geojson")) + sorted(carpeta.glob("*.geo.json"))
    for ruta in locales:
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                gj = json.load(f)
            if gj.get("features"):
                return gj, f"archivo local `{ruta.name}`"
        except Exception:
            continue
    if HAY_REQUESTS:
        respaldo = None
        for url in [u for u in (url_departamental, GEOJSON_URL_DEPARTAMENTOS, GEOJSON_URL_PAIS) if u]:
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                gj = r.json()
                if not gj.get("features"):
                    continue
                if len(gj["features"]) >= 20:
                    return gj, url
                respaldo = respaldo or (gj, url)
            except Exception:
                continue
        if respaldo:
            return respaldo
    return None, "no disponible"


@st.cache_data(show_spinner="Cargando cartografía…")
def cargar_geo(url_departamental: str) -> dict:
    """Lee el GeoJSON, lo empareja con departamento_canonico y prepara centroides y vecindades.

    Devuelve un dict con: geojson (con `id` por polígono), tabla (id, departamento, lon, lat),
    vecinos (contigüidad por vértices compartidos), fuente y ok (True si es departamental).
    """
    gj, fuente = _leer_geojson_crudo(url_departamental)
    vacio = {"geojson": None, "tabla": pd.DataFrame(columns=["id", "departamento_canonico", "lon", "lat"]),
             "vecinos": {}, "fuente": fuente, "ok": False, "n_features": 0}
    if gj is None:
        return vacio
    feats = gj.get("features", [])
    vacio["n_features"] = len(feats)
    if len(feats) < 20:
        return vacio

    # ¿Existe una propiedad con el código DIVIPOLA?
    conteo_codigo: dict[str, int] = {}
    for f in feats:
        for k, v in (f.get("properties") or {}).items():
            s = str(v).strip()
            if s.endswith(".0"):
                s = s[:-2]
            if s.isdigit() and s.zfill(2) in CODIGO_A_CANONICO:
                conteo_codigo[k] = conteo_codigo.get(k, 0) + 1
    clave_codigo = None
    if conteo_codigo:
        mejor = max(conteo_codigo, key=conteo_codigo.get)
        if conteo_codigo[mejor] >= 0.9 * len(feats):
            clave_codigo = mejor

    filas, vertices = [], {}
    for i, f in enumerate(feats):
        props = f.get("properties") or {}
        canon = None
        for v in props.values():  # 1) por nombre
            if isinstance(v, str):
                canon = nombre_a_canonico(v)
                if canon:
                    break
        if canon is None and clave_codigo is not None:  # 2) por código
            s = str(props.get(clave_codigo, "")).strip()
            s = s[:-2] if s.endswith(".0") else s
            canon = CODIGO_A_CANONICO.get(s.zfill(2))
        f["id"] = str(i)
        anillos = list(_iterar_anillos(f.get("geometry")))
        if canon is None or not anillos:
            continue
        mayor = max(anillos, key=len)
        arr = np.asarray(mayor, dtype=float)[:, :2]
        lon, lat = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        filas.append({"id": str(i), "departamento_canonico": canon, "lon": lon, "lat": lat})
        pts = set()
        for anillo in anillos:
            for p in anillo:
                pts.add((round(float(p[0]), 3), round(float(p[1]), 3)))
        vertices.setdefault(canon, set()).update(pts)

    tabla = pd.DataFrame(filas)
    if tabla.empty or tabla["departamento_canonico"].nunique() < 20:
        return vacio
    nombres = sorted(vertices)
    vecinos = {n: [] for n in nombres}
    for a_i, a in enumerate(nombres):
        for b in nombres[a_i + 1:]:
            if not vertices[a].isdisjoint(vertices[b]):
                vecinos[a].append(b)
                vecinos[b].append(a)
    return {"geojson": gj, "tabla": tabla, "vecinos": vecinos, "fuente": fuente,
            "ok": True, "n_features": len(feats)}


# =============================================================================
# 3. UTILIDADES ANALÍTICAS
# =============================================================================
def anios_de(df: pd.DataFrame, indicador: str, desag: str = "Total") -> list[int]:
    sub = df[(df["indicador"] == indicador) & (df["desagregacion"] == desag)].dropna(subset=["valor"])
    return sorted(sub["anio"].unique().tolist(), reverse=True)


def desagregaciones_de(df: pd.DataFrame, indicador: str) -> list[str]:
    d = sorted(df.loc[df["indicador"] == indicador, "desagregacion"].dropna().unique().tolist())
    return (["Total"] + [x for x in d if x != "Total"]) if "Total" in d else d


def unidad_de(catalogo: pd.DataFrame, panel: pd.DataFrame, indicador: str) -> str:
    fila = catalogo.loc[catalogo["indicador"] == indicador, "unidad"]
    if len(fila) and isinstance(fila.iloc[0], str) and fila.iloc[0]:
        return fila.iloc[0]
    u = panel.loc[panel["indicador"] == indicador, "unidad"]
    return u.iloc[0] if len(u) else ""


def matriz_ancha(df: pd.DataFrame, anio: int, indicadores: list[str], desag: str = "Total",
                 ultimo_disponible: bool = False) -> pd.DataFrame:
    """Pivotea a departamento × indicador para un año.

    Con `ultimo_disponible=True` cada celda toma el dato más reciente con año ≤ `anio`.
    """
    sub = df[df["indicador"].isin(indicadores) & (df["desagregacion"] == desag)].dropna(subset=["valor"])
    if ultimo_disponible:
        sub = sub[sub["anio"] <= anio].sort_values("anio")
        ancho = sub.groupby(["departamento_canonico", "indicador"])["valor"].last().unstack("indicador")
    else:
        sub = sub[sub["anio"] == anio]
        ancho = sub.pivot_table(index="departamento_canonico", columns="indicador", values="valor", aggfunc="mean")
    return ancho.reindex(columns=[i for i in indicadores if i in ancho.columns])


def imputar_media(ancho: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Imputa nulos con la media de cada columna. Devuelve (matriz, nulos_por_columna)."""
    nulos = ancho.isna().sum()
    return ancho.fillna(ancho.mean(numeric_only=True)), nulos[nulos > 0]


def serie_ultimo_anio(df: pd.DataFrame, indicador: str, desag: str = "Total"):
    sub = df[(df["indicador"] == indicador) & (df["desagregacion"] == desag)].dropna(subset=["valor"])
    if sub.empty:
        return None, None
    anio = int(sub["anio"].max())
    return sub[sub["anio"] == anio].copy(), anio


def panel_ancho_ml(deptos: pd.DataFrame, indicadores: list[str]) -> pd.DataFrame:
    """Departamento-año × indicador (desagregación Total)."""
    sub = deptos[deptos["indicador"].isin(indicadores) & (deptos["desagregacion"] == "Total")]
    sub = sub.dropna(subset=["valor"])
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index=["departamento_canonico", "anio"], columns="indicador",
                           values="valor", aggfunc="mean").sort_index()


# ---------------------------- PCA --------------------------------------------
@st.cache_resource(show_spinner="Estimando componentes principales…")
def ajustar_pca(matriz: pd.DataFrame, negativos: tuple, n_componentes: int) -> dict:
    """PCA sobre variables estandarizadas y alineadas (mayor = mejor)."""
    cols = list(matriz.columns)
    signos = np.array([-1.0 if c in negativos else 1.0 for c in cols])
    Z = StandardScaler().fit_transform(matriz.values) * signos
    pca = PCA(n_components=n_componentes, random_state=42).fit(Z)
    cargas = pca.components_.copy()
    puntajes = pca.transform(Z)
    if cargas[0].sum() < 0:  # orientar PC1: valores altos = mejor situación relativa
        cargas[0] *= -1
        puntajes[:, 0] *= -1
    pc1 = puntajes[:, 0]
    rango = pc1.max() - pc1.min()
    indice = (pc1 - pc1.min()) / rango * 100 if rango > 0 else np.full_like(pc1, 50.0)
    nombres_pc = [f"PC{i + 1}" for i in range(n_componentes)]
    return {
        "varianza": pca.explained_variance_ratio_,
        "cargas": pd.DataFrame(cargas.T, index=cols, columns=nombres_pc),
        "puntajes": pd.DataFrame(puntajes, index=matriz.index, columns=nombres_pc),
        "indice": pd.Series(indice, index=matriz.index, name="indice_sintetico"),
    }


# ---------------------------- Moran ------------------------------------------
def pesos_knn(coords: np.ndarray, nombres: list[str], k: int) -> dict:
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(dist, np.inf)
    k = max(1, min(k, len(nombres) - 1))
    orden = np.argsort(dist, axis=1)[:, :k]
    return {nombres[i]: [nombres[j] for j in orden[i]] for i in range(len(nombres))}


def moran_global(y: np.ndarray, vecinos: dict, nombres: list[str], permutaciones: int = 999) -> dict:
    """I de Moran global con pesos estandarizados por fila. Usa esda/libpysal si están instalados."""
    n = len(nombres)
    pos = {nm: i for i, nm in enumerate(nombres)}
    Wm = np.zeros((n, n))
    for nm, vs in vecinos.items():
        for v in vs:
            if nm in pos and v in pos and nm != v:
                Wm[pos[nm], pos[v]] = 1.0
    filas = Wm.sum(axis=1, keepdims=True)
    Wm = np.divide(Wm, filas, out=np.zeros_like(Wm), where=filas > 0)
    z = y - y.mean()
    rezago = Wm @ z
    resultado = {"rezago": rezago, "z": z, "motor": "numpy"}

    if HAY_PYSAL:
        try:
            w = PysalW({nm: [v for v in vecinos.get(nm, []) if v in pos and v != nm] for nm in nombres},
                       id_order=nombres, silence_warnings=True)
            w.transform = "r"
            mi = Moran(y, w, permutations=permutaciones)
            resultado.update({"I": float(mi.I), "EI": float(mi.EI), "z_score": float(mi.z_sim),
                              "p_valor": float(mi.p_sim), "motor": "esda + libpysal"})
            return resultado
        except Exception:
            pass

    s0 = Wm.sum()
    denom = float(z @ z)
    if s0 == 0 or denom == 0:
        resultado.update({"I": np.nan, "EI": -1 / (n - 1), "z_score": np.nan, "p_valor": np.nan})
        return resultado
    I = (n / s0) * float(z @ Wm @ z) / denom
    rng = np.random.default_rng(42)
    sims = np.empty(permutaciones)
    for b in range(permutaciones):
        zp = rng.permutation(z)
        sims[b] = (n / s0) * float(zp @ Wm @ zp) / denom
    arriba = int((sims >= I).sum())
    mayor = min(arriba, permutaciones - arriba)
    sd = sims.std()
    resultado.update({"I": I, "EI": -1 / (n - 1), "p_valor": (mayor + 1) / (permutaciones + 1),
                      "z_score": (I - sims.mean()) / sd if sd > 0 else np.nan})
    return resultado


# ---------------------------- Series de tiempo --------------------------------
def construir_series(panel: pd.DataFrame, indicadores: list[str], territorio: str, agregacion: str) -> pd.DataFrame:
    sub = panel[panel["indicador"].isin(indicadores) & (panel["desagregacion"] == "Total")].dropna(subset=["valor"])
    if agregacion == "territorio":
        sub = sub[sub["departamento_canonico"] == territorio]
        ancho = sub.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="mean")
    else:
        sub = sub[sub["departamento_canonico"] != TOTAL_NACIONAL]
        ancho = sub.pivot_table(index="anio", columns="indicador", values="valor",
                                aggfunc="sum" if agregacion == "suma" else "mean")
    return ancho.reindex(columns=[i for i in indicadores if i in ancho.columns]).sort_index()


@st.cache_resource(show_spinner="Estimando VAR…")
def estimar_var(datos: pd.DataFrame, rezagos: int):
    return VAR(datos.reset_index(drop=True)).fit(rezagos)


@st.cache_resource(show_spinner="Estimando VEC…")
def estimar_vec(datos: pd.DataFrame, rezagos_dif: int, rango: int):
    return VECM(datos.reset_index(drop=True), k_ar_diff=rezagos_dif, coint_rank=rango, deterministic="co").fit()


# ---------------------------- Machine Learning --------------------------------
HP_DEFECTO = (1.0, 0.05, 10.0, 0.1, "rbf")  # alpha Ridge, alpha LASSO, C, epsilon, kernel SVR


def _crear_modelos(hp: tuple) -> dict:
    def envolver(regresor):
        tuberia = Pipeline([
            ("imputador", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("escalador", StandardScaler()),
            ("modelo", regresor),
        ])
        return TransformedTargetRegressor(regressor=tuberia, transformer=StandardScaler())

    return {
        "Ridge": envolver(Ridge(alpha=hp[0])),
        "LASSO": envolver(Lasso(alpha=hp[1], max_iter=50000)),
        "SVR": envolver(SVR(C=hp[2], epsilon=hp[3], kernel=hp[4])),
    }


@st.cache_resource(show_spinner="Entrenando y validando modelos…")
def entrenar_modelos(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, hp: tuple, esquema: str) -> dict:
    """Entrena Ridge, LASSO y SVR; devuelve modelos ajustados, métricas de CV y predicciones fuera de muestra."""
    n_grupos = grupos.nunique()
    if esquema == "grupos" and n_grupos >= 3:
        cv = GroupKFold(n_splits=min(5, n_grupos))
        kw = {"groups": grupos.values}
    else:
        cv = KFold(n_splits=max(2, min(5, len(X) // 4)), shuffle=True, random_state=42)
        kw = {}
    puntuaciones = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    filas, modelos, preds = [], {}, {}
    for nombre, modelo in _crear_modelos(hp).items():
        res = cross_validate(modelo, X, y, cv=cv, scoring=puntuaciones, **kw)
        pred = cross_val_predict(modelo, X, y, cv=cv, **kw)
        sc_res = float(((y.values - pred) ** 2).sum())
        sc_tot = float(((y.values - y.values.mean()) ** 2).sum())
        filas.append({
            "modelo": nombre,
            "R² (media CV)": res["test_r2"].mean(),
            "R² (desv.)": res["test_r2"].std(),
            "R² fuera de muestra (agrupado)": 1 - sc_res / sc_tot if sc_tot > 0 else np.nan,
            "RMSE": -res["test_rmse"].mean(),
            "RMSE (desv.)": res["test_rmse"].std(),
            "MAE": -res["test_mae"].mean(),
        })
        modelos[nombre] = modelo.fit(X, y)
        preds[nombre] = pred
    return {"modelos": modelos, "metricas": pd.DataFrame(filas), "predicciones": preds,
            "n_pliegues": cv.get_n_splits(), "esquema": "GroupKFold por departamento" if kw else "KFold aleatorio"}


@st.cache_resource(show_spinner="Calculando valores SHAP…")
def calcular_shap(_modelo, firma: str, X_fondo: pd.DataFrame, X_eval: pd.DataFrame) -> dict:
    """Valores SHAP (model-agnostic). `firma` identifica el modelo para la caché."""
    cols = list(X_fondo.columns)

    def f(A):
        return _modelo.predict(pd.DataFrame(np.asarray(A), columns=cols))

    explicador = shap.Explainer(f, X_fondo)
    sv = explicador(X_eval)
    base = np.asarray(sv.base_values).reshape(-1)
    return {"valores": np.asarray(sv.values), "base": float(base.mean())}


def contribuciones_aproximadas(modelo, X_fondo: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    """Respaldo sin SHAP: efecto de llevar cada variable desde la media hasta su valor (ceteris paribus)."""
    medias = X_fondo.mean()
    base = pd.DataFrame([medias.values], columns=X_fondo.columns)
    f_base = float(modelo.predict(base)[0])
    salida = []
    for c in X_fondo.columns:
        tmp = base.copy()
        tmp[c] = float(x[c].iloc[0])
        salida.append(float(modelo.predict(tmp)[0]) - f_base)
    return np.array(salida)


def selector_ml(deptos: pd.DataFrame, indicadores_total: list[str], clave: str):
    """Controles compartidos (objetivo y variables) entre los módulos 6 y 7."""
    c1, c2 = st.columns([1, 2])
    obj_defecto = st.session_state.get("ml_objetivo", "IPM" if "IPM" in indicadores_total else indicadores_total[0])
    if obj_defecto not in indicadores_total:
        obj_defecto = indicadores_total[0]
    objetivo = c1.selectbox("Variable objetivo", indicadores_total, index=indicadores_total.index(obj_defecto),
                            format_func=etiqueta, key=f"{clave}_obj")
    candidatas = [i for i in indicadores_total if i != objetivo]
    defecto = [i for i in st.session_state.get("ml_features", []) if i in candidatas]
    if not defecto:
        defecto = [i for i in ["PIB_per_capita", "IDC", "PIB_tasa_crecimiento", "Tasa de Desocupación (TD)",
                               "Tasa de Ocupación (TO)", "Tasa Global de Participación (TGP)"] if i in candidatas]
    if not defecto:
        defecto = candidatas[:4]
    features = c2.multiselect("Variables explicativas (insumos de política)", candidatas, default=defecto,
                              format_func=etiqueta, key=f"{clave}_feat")
    st.session_state["ml_objetivo"], st.session_state["ml_features"] = objetivo, features
    return objetivo, features


def preparar_dataset_ml(deptos: pd.DataFrame, objetivo: str, features: list[str]):
    """Devuelve (X, y, grupos, ancho_completo, features_validas) o None con advertencia."""
    if not features:
        st.warning("Selecciona al menos una variable explicativa.")
        return None
    ancho = panel_ancho_ml(deptos, [objetivo] + features)
    if ancho.empty or objetivo not in ancho.columns:
        st.warning("No hay datos de la variable objetivo en desagregación «Total».")
        return None
    muestra = ancho.dropna(subset=[objetivo])
    validas = [f for f in features if f in muestra.columns and muestra[f].notna().sum() > 0]
    descartadas = [etiqueta(f) for f in features if f not in validas]
    if descartadas:
        st.warning("Sin traslape temporal/territorial con el objetivo, se descartan: " + ", ".join(descartadas) + ".")
    if not validas:
        st.warning("Ninguna variable explicativa coincide en años y departamentos con el objetivo.")
        return None
    muestra = muestra[muestra[validas].notna().any(axis=1)]
    if len(muestra) < 15:
        st.warning(f"Solo hay {len(muestra)} observaciones departamento-año; se requieren al menos 15 para validar.")
        return None
    pct = (muestra[validas].isna().mean() * 100).round(1)
    if (pct > 0).any():
        detalle = "; ".join(f"{etiqueta(k)}: {v}%" for k, v in pct[pct > 0].items())
        st.warning("Valores faltantes imputados con la media dentro de cada pliegue de validación → " + detalle)
    X = muestra[validas].reset_index(drop=True)
    y = muestra[objetivo].reset_index(drop=True)
    grupos = pd.Series(muestra.index.get_level_values("departamento_canonico")).reset_index(drop=True)
    return X, y, grupos, ancho, validas


# =============================================================================
# 4. MÓDULOS (PÁGINAS)
# =============================================================================
def pagina_inicio(panel, deptos, catalogo, catalogo_reconstruido):
    st.markdown(
        """
        <div class="obs-hero">
          <h1>Observatorio Integrado de Indicadores Socioeconómicos de Colombia</h1>
          <p>Fase 4 · Panel departamental integrado (DANE · Consejo Privado de Competitividad) con
          exploración, econometría espacial y temporal, aprendizaje automático y simulación de política.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observaciones", f"{len(panel):,}")
    c2.metric("Indicadores", f"{panel['indicador'].nunique()}")
    c3.metric("Territorios", f"{deptos['departamento_canonico'].nunique()}")
    c4.metric("Periodo", f"{int(panel['anio'].min())}–{int(panel['anio'].max())}")
    c5.metric("Fuentes", f"{panel.loc[panel['fuente'] != '', 'fuente'].nunique()}")

    tab_cat, tab_cob, tab_cal = st.tabs(["Catálogo de indicadores", "Cobertura del panel", "Calidad y advertencias"])

    with tab_cat:
        if catalogo_reconstruido:
            st.warning("No se encontró `catalogo_indicadores.csv`; el catálogo se reconstruyó desde el panel.")
        f1, f2 = st.columns([2, 1])
        texto = f1.text_input("Buscar indicador o descripción", "")
        fuentes = sorted(catalogo["fuente"].dropna().astype(str).unique().tolist())
        sel_fuentes = f2.multiselect("Fuente", fuentes, default=[])
        vista = catalogo.copy()
        if texto:
            patron = norm_txt(texto)
            mascara = vista.apply(lambda r: patron in norm_txt(f"{r['indicador']} {r.get('descripcion', '')}"), axis=1)
            vista = vista[mascara]
        if sel_fuentes:
            vista = vista[vista["fuente"].astype(str).isin(sel_fuentes)]
        columnas = [c for c in ["indicador", "descripcion", "unidad", "fuente", "anio_min", "anio_max",
                                "n_territorios"] if c in vista.columns]
        mostrar_tabla(vista[columnas], alto=460)
        st.download_button("Descargar catálogo (CSV)", a_csv(vista[columnas]), "catalogo_indicadores.csv", "text/csv")

    with tab_cob:
        cobertura = (deptos.dropna(subset=["valor"]).groupby(["indicador", "anio"])["departamento_canonico"]
                     .nunique().unstack("anio").sort_index())
        if cobertura.empty:
            st.warning("El panel no tiene valores válidos para calcular cobertura.")
        else:
            fig = px.imshow(cobertura, aspect="auto", color_continuous_scale="Blues", text_auto=True,
                            labels=dict(x="Año", y="", color="Departamentos"))
            fig.update_layout(title="Número de departamentos con dato por indicador y año")
            fig.update_xaxes(type="category")
            mostrar_fig(fig, alto=max(420, 26 * len(cobertura)))
        por_fuente = (panel.groupby("fuente").agg(indicadores=("indicador", "nunique"),
                                                 territorios=("departamento_canonico", "nunique"),
                                                 observaciones=("valor", "count"),
                                                 anio_min=("anio", "min"), anio_max=("anio", "max")).reset_index())
        st.markdown("**Cobertura por fuente**")
        mostrar_tabla(por_fuente)

    with tab_cal:
        for a in ADVERTENCIAS_METODOLOGICAS:
            st.markdown(f"- {a}")
        marcas = panel.dropna(subset=["marca_calidad"])
        if marcas.empty:
            st.info("El panel no registra marcas de calidad.")
        else:
            resumen = (marcas.groupby(["marca_calidad", "indicador"]).agg(
                observaciones=("valor", "size"), anio_min=("anio", "min"), anio_max=("anio", "max")).reset_index())
            st.markdown("**Marcas de calidad registradas en el panel**")
            mostrar_tabla(resumen, alto=360)
        nulos = int(panel["valor"].isna().sum())
        st.caption(f"Valores nulos en el panel: {nulos:,} de {len(panel):,} filas ({nulos / max(len(panel), 1):.1%}).")


def pagina_explorador(panel, deptos, catalogo):
    st.header("Explorador, rankings y evolución temporal")
    indicadores = sorted(panel["indicador"].unique().tolist())
    tab_rank, tab_evo, tab_brecha = st.tabs(["Ranking departamental", "Evolución temporal", "Brechas del IPM"])

    with tab_rank:
        c1, c2, c3, c4 = st.columns([2, 1.3, 1, 1.2])
        defecto = indicadores.index("IPM") if "IPM" in indicadores else 0
        ind = c1.selectbox("Indicador", indicadores, index=defecto, format_func=etiqueta, key="rk_ind")
        desag = c2.selectbox("Desagregación", desagregaciones_de(deptos, ind), key="rk_des")
        anios = anios_de(deptos, ind, desag)
        if not anios:
            st.warning("No hay datos departamentales para esta combinación de indicador y desagregación.")
        else:
            anio = c3.selectbox("Año", anios, key="rk_anio")
            orden = c4.radio("Orden", ["Mayor a menor", "Menor a mayor"], horizontal=True, key="rk_ord")
            datos = deptos[(deptos["indicador"] == ind) & (deptos["desagregacion"] == desag) &
                           (deptos["anio"] == anio)].dropna(subset=["valor"])
            datos = datos.groupby("departamento_canonico", as_index=False)["valor"].mean()
            datos = datos.sort_values("valor", ascending=(orden == "Menor a mayor")).reset_index(drop=True)
            datos["puesto"] = np.arange(1, len(datos) + 1)
            tope = st.slider("Departamentos a mostrar", 5, max(5, len(datos)), max(5, len(datos)), key="rk_top")
            unidad = unidad_de(catalogo, panel, ind)
            vista = datos.head(tope)
            fig = px.bar(vista.iloc[::-1], x="valor", y="departamento_canonico", orientation="h",
                         color_discrete_sequence=[COLOR_PRINCIPAL], labels={"valor": unidad or "valor",
                                                                           "departamento_canonico": ""})
            nacional = panel[(panel["indicador"] == ind) & (panel["desagregacion"] == desag) &
                             (panel["anio"] == anio) & (panel["departamento_canonico"] == TOTAL_NACIONAL)]["valor"].dropna()
            if len(nacional):
                fig.add_vline(x=float(nacional.iloc[0]), line_dash="dash", line_color=COLOR_SECUNDARIO,
                              annotation_text="Total nacional", annotation_position="top")
            fig.update_layout(title=f"{etiqueta(ind)} · {desag} · {anio}")
            mostrar_fig(fig, alto=max(420, 22 * len(vista) + 120))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Departamentos con dato", len(datos))
            m2.metric("Media", fmt_num(datos["valor"].mean()))
            m3.metric("Mediana", fmt_num(datos["valor"].median()))
            m4.metric("Rango (máx − mín)", fmt_num(datos["valor"].max() - datos["valor"].min()))
            if ind in POLARIDAD_NEGATIVA:
                st.caption("En este indicador un valor más alto representa una situación más desfavorable.")
            with st.expander("Ver tabla del ranking"):
                mostrar_tabla(datos[["puesto", "departamento_canonico", "valor"]])
                st.download_button("Descargar ranking (CSV)", a_csv(datos), f"ranking_{ind}_{anio}.csv", "text/csv")

    with tab_evo:
        c1, c2 = st.columns([2, 1.3])
        defecto = indicadores.index("PIB_per_capita") if "PIB_per_capita" in indicadores else 0
        ind = c1.selectbox("Indicador", indicadores, index=defecto, format_func=etiqueta, key="ev_ind")
        desag = c2.selectbox("Desagregación", desagregaciones_de(panel, ind), key="ev_des")
        base = panel[(panel["indicador"] == ind) & (panel["desagregacion"] == desag)].dropna(subset=["valor"])
        territorios = sorted(base["departamento_canonico"].unique().tolist())
        if not territorios:
            st.warning("No hay datos para esta combinación.")
        else:
            defecto_t = [d for d in DEPARTAMENTOS_REFERENCIA if d in territorios][:6] or territorios[:5]
            if TOTAL_NACIONAL in territorios:
                defecto_t = [TOTAL_NACIONAL] + defecto_t
            sel = st.multiselect("Territorios", territorios, default=defecto_t, key="ev_ter")
            base100 = st.checkbox("Expresar como índice (primer año disponible = 100)", value=False, key="ev_b100")
            serie = base[base["departamento_canonico"].isin(sel)]
            serie = serie.groupby(["departamento_canonico", "anio"], as_index=False)["valor"].mean().sort_values("anio")
            if serie.empty:
                st.warning("Selecciona al menos un territorio con datos.")
            else:
                etiqueta_y = unidad_de(catalogo, panel, ind) or "valor"
                if base100:
                    primero = serie.groupby("departamento_canonico")["valor"].transform("first")
                    if (primero == 0).any():
                        st.warning("Algún territorio tiene valor inicial 0; se omite del índice base 100.")
                    serie = serie[primero != 0].copy()
                    serie["valor"] = serie["valor"] / primero[primero != 0] * 100
                    etiqueta_y = "Índice (base 100)"
                fig = px.line(serie, x="anio", y="valor", color="departamento_canonico", markers=True,
                              labels={"anio": "Año", "valor": etiqueta_y, "departamento_canonico": "Territorio"})
                fig.update_layout(title=f"{etiqueta(ind)} · {desag}", hovermode="x unified")
                fig.update_xaxes(dtick=1 if serie["anio"].nunique() <= 12 else 2)
                mostrar_fig(fig, alto=500)
                marcas = panel[(panel["indicador"] == ind)].dropna(subset=["marca_calidad"])
                if not marcas.empty:
                    lista = ", ".join(sorted(marcas["marca_calidad"].astype(str).unique()))
                    st.caption(f"Marcas de calidad presentes en esta serie: {lista}.")

    with tab_brecha:
        ipm = deptos[deptos["indicador"] == "IPM"].dropna(subset=["valor"])
        if ipm.empty:
            st.warning("El panel no contiene el indicador IPM; no es posible calcular brechas.")
            return
        tipos = {
            "Urbano–rural (rural − cabeceras)": ("Cabeceras", "Centros poblados y rural disperso"),
            "Por sexo (mujer − hombre)": ("Hombre", "Mujer"),
        }
        disponibles = {k: v for k, v in tipos.items() if set(v) <= set(ipm["desagregacion"].unique())}
        if not disponibles:
            st.warning("El IPM no incluye desagregaciones urbano–rural ni por sexo en este panel.")
            return
        c1, c2 = st.columns([2, 1])
        tipo = c1.selectbox("Brecha", list(disponibles), key="br_tipo")
        ref, comp = disponibles[tipo]
        anio = c2.selectbox("Año", sorted(ipm["anio"].unique().tolist(), reverse=True), key="br_anio")
        tabla = ipm[ipm["anio"] == anio].pivot_table(index="departamento_canonico", columns="desagregacion",
                                                    values="valor", aggfunc="mean")
        if ref not in tabla.columns or comp not in tabla.columns:
            st.warning("No hay datos de ambas desagregaciones para el año seleccionado.")
            return
        tabla = tabla[[ref, comp]].dropna()
        if tabla.empty:
            st.warning("Ningún departamento tiene ambas desagregaciones en ese año.")
            return
        tabla["brecha"] = tabla[comp] - tabla[ref]
        tabla = tabla.sort_values("brecha")
        xs, ys = [], []
        for dep, fila in tabla.iterrows():
            xs += [fila[ref], fila[comp], None]
            ys += [dep, dep, None]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#C5CCD6", width=3), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tabla[ref], y=tabla.index, mode="markers", name=ref,
                                 marker=dict(color=COLOR_PRINCIPAL, size=10)))
        fig.add_trace(go.Scatter(x=tabla[comp], y=tabla.index, mode="markers", name=comp,
                                 marker=dict(color=COLOR_SECUNDARIO, size=10)))
        fig.update_layout(title=f"IPM {anio}: {ref} frente a {comp}", xaxis_title="IPM (%)",
                          legend=dict(orientation="h", y=1.06))
        mostrar_fig(fig, alto=max(460, 22 * len(tabla) + 120))
        m1, m2, m3 = st.columns(3)
        m1.metric("Brecha media (p.p.)", fmt_num(tabla["brecha"].mean(), 1))
        m2.metric("Mayor brecha", f"{tabla['brecha'].idxmax()}", fmt_num(tabla["brecha"].max(), 1) + " p.p.",
                  delta_color="off")
        m3.metric("Menor brecha", f"{tabla['brecha'].idxmin()}", fmt_num(tabla["brecha"].min(), 1) + " p.p.",
                  delta_color="off")
        with st.expander("Ver tabla de brechas"):
            mostrar_tabla(tabla.round(2).reset_index())
            st.download_button("Descargar brechas (CSV)", a_csv(tabla.reset_index()), f"brecha_ipm_{anio}.csv", "text/csv")


def _controles_transversales(deptos, clave: str):
    """Controles comunes de corte transversal (año, modo e indicadores) para EDA y PCA."""
    totales = deptos[deptos["desagregacion"] == "Total"].dropna(subset=["valor"])
    indicadores = sorted(totales["indicador"].unique().tolist())
    anios = sorted(totales["anio"].unique().tolist(), reverse=True)
    if not indicadores or not anios:
        st.warning("El panel no tiene datos en desagregación «Total».")
        return None
    c1, c2 = st.columns([1, 2])
    anio_defecto = 2023 if 2023 in anios else anios[0]
    anio = c1.selectbox("Año de análisis", anios, index=anios.index(anio_defecto), key=f"{clave}_anio")
    modo = c2.radio("Datos por indicador", ["Solo el año seleccionado", "Último dato disponible hasta ese año"],
                    horizontal=True, key=f"{clave}_modo")
    defecto = [i for i in INDICADORES_CLAVE if i in indicadores] or indicadores[:6]
    sel = st.multiselect("Indicadores", indicadores, default=defecto, format_func=etiqueta, key=f"{clave}_inds")
    if len(sel) < 2:
        st.warning("Selecciona al menos dos indicadores.")
        return None
    ancho = matriz_ancha(deptos, anio, sel, "Total", ultimo_disponible=modo.startswith("Último"))
    faltantes = [etiqueta(i) for i in sel if i not in ancho.columns]
    if faltantes:
        st.warning(f"Sin datos en {anio} (se excluyen): " + ", ".join(faltantes) + ".")
    if ancho.shape[1] < 2 or ancho.shape[0] < 3:
        st.warning("No hay suficientes indicadores o departamentos con datos para este corte. "
                   "Prueba otro año o el modo «Último dato disponible».")
        return None
    return anio, ancho


def pagina_eda_pca(deptos):
    st.header("Análisis exploratorio y reducción de dimensionalidad")
    tab_corr, tab_pca = st.tabs(["Correlaciones dinámicas", "PCA e índice sintético (0–100)"])

    with tab_corr:
        res = _controles_transversales(deptos, "corr")
        if res is not None:
            anio, ancho = res
            c1, c2 = st.columns([1, 1])
            metodo = c1.radio("Método", ["pearson", "spearman"], horizontal=True, format_func=str.capitalize,
                              key="corr_met")
            min_obs = c2.slider("Mínimo de departamentos por par", 3, max(3, len(ancho)), min(8, len(ancho)),
                                key="corr_min")
            corr = ancho.corr(method=metodo, min_periods=min_obs)
            if corr.isna().all().all():
                st.warning("Ningún par de indicadores alcanza el mínimo de departamentos en común.")
            else:
                nombres = [etiqueta(c) for c in corr.columns]
                fig = px.imshow(corr.values, x=nombres, y=nombres, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                text_auto=".2f", aspect="auto")
                fig.update_layout(title=f"Matriz de correlaciones ({metodo.capitalize()}) · {anio} · "
                                        f"{len(ancho)} departamentos")
                mostrar_fig(fig, alto=max(480, 48 * len(nombres)))
                if corr.isna().any().any():
                    st.caption("Las celdas vacías corresponden a pares sin suficientes departamentos en común.")
                st.markdown("**Relación bivariada**")
                d1, d2 = st.columns(2)
                cols = list(ancho.columns)
                vx = d1.selectbox("Eje X", cols, index=0, format_func=etiqueta, key="corr_x")
                vy = d2.selectbox("Eje Y", cols, index=min(1, len(cols) - 1), format_func=etiqueta, key="corr_y")
                par = ancho[[vx, vy]].dropna() if vx != vy else pd.DataFrame()
                if len(par) < 3:
                    st.warning("Elige dos indicadores distintos con al menos 3 departamentos en común.")
                else:
                    fig = px.scatter(par.reset_index(), x=vx, y=vy, text="departamento_canonico",
                                     color_discrete_sequence=[COLOR_PRINCIPAL],
                                     labels={vx: etiqueta(vx), vy: etiqueta(vy)})
                    fig.update_traces(textposition="top center", textfont_size=9, marker_size=9)
                    if par[vx].std() > 0:
                        pendiente, intercepto = np.polyfit(par[vx], par[vy], 1)
                        xs = np.linspace(par[vx].min(), par[vx].max(), 50)
                        fig.add_trace(go.Scatter(x=xs, y=intercepto + pendiente * xs, mode="lines", name="Ajuste lineal",
                                                 line=dict(color=COLOR_SECUNDARIO, dash="dash")))
                    r = par[vx].corr(par[vy], method=metodo)
                    fig.update_layout(title=f"r = {r:.2f} · n = {len(par)}", showlegend=False)
                    mostrar_fig(fig, alto=520)
                nota("Correlaciones ecológicas (nivel departamental): no permiten inferir relaciones individuales ni causalidad.")
                st.download_button("Descargar matriz de correlaciones (CSV)", a_csv(corr, indice=True),
                                   f"correlaciones_{anio}.csv", "text/csv")

    with tab_pca:
        res = _controles_transversales(deptos, "pca")
        if res is None:
            return
        anio, ancho = res
        c1, c2 = st.columns([2, 1])
        negativos = c1.multiselect("Indicadores donde un valor más alto es peor (se invierten)", list(ancho.columns),
                                   default=[c for c in ancho.columns if c in POLARIDAD_NEGATIVA],
                                   format_func=etiqueta, key="pca_neg")
        cobertura_min = c2.slider("Cobertura mínima por departamento (%)", 30, 100, 60, 10, key="pca_cob")
        constantes = [c for c in ancho.columns if ancho[c].nunique(dropna=True) < 2]
        if constantes:
            st.warning("Indicadores sin variación (se excluyen): " + ", ".join(etiqueta(c) for c in constantes) + ".")
            ancho = ancho.drop(columns=constantes)
        cobertura = ancho.notna().mean(axis=1) * 100
        excluidos = cobertura[cobertura < cobertura_min].index.tolist()
        ancho = ancho.loc[cobertura >= cobertura_min]
        if excluidos:
            st.warning(f"{len(excluidos)} departamento(s) excluidos por cobertura inferior a {cobertura_min}%: "
                       + ", ".join(excluidos) + ".")
        if ancho.shape[0] < 4 or ancho.shape[1] < 2:
            st.warning("Datos insuficientes para PCA (se requieren ≥ 4 departamentos y ≥ 2 indicadores).")
            return
        matriz, nulos = imputar_media(ancho)
        if len(nulos):
            st.warning("Nulos imputados con la media del indicador → "
                       + "; ".join(f"{etiqueta(k)}: {int(v)}" for k, v in nulos.items()) + ".")
        n_max = min(matriz.shape[0] - 1, matriz.shape[1])
        n_comp = st.slider("Componentes a extraer", 2, max(2, n_max), min(max(2, n_max), 3), key="pca_n") if n_max > 2 else 2
        try:
            pca = ajustar_pca(matriz, tuple(sorted(negativos)), int(n_comp))
        except Exception as e:
            st.warning(f"No fue posible estimar el PCA con esta selección ({e}).")
            return

        var = pca["varianza"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Varianza explicada por PC1", f"{var[0]:.1%}")
        m2.metric("PC1 + PC2", f"{var[:2].sum():.1%}")
        m3.metric("Departamentos · indicadores", f"{matriz.shape[0]} · {matriz.shape[1]}")
        if var[0] < 0.4:
            st.warning("PC1 explica menos del 40% de la varianza: el índice sintético resume solo parcialmente los indicadores.")

        izq, der = st.columns([1.15, 1])
        with izq:
            ranking = pca["indice"].sort_values().reset_index()
            ranking.columns = ["departamento_canonico", "indice_sintetico"]
            fig = px.bar(ranking, x="indice_sintetico", y="departamento_canonico", orientation="h",
                         color="indice_sintetico", color_continuous_scale="Blues",
                         labels={"indice_sintetico": "Índice (0–100)", "departamento_canonico": ""})
            fig.update_layout(title=f"Índice Sintético Multidimensional · {anio}", coloraxis_showscale=False)
            mostrar_fig(fig, alto=max(460, 22 * len(ranking) + 120))
        with der:
            nombres_pc = [f"PC{i + 1}" for i in range(len(var))]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=nombres_pc, y=var * 100, name="Individual", marker_color=COLOR_PRINCIPAL))
            fig.add_trace(go.Scatter(x=nombres_pc, y=np.cumsum(var) * 100, name="Acumulada", mode="lines+markers",
                                     line=dict(color=COLOR_SECUNDARIO)))
            fig.update_layout(title="Varianza explicada (%)", yaxis_range=[0, 105],
                              legend=dict(orientation="h", y=1.12))
            mostrar_fig(fig, alto=340)
            cargas = pca["cargas"].copy()
            cargas.index = [("(−) " if c in negativos else "") + etiqueta(c) for c in cargas.index]
            st.markdown("**Cargas factoriales**")
            mostrar_tabla(cargas.round(3).reset_index().rename(columns={"index": "indicador"}))

        puntajes, cargas_raw = pca["puntajes"], pca["cargas"]
        escala = 0.85 * np.abs(puntajes[["PC1", "PC2"]].values).max() / max(np.abs(cargas_raw[["PC1", "PC2"]].values).max(), 1e-9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=puntajes["PC1"], y=puntajes["PC2"], mode="markers+text", text=puntajes.index,
            textposition="top center", textfont=dict(size=9),
            marker=dict(size=11, color=pca["indice"], colorscale="Blues", showscale=True,
                        colorbar=dict(title="Índice"), line=dict(color="#ffffff", width=1)),
            hovertemplate="%{text}<br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>", name="Departamentos"))
        for var_nombre, fila in cargas_raw.iterrows():
            x1, y1 = fila["PC1"] * escala, fila["PC2"] * escala
            fig.add_annotation(x=x1, y=y1, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowwidth=1.6, arrowcolor=COLOR_SECUNDARIO)
            fig.add_annotation(x=x1 * 1.08, y=y1 * 1.08, showarrow=False,
                               text=("(−) " if var_nombre in negativos else "") + etiqueta(var_nombre),
                               font=dict(size=10, color=COLOR_SECUNDARIO))
        fig.add_hline(y=0, line_color="#D0D5DD", line_width=1)
        fig.add_vline(x=0, line_color="#D0D5DD", line_width=1)
        fig.update_layout(title="Biplot: departamentos y cargas (PC1 vs PC2)", showlegend=False,
                          xaxis_title=f"PC1 ({var[0]:.1%})", yaxis_title=f"PC2 ({var[1]:.1%})")
        mostrar_fig(fig, alto=640)
        nota("Índice exploratorio, no oficial: PC1 de los indicadores estandarizados y alineados (mayor = mejor), "
             "reescalado a 0–100 entre los departamentos incluidos. «(−)» marca indicadores invertidos. "
             "Los valores no son comparables entre años ni entre selecciones distintas de indicadores.")
        salida = pd.concat([pca["indice"], puntajes], axis=1).sort_values("indice_sintetico", ascending=False)
        st.download_button("Descargar índice y puntajes (CSV)", a_csv(salida.reset_index()),
                           f"indice_sintetico_pca_{anio}.csv", "text/csv")


def pagina_espacial(panel, deptos, catalogo):
    st.header("Econometría espacial y cartografía interactiva")
    with st.sidebar.expander("Cartografía"):
        url = st.text_input("URL de GeoJSON departamental", GEOJSON_URL_DEPARTAMENTOS, key="geo_url")
        st.caption("Tiene prioridad cualquier archivo `.geojson` ubicado en `data/`.")
    geo = cargar_geo(url.strip())
    if not geo["ok"]:
        if geo["n_features"] and geo["n_features"] < 20:
            st.warning(f"El GeoJSON obtenido ({geo['fuente']}) tiene {geo['n_features']} polígono(s): es el contorno "
                       "del país, sin departamentos. Se usa un mapa de burbujas sobre las capitales y pesos KNN. "
                       "Para el coroplético, guarda un GeoJSON departamental en `data/`.")
        else:
            st.warning("No se pudo cargar un GeoJSON departamental (sin conexión o formato no reconocido). "
                       "Se usa un mapa de burbujas sobre las capitales y pesos KNN.")

    indicadores = sorted(deptos["indicador"].unique().tolist())
    c1, c2, c3 = st.columns([2, 1.3, 1])
    ind = c1.selectbox("Indicador", indicadores, index=indicadores.index("IPM") if "IPM" in indicadores else 0,
                       format_func=etiqueta, key="sp_ind")
    desag = c2.selectbox("Desagregación", desagregaciones_de(deptos, ind), key="sp_des")
    anios = anios_de(deptos, ind, desag)
    if not anios:
        st.warning("No hay datos departamentales para esta combinación.")
        return
    anio = c3.selectbox("Año", anios, key="sp_anio")
    datos = deptos[(deptos["indicador"] == ind) & (deptos["desagregacion"] == desag) & (deptos["anio"] == anio)]
    datos = datos.dropna(subset=["valor"]).groupby("departamento_canonico", as_index=False)["valor"].mean()
    unidad = unidad_de(catalogo, panel, ind) or "valor"
    escala_color = "OrRd" if ind in POLARIDAD_NEGATIVA else "Blues"

    tab_mapa, tab_moran = st.tabs(["Mapa", "Autocorrelación espacial (I de Moran)"])
    with tab_mapa:
        if geo["ok"]:
            mapa = geo["tabla"].merge(datos, on="departamento_canonico", how="left")
            sin_geo = sorted(set(datos["departamento_canonico"]) - set(geo["tabla"]["departamento_canonico"]))
            fig = px.choropleth(mapa, geojson=geo["geojson"], locations="id", color="valor",
                                hover_name="departamento_canonico", color_continuous_scale=escala_color,
                                labels={"valor": unidad}, hover_data={"id": False, "valor": ":,.2f"})
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(title=f"{etiqueta(ind)} · {desag} · {anio}")
            mostrar_fig(fig, alto=680)
            sin_dato = int(mapa.groupby("departamento_canonico")["valor"].first().isna().sum())
            if sin_dato:
                st.caption(f"{sin_dato} departamento(s) sin dato para esta selección (sin color en el mapa).")
            if sin_geo:
                st.warning("Sin polígono en el GeoJSON: " + ", ".join(sin_geo) + ".")
            st.caption(f"Cartografía: {geo['fuente']}")
        else:
            burb = datos[datos["departamento_canonico"].isin(CAPITALES)].copy()
            if burb.empty:
                st.warning("No hay departamentos con coordenadas de respaldo para dibujar.")
            else:
                burb["lat"] = burb["departamento_canonico"].map(lambda d: CAPITALES[d][0])
                burb["lon"] = burb["departamento_canonico"].map(lambda d: CAPITALES[d][1])
                tam = burb["valor"] - burb["valor"].min()
                burb["tamano"] = 8 + 30 * (tam / tam.max() if tam.max() > 0 else 0)
                fig = px.scatter_geo(burb, lat="lat", lon="lon", size="tamano", color="valor",
                                     hover_name="departamento_canonico", color_continuous_scale=escala_color,
                                     labels={"valor": unidad}, hover_data={"lat": False, "lon": False, "tamano": False})
                fig.update_geos(lataxis_range=[-5, 14], lonaxis_range=[-83, -66], showcountries=True,
                                countrycolor="#9AA4B2", showland=True, landcolor="#F3F5F8", resolution=50)
                fig.update_layout(title=f"{etiqueta(ind)} · {desag} · {anio} (burbujas en capitales)")
                mostrar_fig(fig, alto=680)

    with tab_moran:
        c1, c2, c3 = st.columns([1.6, 1, 1])
        opciones = ["KNN (k vecinos más cercanos)"]
        if geo["ok"]:
            opciones = ["Contigüidad (fronteras compartidas)"] + opciones
        tipo_w = c1.selectbox("Matriz de pesos espaciales", opciones, key="sp_w")
        k = c2.slider("k (solo KNN)", 2, 8, 4, key="sp_k")
        permutaciones = c3.select_slider("Permutaciones", [199, 499, 999, 1999], value=999, key="sp_perm")

        if geo["ok"]:
            centroides = geo["tabla"].groupby("departamento_canonico")[["lat", "lon"]].mean()
        else:
            centroides = pd.DataFrame(CAPITALES, index=["lat", "lon"]).T
        base = datos[datos["departamento_canonico"].isin(centroides.index)].sort_values("departamento_canonico")
        nombres = base["departamento_canonico"].tolist()
        y = base["valor"].to_numpy(dtype=float)
        if len(nombres) < 8:
            st.warning(f"Solo {len(nombres)} departamentos tienen dato y geometría; se requieren al menos 8.")
            return
        if np.std(y) == 0:
            st.warning("El indicador no varía entre departamentos; el I de Moran no está definido.")
            return
        coords = centroides.loc[nombres, ["lat", "lon"]].to_numpy(dtype=float)
        if tipo_w.startswith("Contigüidad"):
            vecinos = {n: [v for v in geo["vecinos"].get(n, []) if v in nombres] for n in nombres}
            if sum(len(v) for v in vecinos.values()) == 0:
                st.warning("El GeoJSON no comparte vértices entre polígonos; se usan pesos KNN.")
                vecinos = pesos_knn(coords, nombres, k)
            else:
                islas = [n for n, v in vecinos.items() if not v]
                if islas:
                    cercano = pesos_knn(coords, nombres, 1)
                    for isla in islas:
                        vecinos[isla] = cercano[isla]
                        for v in cercano[isla]:
                            if isla not in vecinos[v]:
                                vecinos[v].append(isla)
                    st.info("Sin vecinos contiguos entre los departamentos con dato (se enlazan al más cercano): "
                            + ", ".join(islas) + ".")
        else:
            vecinos = pesos_knn(coords, nombres, k)

        res = moran_global(y, vecinos, nombres, int(permutaciones))
        if np.isnan(res["I"]):
            st.warning("No fue posible calcular el I de Moran con esta configuración.")
            return
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("I de Moran", f"{res['I']:.3f}")
        m2.metric("Valor esperado E[I]", f"{res['EI']:.3f}")
        m3.metric("z (permutaciones)", fmt_num(res["z_score"]))
        m4.metric("p-valor (pseudo)", f"{res['p_valor']:.3f}")
        if res["p_valor"] < 0.05 and res["I"] > res["EI"]:
            st.success("Autocorrelación espacial positiva significativa (5%): departamentos vecinos tienden a "
                       "presentar valores similares (clusterización).")
        elif res["p_valor"] < 0.05:
            st.success("Autocorrelación espacial negativa significativa (5%): departamentos vecinos tienden a "
                       "presentar valores disímiles.")
        else:
            st.info("No se rechaza la aleatoriedad espacial al 5%: no hay evidencia de clusterización.")
        st.caption(f"n = {len(nombres)} departamentos · pesos estandarizados por fila · motor de cálculo: {res['motor']}.")
        if not HAY_PYSAL:
            st.caption("`libpysal`/`esda` no están instalados; se usó una implementación equivalente en NumPy.")

        zs = res["z"] / res["z"].std()
        rezago = res["rezago"] / res["z"].std()
        cuad = np.where((zs >= 0) & (rezago >= 0), "Alto–Alto",
                        np.where((zs < 0) & (rezago < 0), "Bajo–Bajo",
                                 np.where(zs >= 0, "Alto–Bajo", "Bajo–Alto")))
        disp = pd.DataFrame({"departamento": nombres, "z": zs, "rezago": rezago, "Cuadrante": cuad})
        fig = px.scatter(disp, x="z", y="rezago", color="Cuadrante", text="departamento",
                         color_discrete_map={"Alto–Alto": "#B2382F", "Bajo–Bajo": COLOR_PRINCIPAL,
                                             "Alto–Bajo": "#E8A19A", "Bajo–Alto": "#9CC3DE"},
                         labels={"z": f"{etiqueta(ind)} (estandarizado)", "rezago": "Rezago espacial"})
        fig.update_traces(textposition="top center", textfont_size=9, marker_size=10)
        xs = np.linspace(zs.min(), zs.max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=res["I"] * xs, mode="lines", name=f"Pendiente = I ({res['I']:.2f})",
                                 line=dict(color="#344054", dash="dash")))
        fig.add_hline(y=0, line_color="#D0D5DD")
        fig.add_vline(x=0, line_color="#D0D5DD")
        fig.update_layout(title="Diagrama de dispersión de Moran")
        mostrar_fig(fig, alto=560)
        if not geo["ok"]:
            nota("Los pesos se construyeron con las coordenadas aproximadas de las capitales departamentales.")


def pagina_series(panel, catalogo):
    st.header("Econometría de series de tiempo (VAR / VEC)")
    if not HAY_STATSMODELS:
        st.error("Este módulo requiere `statsmodels`. Instálalo con `pip install statsmodels` y recarga la aplicación.")
        return
    totales = panel[panel["desagregacion"] == "Total"].dropna(subset=["valor"])
    indicadores = sorted(totales["indicador"].unique().tolist())
    if len(indicadores) < 2:
        st.warning("Se requieren al menos dos indicadores en desagregación «Total».")
        return

    c1, c2 = st.columns([1.2, 2])
    modos = {"Promedio simple de departamentos": "media", "Suma de departamentos": "suma",
             "Total nacional (dato oficial)": "territorio", "Un departamento": "territorio"}
    modo = c1.selectbox("Unidad de análisis", list(modos), key="ts_modo")
    territorio = TOTAL_NACIONAL
    if modo == "Un departamento":
        lista = sorted(d for d in totales["departamento_canonico"].unique() if d != TOTAL_NACIONAL)
        territorio = c1.selectbox("Departamento", lista, key="ts_dep")
    defecto = [i for i in ["PIB_tasa_crecimiento", "Tasa de Desocupación (TD)", "Tasa de Ocupación (TO)"]
               if i in indicadores] or indicadores[:2]
    sel = c2.multiselect("Variables endógenas (2 a 4)", indicadores, default=defecto[:3], format_func=etiqueta,
                         key="ts_vars")
    if not 2 <= len(sel) <= 4:
        st.warning("Selecciona entre 2 y 4 variables.")
        return

    series = construir_series(panel, sel, territorio, modos[modo])
    faltan = [etiqueta(i) for i in sel if i not in series.columns]
    if faltan:
        st.warning("Sin datos para esta unidad de análisis: " + ", ".join(faltan)
                   + ". Prueba con «Promedio simple de departamentos».")
        return
    # Muestra común: recorta extremos sin dato e interpola huecos internos.
    validas = series.dropna(how="any")
    if validas.empty:
        st.warning("Las variables seleccionadas no coinciden en ningún año.")
        return
    series = series.loc[validas.index.min():validas.index.max()]
    series = series.reindex(range(int(series.index.min()), int(series.index.max()) + 1))
    series.index.name = "anio"
    huecos = int(series.isna().sum().sum())
    if huecos:
        st.warning(f"Se interpolaron linealmente {huecos} valor(es) faltante(s) dentro del periodo común.")
        series = series.interpolate(method="linear", limit_direction="both")

    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    tipo_modelo = c1.radio("Modelo", ["VAR", "VEC"], horizontal=True, key="ts_tipo")
    transformaciones = ["Niveles", "Primera diferencia"]
    if (series > 0).all().all():
        transformaciones.append("Diferencia logarítmica (×100)")
    if tipo_modelo == "VEC":
        transformacion = "Niveles"
        c2.caption("El VEC se estima en niveles: modela la relación de largo plazo entre series integradas.")
    else:
        transformacion = c2.selectbox("Transformación", transformaciones, key="ts_trans")
    horizonte = c3.slider("Horizonte IRF (años)", 3, 12, 8, key="ts_h")

    if transformacion == "Primera diferencia":
        datos = series.diff().dropna()
    elif transformacion.startswith("Diferencia log"):
        datos = (np.log(series).diff() * 100).dropna()
    else:
        datos = series.copy()
    constantes = [c for c in datos.columns if datos[c].std() == 0]
    if constantes:
        st.warning("Series sin variación en la muestra: " + ", ".join(etiqueta(c) for c in constantes) + ".")
        return
    n, kv = len(datos), datos.shape[1]
    if n < 10:
        st.warning(f"La muestra común tiene solo {n} observaciones anuales; se requieren al menos 10 para estimar "
                   "un VAR/VEC. Pobreza monetaria y Gini (2021–2025) e IPM (2018–2025) son demasiado cortas: "
                   "usa indicadores de PIB o del mercado laboral, o cambia la unidad de análisis.")
        return

    fig = px.line(datos.reset_index().melt(id_vars="anio", var_name="indicador", value_name="valor"),
                  x="anio", y="valor", facet_row="indicador", markers=True,
                  color_discrete_sequence=[COLOR_PRINCIPAL], labels={"anio": "Año", "valor": ""})
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=etiqueta(a.text.split("=")[-1])))
    fig.update_layout(title=f"Series analizadas · {modo if modo != 'Un departamento' else territorio} · "
                            f"{transformacion.lower()} · {int(datos.index.min())}–{int(datos.index.max())} (n = {n})")
    mostrar_fig(fig, alto=170 * kv + 90)
    nota(f"Series anuales cortas (n = {n}). Las pruebas tienen baja potencia y los resultados son exploratorios: "
         "úsalos para describir dinámicas conjuntas, no como evidencia causal concluyente.")

    with st.expander("Pruebas de raíz unitaria (Dickey-Fuller aumentada)", expanded=False):
        filas = []
        for col in datos.columns:
            try:
                est, p, rez, *_ = adfuller(datos[col].values, maxlag=max(0, min(2, n // 2 - 3)), autolag="AIC")
                filas.append({"variable": etiqueta(col), "estadístico ADF": round(est, 3), "p-valor": round(p, 3),
                              "rezagos": rez, "conclusión (5%)": "Estacionaria" if p < 0.05 else "No se rechaza raíz unitaria"})
            except Exception:
                filas.append({"variable": etiqueta(col), "estadístico ADF": np.nan, "p-valor": np.nan,
                              "rezagos": np.nan, "conclusión (5%)": "Muestra insuficiente"})
        mostrar_tabla(pd.DataFrame(filas))

    if tipo_modelo == "VAR":
        p_max = min(4, (n - 4) // (kv + 1))
        if p_max < 1:
            st.warning(f"Muestra insuficiente (n = {n}) para un VAR con {kv} variables. Reduce las variables "
                       "o cambia la unidad de análisis.")
            return
        p = st.slider("Rezagos (p)", 1, p_max, 1, key="ts_p") if p_max > 1 else 1
        try:
            res = estimar_var(datos, int(p))
        except Exception as e:
            st.warning(f"No se pudo estimar el VAR ({e}). Prueba otra transformación o menos rezagos.")
            return
        try:
            estable = bool(res.is_stable())
        except Exception:
            estable = False
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observaciones efectivas", int(res.nobs))
        m2.metric("AIC", fmt_num(res.aic))
        m3.metric("BIC", fmt_num(res.bic))
        m4.metric("Estabilidad", "Estable" if estable else "No estable")
        if not estable:
            st.warning("El VAR no es estable (raíces fuera del círculo unitario): las IRF pueden no converger. "
                       "Considera diferenciar las series o estimar un VEC.")

        st.subheader("Causalidad de Granger")
        gr = pd.DataFrame(np.nan, index=[etiqueta(c) for c in datos.columns],
                          columns=[etiqueta(c) for c in datos.columns])
        for causa in datos.columns:
            for efecto in datos.columns:
                if causa == efecto:
                    continue
                try:
                    gr.loc[etiqueta(causa), etiqueta(efecto)] = res.test_causality(efecto, [causa], kind="f").pvalue
                except Exception:
                    pass
        fig = px.imshow(gr.values.astype(float), x=list(gr.columns), y=list(gr.index), zmin=0, zmax=0.2,
                        color_continuous_scale="Blues_r", text_auto=".3f", aspect="auto",
                        labels=dict(x="Variable explicada", y="Variable causante", color="p-valor"))
        fig.update_layout(title="p-valores: H₀ = la fila no causa (Granger) a la columna")
        mostrar_fig(fig, alto=140 + 80 * kv)
        st.caption("p < 0,05 indica que los rezagos de la fila mejoran la predicción de la columna.")
        objeto_irf, modelo_txt = res, res
    else:
        kd_max = max(0, min(3, (n - 3) // (kv + 1) - 1))
        if n < kv + 6:
            st.warning(f"Muestra insuficiente (n = {n}) para un VEC con {kv} variables.")
            return
        c1, c2 = st.columns(2)
        kd = c1.slider("Rezagos en diferencias", 0, kd_max, min(1, kd_max), key="ts_kd") if kd_max > 0 else 0
        try:
            joh = coint_johansen(datos.values, det_order=0, k_ar_diff=int(kd))
            tabla_j = pd.DataFrame({"H₀: rango ≤": range(kv), "estadístico de traza": np.round(joh.lr1, 3),
                                    "crítico 90%": joh.cvt[:, 0], "crítico 95%": joh.cvt[:, 1],
                                    "crítico 99%": joh.cvt[:, 2]})
            tabla_j["rechaza al 5%"] = np.where(tabla_j["estadístico de traza"] > tabla_j["crítico 95%"], "Sí", "No")
            sugerido = 0
            for rechaza in tabla_j["rechaza al 5%"]:
                if rechaza != "Sí":
                    break
                sugerido += 1
            st.subheader("Cointegración de Johansen (traza)")
            mostrar_tabla(tabla_j)
            if sugerido == 0:
                st.warning("La prueba no detecta relaciones de cointegración al 5%; un VAR en diferencias sería "
                           "más apropiado. El VEC se estima con rango 1 solo con fines ilustrativos.")
            elif sugerido >= kv:
                st.warning("La prueba sugiere rango completo (series estacionarias en niveles); "
                           "un VAR en niveles sería más apropiado.")
        except Exception as e:
            sugerido = 1
            st.warning(f"No se pudo ejecutar la prueba de Johansen ({e}).")
        rango_defecto = int(min(max(1, sugerido), kv - 1))
        rango = c2.slider("Rango de cointegración (r)", 1, kv - 1, rango_defecto, key="ts_r") if kv > 2 else 1
        try:
            res = estimar_vec(datos, int(kd), int(rango))
        except Exception as e:
            st.warning(f"No se pudo estimar el VEC ({e}). Prueba con menos rezagos o variables.")
            return
        nombres_e = [etiqueta(c) for c in datos.columns]
        ce = [f"EC{i + 1}" for i in range(int(rango))]
        a, b = st.columns(2)
        a.markdown("**α · velocidad de ajuste al equilibrio**")
        with a:
            mostrar_tabla(pd.DataFrame(res.alpha, index=nombres_e, columns=ce).round(4).reset_index()
                          .rename(columns={"index": "ecuación"}))
        b.markdown("**β · vectores de cointegración**")
        with b:
            mostrar_tabla(pd.DataFrame(res.beta, index=nombres_e, columns=ce).round(4).reset_index()
                          .rename(columns={"index": "variable"}))
        st.subheader("Causalidad de Granger")
        filas = []
        for causa in datos.columns:
            for efecto in datos.columns:
                if causa == efecto:
                    continue
                try:
                    t = res.test_granger_causality(caused=efecto, causing=causa)
                    filas.append({"causa": etiqueta(causa), "efecto": etiqueta(efecto),
                                  "estadístico": round(float(t.test_statistic), 3), "p-valor": round(float(t.pvalue), 4),
                                  "conclusión (5%)": "Causa en sentido de Granger" if t.pvalue < 0.05 else "No se rechaza H₀"})
                except Exception:
                    filas.append({"causa": etiqueta(causa), "efecto": etiqueta(efecto), "estadístico": np.nan,
                                  "p-valor": np.nan, "conclusión (5%)": "No calculable con esta especificación"})
        mostrar_tabla(pd.DataFrame(filas))
        objeto_irf, modelo_txt = res, res

    st.subheader("Funciones de impulso-respuesta")
    c1, c2 = st.columns([2, 1])
    impulso = c1.selectbox("Choque (impulso) en", list(datos.columns), format_func=etiqueta, key="ts_imp")
    ortogonal = c2.checkbox("Ortogonalizadas (Cholesky)", value=False, key="ts_orth")
    try:
        irf = objeto_irf.irf(int(horizonte))
        matriz = irf.orth_irfs if ortogonal else irf.irfs
        j = list(datos.columns).index(impulso)
        errores = None
        if tipo_modelo == "VAR":
            try:
                errores = irf.stderr(orth=ortogonal)
            except Exception:
                errores = None
        fig = go.Figure()
        paleta = [COLOR_PRINCIPAL, COLOR_SECUNDARIO, "#3C8D5A", "#7A5AA6"]
        pasos = list(range(matriz.shape[0]))
        for i, col in enumerate(datos.columns):
            color = paleta[i % len(paleta)]
            fig.add_trace(go.Scatter(x=pasos, y=matriz[:, i, j], mode="lines+markers", name=etiqueta(col),
                                     line=dict(color=color)))
            if errores is not None and np.isfinite(errores[:, i, j]).all():
                sup, inf = matriz[:, i, j] + 1.96 * errores[:, i, j], matriz[:, i, j] - 1.96 * errores[:, i, j]
                fig.add_trace(go.Scatter(x=pasos + pasos[::-1], y=list(sup) + list(inf[::-1]), fill="toself",
                                         fillcolor=color, opacity=0.12, line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
        fig.add_hline(y=0, line_color="#98A2B3", line_width=1)
        fig.update_layout(title=f"Respuesta ante un choque en {etiqueta(impulso)}"
                                + (" (1 desv. estándar)" if ortogonal else " (1 unidad)"),
                          xaxis_title="Años tras el choque", yaxis_title="Respuesta", hovermode="x unified")
        mostrar_fig(fig, alto=480)
        if ortogonal:
            st.caption("La ortogonalización depende del orden de las variables seleccionadas (descomposición de Cholesky).")
        if errores is not None:
            st.caption("Bandas: ±1,96 errores estándar asintóticos; con muestras cortas son solo orientativas.")
    except Exception as e:
        st.warning(f"No se pudieron calcular las funciones de impulso-respuesta ({e}).")

    with st.expander("Resumen completo del modelo"):
        try:
            st.text(str(modelo_txt.summary()))
        except Exception as e:
            st.warning(f"Resumen no disponible ({e}).")
    st.download_button("Descargar series utilizadas (CSV)", a_csv(datos.reset_index()), "series_var_vec.csv", "text/csv")


def pagina_ml(deptos):
    st.header("Machine Learning: regresión y regularización")
    indicadores_total = sorted(deptos.loc[(deptos["desagregacion"] == "Total") & deptos["valor"].notna(),
                                          "indicador"].unique().tolist())
    if len(indicadores_total) < 2:
        st.warning("Se requieren al menos dos indicadores en desagregación «Total».")
        return
    objetivo, features = selector_ml(deptos, indicadores_total, "ml")

    with st.expander("Hiperparámetros y validación", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        a_ridge = c1.select_slider("Ridge · α", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], value=1.0)
        a_lasso = c2.select_slider("LASSO · α", [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0], value=0.05)
        c_svr = c3.select_slider("SVR · C", [0.1, 1.0, 5.0, 10.0, 50.0, 100.0], value=10.0)
        eps = c4.select_slider("SVR · ε", [0.01, 0.05, 0.1, 0.2, 0.5], value=0.1)
        d1, d2 = st.columns(2)
        kernel = d1.radio("SVR · kernel", ["rbf", "linear"], horizontal=True)
        esquema = d2.radio("Validación cruzada", ["Por departamento (GroupKFold)", "Aleatoria (KFold)"],
                           horizontal=True)
        st.caption("α y ε se expresan sobre variables y objetivo estandarizados. GroupKFold evalúa la capacidad "
                   "de generalizar a departamentos no vistos y evita fuga de información entre años.")
    hp = (float(a_ridge), float(a_lasso), float(c_svr), float(eps), kernel)
    st.session_state["ml_hp"] = hp
    st.session_state["ml_esquema"] = "grupos" if esquema.startswith("Por") else "aleatoria"

    prep = preparar_dataset_ml(deptos, objetivo, features)
    if prep is None:
        return
    X, y, grupos, _, validas = prep
    try:
        res = entrenar_modelos(X, y, grupos, hp, st.session_state["ml_esquema"])
    except Exception as e:
        st.warning(f"No fue posible entrenar los modelos con esta selección ({e}).")
        return

    met = res["metricas"].sort_values("RMSE").reset_index(drop=True)
    mejor = met.loc[0, "modelo"]
    st.session_state["ml_mejor"] = mejor
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observaciones (dpto-año)", len(X))
    m2.metric("Departamentos", grupos.nunique())
    m3.metric("Mejor modelo (RMSE)", mejor)
    m4.metric("R² del mejor (CV)", f"{met.loc[0, 'R² (media CV)']:.3f}")
    st.caption(f"Validación: {res['esquema']}, {res['n_pliegues']} pliegues · objetivo: {etiqueta(objetivo)}.")
    mostrar_tabla(met.round(4))

    izq, der = st.columns(2)
    with izq:
        largo = met.melt(id_vars="modelo", value_vars=["RMSE", "MAE"], var_name="métrica", value_name="valor")
        fig = px.bar(largo, x="modelo", y="valor", color="métrica", barmode="group",
                     color_discrete_sequence=[COLOR_PRINCIPAL, COLOR_SECUNDARIO], labels={"valor": "Error", "modelo": ""})
        fig.update_layout(title="Error de validación cruzada (menor es mejor)")
        mostrar_fig(fig, alto=400)
    with der:
        fig = px.bar(met, x="modelo", y="R² (media CV)", error_y="R² (desv.)",
                     color_discrete_sequence=[COLOR_PRINCIPAL], labels={"modelo": ""})
        fig.update_layout(title="R² medio en validación cruzada (± desv. entre pliegues)")
        mostrar_fig(fig, alto=400)
    if (met["R² (media CV)"] < 0).any():
        st.warning("Algún modelo tiene R² negativo en validación: predice peor que la media. "
                   "Revisa las variables explicativas o el esquema de validación.")

    st.subheader("Observado frente a predicho (fuera de muestra)")
    modelo_sel = st.selectbox("Modelo", list(res["modelos"]), index=list(res["modelos"]).index(mejor), key="ml_modelo")
    pred = pd.DataFrame({"observado": y.values, "predicho": res["predicciones"][modelo_sel], "departamento": grupos.values})
    fig = px.scatter(pred, x="observado", y="predicho", hover_name="departamento",
                     color_discrete_sequence=[COLOR_PRINCIPAL], opacity=0.75)
    lim = [float(min(pred["observado"].min(), pred["predicho"].min())),
           float(max(pred["observado"].max(), pred["predicho"].max()))]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#98A2B3", dash="dash"), showlegend=False))
    fig.update_layout(title=f"{modelo_sel}: {etiqueta(objetivo)}")
    mostrar_fig(fig, alto=480)

    st.subheader("Coeficientes estandarizados y regularización")
    coefs = pd.DataFrame({"variable": [etiqueta(v) for v in validas]})
    for nombre in ("Ridge", "LASSO"):
        coefs[nombre] = res["modelos"][nombre].regressor_.named_steps["modelo"].coef_
    largo = coefs.melt(id_vars="variable", var_name="modelo", value_name="coeficiente")
    fig = px.bar(largo, x="coeficiente", y="variable", color="modelo", barmode="group", orientation="h",
                 color_discrete_sequence=[COLOR_PRINCIPAL, COLOR_SECUNDARIO], labels={"variable": ""})
    fig.add_vline(x=0, line_color="#98A2B3")
    fig.update_layout(title="Efecto de +1 desv. estándar de cada variable (en desv. estándar del objetivo)")
    mostrar_fig(fig, alto=max(340, 46 * len(validas) + 120))
    anuladas = coefs.loc[coefs["LASSO"].abs() < 1e-10, "variable"].tolist()
    if anuladas:
        st.info("LASSO anuló (coeficiente = 0): " + ", ".join(anuladas) + ".")
    nota("Los modelos describen asociaciones predictivas entre indicadores departamentales; no identifican efectos causales.")


def pagina_simulador(deptos, catalogo, panel):
    st.header("Simulador de política pública e interpretabilidad (XAI)")
    indicadores_total = sorted(deptos.loc[(deptos["desagregacion"] == "Total") & deptos["valor"].notna(),
                                          "indicador"].unique().tolist())
    if len(indicadores_total) < 2:
        st.warning("Se requieren al menos dos indicadores en desagregación «Total».")
        return
    objetivo, features = selector_ml(deptos, indicadores_total, "sim")
    prep = preparar_dataset_ml(deptos, objetivo, features)
    if prep is None:
        return
    X, y, grupos, ancho, validas = prep
    hp = st.session_state.get("ml_hp", HP_DEFECTO)
    esquema = st.session_state.get("ml_esquema", "grupos")
    try:
        res = entrenar_modelos(X, y, grupos, hp, esquema)
    except Exception as e:
        st.warning(f"No fue posible entrenar los modelos ({e}).")
        return
    nombres_modelos = list(res["modelos"])
    mejor = res["metricas"].sort_values("RMSE")["modelo"].iloc[0]
    c1, c2 = st.columns([1, 1.4])
    nombre_modelo = c1.selectbox("Modelo predictivo", nombres_modelos, index=nombres_modelos.index(mejor), key="sim_mod")
    territorios = sorted(ancho.index.get_level_values("departamento_canonico").unique().tolist())
    dep = c2.selectbox("Departamento (escenario base)", territorios, key="sim_dep")
    modelo = res["modelos"][nombre_modelo]
    fila_met = res["metricas"].set_index("modelo").loc[nombre_modelo]
    st.caption(f"{nombre_modelo} · R² CV = {fila_met['R² (media CV)']:.3f} · RMSE CV = {fmt_num(fila_met['RMSE'])} · "
               "hiperparámetros definidos en el módulo 6.")
    if fila_met["R² (media CV)"] < 0.3:
        st.warning("El modelo tiene bajo poder predictivo fuera de muestra (R² < 0,3): interpreta la simulación con cautela.")

    # Escenario base: último valor disponible de cada variable para el departamento.
    hist = ancho.xs(dep, level="departamento_canonico").sort_index()
    medias = X.mean()
    base, origen = {}, {}
    for v in validas:
        s = hist[v].dropna() if v in hist.columns else pd.Series(dtype=float)
        if len(s):
            base[v], origen[v] = float(s.iloc[-1]), f"{int(s.index[-1])}"
        else:
            base[v], origen[v] = float(medias[v]), "media nacional (sin dato)"
    sin_dato = [etiqueta(v) for v in validas if origen[v].startswith("media")]
    if sin_dato:
        st.warning(f"{dep} no tiene dato de: " + ", ".join(sin_dato) + ". Se usa la media de los departamentos.")

    st.subheader("Variables de política")
    escenario = {}
    columnas = st.columns(min(3, len(validas)))
    for i, v in enumerate(validas):
        lo, hi = float(min(X[v].min(), base[v])), float(max(X[v].max(), base[v]))
        with columnas[i % len(columnas)]:
            if hi - lo <= 0:
                st.metric(etiqueta(v), fmt_num(base[v]))
                escenario[v] = base[v]
                continue
            margen = 0.1 * (hi - lo)
            escenario[v] = st.slider(etiqueta(v), min_value=lo - margen, max_value=hi + margen, value=base[v],
                                     step=(hi - lo) / 200, key=f"sim_{dep}_{v}",
                                     help=f"Base: {fmt_num(base[v])} ({origen[v]}) · unidad: "
                                          f"{unidad_de(catalogo, panel, v) or 'n/d'}")
    x_base = pd.DataFrame([base])[validas]
    x_esc = pd.DataFrame([escenario])[validas]
    pred_base, pred_esc = float(modelo.predict(x_base)[0]), float(modelo.predict(x_esc)[0])
    obs = hist[objetivo].dropna() if objetivo in hist.columns else pd.Series(dtype=float)
    delta = pred_esc - pred_base
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Observado ({int(obs.index[-1])})" if len(obs) else "Observado", fmt_num(obs.iloc[-1]) if len(obs) else "—")
    m2.metric("Predicción base", fmt_num(pred_base))
    m3.metric("Predicción del escenario", fmt_num(pred_esc), f"{delta:+,.3f}",
              delta_color="inverse" if objetivo in POLARIDAD_NEGATIVA else "normal")
    m4.metric("Cambio relativo", f"{(delta / pred_base * 100) if pred_base else 0:+.1f}%")
    fuera = [etiqueta(v) for v in validas if escenario[v] < X[v].min() or escenario[v] > X[v].max()]
    if fuera:
        st.warning("Valores fuera del rango observado en los datos (extrapolación): " + ", ".join(fuera) + ".")
    cambios = pd.DataFrame({"variable": [etiqueta(v) for v in validas], "base": [base[v] for v in validas],
                            "escenario": [escenario[v] for v in validas]})
    cambios["cambio"] = cambios["escenario"] - cambios["base"]
    with st.expander("Detalle del escenario"):
        mostrar_tabla(cambios.round(3))

    st.subheader("Interpretabilidad del modelo")
    X_imp = X.fillna(medias)
    fondo = X_imp.sample(min(len(X_imp), 80), random_state=42)
    firma = f"{nombre_modelo}|{objetivo}|{'|'.join(validas)}|{hp}|{len(X)}"
    locales, globales, base_shap, metodo = None, None, None, ""
    if HAY_SHAP:
        try:
            g = calcular_shap(modelo, firma, fondo, fondo)
            l = calcular_shap(modelo, firma + "|escenario", fondo, x_esc)
            globales, locales, base_shap = np.abs(g["valores"]).mean(axis=0), l["valores"][0], l["base"]
            metodo = "valores SHAP (`shap.Explainer`)"
        except Exception as e:
            st.warning(f"SHAP no pudo calcularse ({e}); se usa un método alternativo.")
    if locales is None:
        try:
            imp = permutation_importance(modelo, X_imp, y, n_repeats=20, random_state=42,
                                         scoring="neg_root_mean_squared_error")
            globales = np.clip(imp.importances_mean, 0, None)
            locales = contribuciones_aproximadas(modelo, X_imp, x_esc)
            base_shap = float(modelo.predict(pd.DataFrame([medias.values], columns=validas))[0])
            metodo = "importancia por permutación y contribuciones marginales (instala `shap` para valores SHAP)"
        except Exception as e:
            st.warning(f"No fue posible calcular la importancia de variables ({e}).")
            return
    st.caption(f"Método: {metodo}.")
    izq, der = st.columns(2)
    with izq:
        g_df = pd.DataFrame({"variable": [etiqueta(v) for v in validas], "importancia": globales}).sort_values("importancia")
        fig = px.bar(g_df, x="importancia", y="variable", orientation="h", color_discrete_sequence=[COLOR_PRINCIPAL],
                     labels={"variable": "", "importancia": "Importancia media"})
        fig.update_layout(title="Importancia global de cada variable")
        mostrar_fig(fig, alto=max(340, 44 * len(validas) + 120))
    with der:
        l_df = pd.DataFrame({"variable": [etiqueta(v) for v in validas], "contribucion": locales}).sort_values("contribucion")
        positivo_malo = objetivo in POLARIDAD_NEGATIVA
        colores = [("#B2382F" if (c > 0) == positivo_malo else "#3C8D5A") for c in l_df["contribucion"]]
        fig = go.Figure(go.Bar(x=l_df["contribucion"], y=l_df["variable"], orientation="h", marker_color=colores))
        fig.add_vline(x=0, line_color="#98A2B3")
        fig.update_layout(title=f"Contribución al escenario de {dep}",
                          xaxis_title=f"Efecto sobre {etiqueta(objetivo)} frente al valor de referencia ({fmt_num(base_shap)})")
        mostrar_fig(fig, alto=max(340, 44 * len(validas) + 120))
    st.caption("Verde: la variable mueve el objetivo en dirección favorable; rojo: desfavorable.")
    nota("El simulador responde «¿qué predice el modelo si cambian estos indicadores?», manteniendo lo demás constante. "
         "Refleja asociaciones estadísticas entre departamentos, no el efecto causal de una política.")


def pagina_ficha(panel, deptos, catalogo):
    st.header("Ficha departamental y descarga de datos")
    tab_ficha, tab_desc = st.tabs(["Ficha departamental", "Descarga personalizada"])

    with tab_ficha:
        territorios = sorted(deptos["departamento_canonico"].unique().tolist())
        if not territorios:
            st.warning("El panel no contiene departamentos.")
            return
        defecto = territorios.index("Antioquia") if "Antioquia" in territorios else 0
        dep = st.selectbox("Departamento", territorios, index=defecto, key="fi_dep")
        codigo = deptos.loc[deptos["departamento_canonico"] == dep, "codigo_divipola"].dropna()
        st.markdown(f"### {dep} <span class='obs-chip'>DIVIPOLA {codigo.iloc[0] if len(codigo) else '—'}</span>",
                    unsafe_allow_html=True)
        filas = []
        for ind in [i for i in INDICADORES_CLAVE if i in set(deptos["indicador"])]:
            s, anio = serie_ultimo_anio(deptos[deptos["departamento_canonico"] == dep], ind)
            if s is None:
                continue
            valor = float(s["valor"].mean())
            corte = deptos[(deptos["indicador"] == ind) & (deptos["desagregacion"] == "Total") &
                           (deptos["anio"] == anio)].dropna(subset=["valor"])
            corte = corte.groupby("departamento_canonico")["valor"].mean()
            mayor_mejor = ind not in POLARIDAD_NEGATIVA
            puesto = int(corte.rank(ascending=not mayor_mejor, method="min").get(dep, np.nan))
            nac = panel[(panel["indicador"] == ind) & (panel["desagregacion"] == "Total") & (panel["anio"] == anio) &
                        (panel["departamento_canonico"] == TOTAL_NACIONAL)]["valor"].dropna()
            previo = deptos[(deptos["departamento_canonico"] == dep) & (deptos["indicador"] == ind) &
                            (deptos["desagregacion"] == "Total") & (deptos["anio"] < anio)].dropna(subset=["valor"])
            previo = previo.sort_values("anio")
            filas.append({
                "indicador": ind, "anio": anio, "valor": valor, "unidad": unidad_de(catalogo, panel, ind),
                "puesto": puesto, "de": len(corte), "mediana_departamental": float(corte.median()),
                "total_nacional": float(nac.iloc[0]) if len(nac) else np.nan,
                "variacion": valor - float(previo["valor"].iloc[-1]) if len(previo) else np.nan,
                "mayor_mejor": mayor_mejor,
            })
        if not filas:
            st.warning("Este departamento no tiene datos en los indicadores clave.")
        else:
            columnas = st.columns(3)
            for i, f in enumerate(filas):
                dec = 3 if abs(f["valor"]) < 1 else 2
                delta = None if np.isnan(f["variacion"]) else f"{f['variacion']:+,.{dec}f} vs. dato previo"
                columnas[i % 3].metric(f"{etiqueta(f['indicador'])} · {f['anio']}", fmt_num(f["valor"]), delta,
                                       delta_color="normal" if f["mayor_mejor"] else "inverse",
                                       help=f"Unidad: {f['unidad'] or 'n/d'} · puesto {f['puesto']} de {f['de']} "
                                            "(1 = situación más favorable)")
            resumen = pd.DataFrame(filas)
            resumen["indicador"] = resumen["indicador"].map(etiqueta)
            resumen["puesto"] = resumen["puesto"].astype(str) + " / " + resumen["de"].astype(str)
            st.markdown("**Resumen ejecutivo**")
            vista = resumen[["indicador", "anio", "valor", "unidad", "puesto", "mediana_departamental",
                             "total_nacional"]].round(3)
            mostrar_tabla(vista)
            st.caption("Puesto 1 = situación más favorable según la polaridad del indicador. GEIH, pobreza monetaria "
                       "y Gini se comparan solo entre los departamentos cubiertos por la encuesta.")

            mejores = [f for f in filas if f["de"] >= 4 and f["puesto"] <= max(1, f["de"] // 4)]
            rezagos = [f for f in filas if f["de"] >= 4 and f["puesto"] > f["de"] - max(1, f["de"] // 4)]
            a, b = st.columns(2)
            a.markdown("**Fortalezas relativas (cuartil superior)**")
            a.markdown("\n".join(f"- {etiqueta(f['indicador'])}: puesto {f['puesto']} de {f['de']}" for f in mejores)
                       or "_Sin indicadores en el cuartil superior._")
            b.markdown("**Rezagos relativos (cuartil inferior)**")
            b.markdown("\n".join(f"- {etiqueta(f['indicador'])}: puesto {f['puesto']} de {f['de']}" for f in rezagos)
                       or "_Sin indicadores en el cuartil inferior._")

            ind_sel = st.selectbox("Trayectoria del indicador", [f["indicador"] for f in filas], format_func=etiqueta,
                                   key="fi_ind")
            tray = panel[(panel["indicador"] == ind_sel) & (panel["desagregacion"] == "Total") &
                         panel["departamento_canonico"].isin([dep, TOTAL_NACIONAL])].dropna(subset=["valor"])
            mediana = (deptos[(deptos["indicador"] == ind_sel) & (deptos["desagregacion"] == "Total")]
                       .dropna(subset=["valor"]).groupby("anio", as_index=False)["valor"].median())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mediana["anio"], y=mediana["valor"], name="Mediana departamental", mode="lines",
                                     line=dict(color=COLOR_NEUTRO, dash="dot")))
            nac = tray[tray["departamento_canonico"] == TOTAL_NACIONAL].sort_values("anio")
            if not nac.empty:
                fig.add_trace(go.Scatter(x=nac["anio"], y=nac["valor"], name=TOTAL_NACIONAL, mode="lines",
                                         line=dict(color=COLOR_SECUNDARIO, dash="dash")))
            prop = tray[tray["departamento_canonico"] == dep].sort_values("anio")
            fig.add_trace(go.Scatter(x=prop["anio"], y=prop["valor"], name=dep, mode="lines+markers",
                                     line=dict(color=COLOR_PRINCIPAL, width=3)))
            fig.update_layout(title=f"{etiqueta(ind_sel)} · {dep}", xaxis_title="Año",
                              yaxis_title=unidad_de(catalogo, panel, ind_sel) or "valor", hovermode="x unified")
            mostrar_fig(fig, alto=440)
            completos = panel[panel["departamento_canonico"] == dep]
            d1, d2 = st.columns(2)
            d1.download_button("Descargar ficha (CSV)", a_csv(vista), f"ficha_{norm_txt(dep).replace(' ', '_')}.csv", "text/csv")
            d2.download_button("Descargar todos los datos del departamento (CSV)", a_csv(completos),
                               f"datos_{norm_txt(dep).replace(' ', '_')}.csv", "text/csv")

    with tab_desc:
        indicadores = sorted(panel["indicador"].unique().tolist())
        territorios = sorted(panel["departamento_canonico"].unique().tolist())
        c1, c2 = st.columns(2)
        sel_ind = c1.multiselect("Indicadores (vacío = todos)", indicadores, format_func=etiqueta, key="dl_ind")
        sel_ter = c2.multiselect("Territorios (vacío = todos)", territorios, key="dl_ter")
        a_min, a_max = int(panel["anio"].min()), int(panel["anio"].max())
        c3, c4, c5 = st.columns([2, 1.5, 1])
        rango = c3.slider("Años", a_min, a_max, (a_min, a_max), key="dl_anios") if a_max > a_min else (a_min, a_max)
        sel_des = c4.multiselect("Desagregaciones (vacío = todas)", sorted(panel["desagregacion"].unique().tolist()),
                                 key="dl_des")
        formato = c5.radio("Formato", ["Largo", "Ancho"], key="dl_fmt")
        filtro = panel[(panel["anio"] >= rango[0]) & (panel["anio"] <= rango[1])]
        if sel_ind:
            filtro = filtro[filtro["indicador"].isin(sel_ind)]
        if sel_ter:
            filtro = filtro[filtro["departamento_canonico"].isin(sel_ter)]
        if sel_des:
            filtro = filtro[filtro["desagregacion"].isin(sel_des)]
        if filtro.empty:
            st.warning("Ningún registro coincide con los filtros seleccionados.")
            return
        if formato == "Ancho":
            salida = filtro.pivot_table(index=["departamento_canonico", "codigo_divipola", "anio", "desagregacion"],
                                        columns="indicador", values="valor", aggfunc="mean").reset_index()
            salida.columns.name = None
        else:
            orden = [c for c in ["departamento_canonico", "codigo_divipola", "anio", "indicador", "desagregacion",
                                 "valor", "unidad", "fuente", "marca_calidad"] if c in filtro.columns]
            salida = filtro[orden].sort_values(["indicador", "departamento_canonico", "anio"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Filas", f"{len(salida):,}")
        m2.metric("Indicadores", filtro["indicador"].nunique())
        m3.metric("Territorios", filtro["departamento_canonico"].nunique())
        mostrar_tabla(salida.head(200), alto=380)
        st.caption("Vista previa de las primeras 200 filas. El CSV usa codificación UTF-8 con BOM (compatible con Excel).")
        st.download_button("Descargar selección (CSV)", a_csv(salida), "observatorio_seleccion.csv", "text/csv",
                           type="primary")


# =============================================================================
# 5. APLICACIÓN PRINCIPAL
# =============================================================================
def main() -> None:
    inyectar_css()
    st.sidebar.markdown("## 📊 Observatorio\n**Indicadores socioeconómicos de Colombia**")
    pagina = st.sidebar.radio("Módulo analítico", PAGINAS, key="nav")
    st.sidebar.divider()

    ruta_panel = _buscar_archivo(ARCHIVO_PANEL)
    try:
        if ruta_panel is not None:
            panel = cargar_panel(str(ruta_panel))
        else:
            st.warning(f"No se encontró `data/{ARCHIVO_PANEL}`. Cárgalo manualmente para continuar.")
            subido = st.file_uploader("Panel del observatorio (.parquet)", type=["parquet"])
            if subido is None:
                st.stop()
            panel = cargar_panel(None, subido.getvalue())
    except Exception as e:
        st.error(f"No fue posible leer el panel: {e}")
        st.stop()
    if panel.empty:
        st.error("El panel está vacío después de la limpieza.")
        st.stop()

    ruta_cat = _buscar_archivo(ARCHIVO_CATALOGO)
    catalogo, reconstruido = cargar_catalogo(str(ruta_cat) if ruta_cat else None, panel)
    deptos = panel[panel["departamento_canonico"] != TOTAL_NACIONAL]

    st.sidebar.caption(
        f"{len(panel):,} observaciones · {panel['indicador'].nunique()} indicadores · "
        f"{int(panel['anio'].min())}–{int(panel['anio'].max())}"
    )
    faltantes = [n for n, ok in [("statsmodels", HAY_STATSMODELS), ("shap", HAY_SHAP),
                                 ("libpysal + esda", HAY_PYSAL)] if not ok]
    if faltantes:
        st.sidebar.caption("Librerías opcionales no instaladas: " + ", ".join(faltantes) + ".")

    if pagina == PAGINAS[0]:
        pagina_inicio(panel, deptos, catalogo, reconstruido)
    elif pagina == PAGINAS[1]:
        pagina_explorador(panel, deptos, catalogo)
    elif pagina == PAGINAS[2]:
        pagina_eda_pca(deptos)
    elif pagina == PAGINAS[3]:
        pagina_espacial(panel, deptos, catalogo)
    elif pagina == PAGINAS[4]:
        pagina_series(panel, catalogo)
    elif pagina == PAGINAS[5]:
        pagina_ml(deptos)
    elif pagina == PAGINAS[6]:
        pagina_simulador(deptos, catalogo, panel)
    else:
        pagina_ficha(panel, deptos, catalogo)


main()
