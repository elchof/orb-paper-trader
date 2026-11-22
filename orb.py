import numpy as np
import pandas as pd
from datetime import time as hora_t

PUNTO_USD = 2.0        # valor del punto de MNQ por contrato
COSTE_LADO_USD = 0.60  # comision 0,25 + tasa CME 0,35 por lado (cifras de IBKR segun Perplexity)


def agregar_15m(m1):
    # m1: velas de 1 minuto con columnas open, high, low, close e indice de tiempo
    v = m1.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return v.dropna()


def buscar_entrada(sesion, ema_sesion, r_alto, r_bajo):
    # primera vela cuyo cierre rompe el rango a favor de la EMA. Devuelve (posicion, hora, lado, entrada, stop, R) o None
    for i, (t, v) in enumerate(sesion.iterrows()):
        c = v["close"]
        e = ema_sesion.loc[t]
        if np.isnan(e):
            continue
        if c > r_alto and c > e:
            lado = 1
        elif c < r_bajo and c < e:
            lado = -1
        else:
            continue
        stop = v["low"] if lado == 1 else v["high"]
        R = abs(c - stop)
        if R == 0:
            continue
        return i, t, lado, c, stop, R
    return None


def buscar_salida(posteriores, lado, stop, objetivo):
    # devuelve (precio, motivo) o None si todavia no se toco ni el stop ni el objetivo
    for _, v in posteriores.iterrows():
        if lado == 1:
            if v["low"] <= stop:        # si toca stop y objetivo en la misma vela, cuenta el stop
                return stop, "stop"
            if v["high"] >= objetivo:
                return objetivo, "objetivo"
        else:
            if v["high"] >= stop:
                return stop, "stop"
            if v["low"] <= objetivo:
                return objetivo, "objetivo"
    return None


def operaciones_orb(df, objetivo_r=2.0, slippage_puntos=0.0, ema_span=200):
    # df: velas de 15m con indice con zona horaria America/New_York
    ema = df["close"].ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    filas = []
    for dia, d in df.groupby(df.index.date):
        rango = d.between_time("09:15", "09:15")   # la vela que va de 9:15 a 9:30
        if len(rango) != 1:
            continue
        r_alto, r_bajo = rango["high"].iloc[0], rango["low"].iloc[0]
        sesion = d.between_time("09:30", "15:45")  # velas que empiezan entre las 9:30 y las 15:45
        if sesion.empty:
            continue
        hallada = buscar_entrada(sesion, ema.reindex(sesion.index), r_alto, r_bajo)
        if hallada is None:
            continue
        i, _, lado, px, stop, R = hallada
        objetivo = px + lado * objetivo_r * R
        res = buscar_salida(sesion.iloc[i + 1:], lado, stop, objetivo)
        salida, motivo = res if res is not None else (sesion["close"].iloc[-1], "fin_sesion")
        entrada_ef = px + lado * slippage_puntos
        salida_ef = salida - lado * slippage_puntos
        pnl_pts = lado * (salida_ef - entrada_ef)
        filas.append({
            "fecha": dia, "lado": lado, "entrada": px, "stop": stop, "objetivo": objetivo,
            "salida": salida, "motivo": motivo, "R_puntos": R, "pnl_puntos": pnl_pts,
            "neto_usd": pnl_pts * PUNTO_USD - 2 * COSTE_LADO_USD, "r_multiple": pnl_pts / R,
        })
    return pd.DataFrame(filas)


def evaluar_hoy(df, ahora, objetivo_r=2.0, ema_span=200):
    # df: velas de 15m ya cerradas (indice America/New_York); ahora: hora actual de Nueva York
    ema = df["close"].ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    out = {"ultima_vela": df.index[-1], "ultimo_cierre": df["close"].iloc[-1], "ema": ema.iloc[-1]}
    d = df[df.index.date == ahora.date()]
    if d.empty:
        out["estado"] = "SIN_DATOS_HOY"
        return out
    rango = d.between_time("09:15", "09:15")
    if len(rango) != 1:
        out["estado"] = "RANGO_PENDIENTE" if ahora.time() < hora_t(9, 30) else "SIN_VELA_RANGO"
        return out
    r_alto, r_bajo = rango["high"].iloc[0], rango["low"].iloc[0]
    out.update(r_alto=r_alto, r_bajo=r_bajo)
    terminada = d.index[-1].time() >= hora_t(15, 45)   # ya esta cerrada la ultima vela de la sesion
    sesion = d.between_time("09:30", "15:45")
    hallada = buscar_entrada(sesion, ema.reindex(sesion.index), r_alto, r_bajo) if not sesion.empty else None
    if hallada is None:
        out["estado"] = "SESION_TERMINADA_SIN_ENTRADA" if terminada else "SIN_ENTRADA"
        return out
    i, t, lado, px, stop, R = hallada
    objetivo = px + lado * objetivo_r * R
    out.update(hora_entrada=t, lado=lado, entrada=px, stop=stop, objetivo=objetivo, R=R)
    res = buscar_salida(sesion.iloc[i + 1:], lado, stop, objetivo)
    if res is not None:
        out["estado"] = "CERRADA"
        out["salida"], out["motivo"] = res
    elif terminada:
        out["estado"] = "CERRADA"
        out["salida"], out["motivo"] = sesion["close"].iloc[-1], "fin_sesion"
    else:
        out["estado"] = "ENTRADA_AHORA" if t == d.index[-1] else "ABIERTA"
    return out


def resumen(t):
    if t.empty:
        return {"operaciones": 0}
    gan = t["neto_usd"] > 0
    pos = t.loc[gan, "neto_usd"].sum()
    neg = -t.loc[~gan, "neto_usd"].sum()
    equity = t["neto_usd"].cumsum()
    return {
        "operaciones": len(t),
        "largos": int((t["lado"] == 1).sum()),
        "cortos": int((t["lado"] == -1).sum()),
        "ganadoras_%": 100 * gan.mean(),
        "objetivo_%": 100 * (t["motivo"] == "objetivo").mean(),
        "stop_%": 100 * (t["motivo"] == "stop").mean(),
        "fin_sesion_%": 100 * (t["motivo"] == "fin_sesion").mean(),
        "r_medio": t["r_multiple"].mean(),
        "neto_usd_total": t["neto_usd"].sum(),
        "neto_usd_por_operacion": t["neto_usd"].mean(),
        "factor_beneficio": pos / neg if neg > 0 else float("inf"),
        "drawdown_max_usd": (equity - equity.cummax()).min(),
    }
