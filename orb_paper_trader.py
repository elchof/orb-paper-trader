import json
import os

import pandas as pd
import yfinance as yf

from orb import evaluar_hoy, PUNTO_USD, COSTE_LADO_USD

# ==============================================================================
# CONFIGURACION (ORB 9:15-9:30 NY | EMA 200 | objetivo 2R | 1 contrato MNQ)
# ==============================================================================
CONFIG = {
    "ticker": "NQ=F",
    "initial_capital": 100.0,   # Saldo inicial paper (USD)
    "slippage_puntos": 0.25,      # 1 tick por lado, igual que el backtest
    "dias_a_revisar": 5,          # Si falla alguna ejecucion, recupera los ultimos dias
    "state_file": "orb_state.json",
    "log_file": "orb_trades_log.csv",
}


# ==============================================================================
# ESTADO Y LOG
# ==============================================================================
def load_state(cfg):
    if os.path.exists(cfg["state_file"]):
        with open(cfg["state_file"], "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("balance", cfg["initial_capital"])
    data.setdefault("open", {})
    data.setdefault("logged", [])
    return data


def save_state(state, cfg):
    with open(cfg["state_file"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def append_closed_trade_csv(row, cfg):
    path = cfg["log_file"]
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))


# ==============================================================================
# DATOS (igual que command_center_orb.py)
# ==============================================================================
def descargar(ticker):
    d = yf.download(ticker, period="30d", interval="15m", progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if d.empty:
        return None
    if d.index.tz is None:
        raise SystemExit("[!] Los datos no traen zona horaria: no puedo situar las 9:15 de Nueva York.")
    d.index = d.index.tz_convert("America/New_York")
    return d


# ==============================================================================
# UNA PASADA
# ==============================================================================
def run_cycle(cfg):
    state = load_state(cfg)
    ahora = pd.Timestamp.now(tz="America/New_York")
    state.setdefault("start_date", str(ahora.date()))   # no se registran dias anteriores al primer arranque

    print("=" * 70)
    print(f"ORB PAPER TRADER | Saldo: ${state['balance']:,.2f} | NY: {ahora.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    d = descargar(cfg["ticker"])
    if d is None:
        print("  Yahoo no devolvio datos.")
        return
    d = d[d.index + pd.Timedelta(minutes=15) <= ahora]   # solo velas ya cerradas
    if d.empty:
        print("  No hay velas cerradas.")
        return

    for dia in sorted(set(d.index.date))[-cfg["dias_a_revisar"]:]:
        clave = str(dia)
        if clave < state["start_date"] or clave in state["logged"]:
            continue
        es_hoy = dia == ahora.date()
        if es_hoy:
            sub, ref = d, ahora
        else:
            sub = d[d.index.date <= dia]
            ref = pd.Timestamp(f"{dia} 23:59", tz="America/New_York")

        r = evaluar_hoy(sub, ref)
        estado = r["estado"]

        if estado == "CERRADA":
            lado, px, R = int(r["lado"]), float(r["entrada"]), float(r["R"])
            slip = cfg["slippage_puntos"]
            entrada_ef = px + lado * slip
            salida_ef = float(r["salida"]) - lado * slip
            pnl_pts = lado * (salida_ef - entrada_ef)
            neto = round(pnl_pts * PUNTO_USD - 2 * COSTE_LADO_USD, 2)
            state["balance"] = round(state["balance"] + neto, 2)
            append_closed_trade_csv({
                "fecha": clave,
                "lado": "LARGO" if lado == 1 else "CORTO",
                "hora_entrada": str(r["hora_entrada"]),
                "entrada": px,
                "stop": float(r["stop"]),
                "objetivo": float(r["objetivo"]),
                "salida": float(r["salida"]),
                "motivo": r["motivo"],
                "R_puntos": R,
                "r_multiple": round(pnl_pts / R, 3),
                "neto_usd": neto,
                "balance_after": state["balance"],
            }, cfg)
            state["logged"].append(clave)
            state["logged"] = state["logged"][-200:]
            state["open"].pop(clave, None)
            icono = "OK" if neto > 0 else "XX"
            print(f"  [{icono}] {clave} {'LARGO' if lado == 1 else 'CORTO'} cerrado por {r['motivo']} "
                  f"| Neto: {neto:+.2f} USD | Saldo: ${state['balance']:,.2f}")

        elif estado in ("ENTRADA_AHORA", "ABIERTA"):
            lado = int(r["lado"])
            state["open"][clave] = {
                "lado": "LARGO" if lado == 1 else "CORTO",
                "hora_entrada": str(r["hora_entrada"]),
                "entrada": float(r["entrada"]),
                "stop": float(r["stop"]),
                "objetivo": float(r["objetivo"]),
                "R_puntos": float(r["R"]),
            }
            print(f"  [ABIERTA] {clave} {state['open'][clave]['lado']} | Entrada {r['entrada']:.2f} "
                  f"| Stop {r['stop']:.2f} | Objetivo {r['objetivo']:.2f}")

        elif es_hoy:
            print(f"  [{estado}] {clave}")

    save_state(state, cfg)
    print("=" * 70)


if __name__ == "__main__":
    run_cycle(CONFIG)
