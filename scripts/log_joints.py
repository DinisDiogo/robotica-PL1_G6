import time
import csv
from pathlib import Path
import yaml
from lite6.session import safe_session, check

# 1. Carregar parâmetros a partir de config/robot.yaml (sem IP hardcoded)
config_path = Path(__file__).resolve().parent.parent / "config" / "robot.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ip = cfg["robot"]["ip"]
boundary = cfg["robot"]["reduced_mode"]["tcp_boundary_mm"]
payload_kg = cfg["tool"]["payload_kg"]

# 2. Iniciar sessão segura através de lite6.session
arm = safe_session(ip, boundary, payload_kg)

try:
    print("Atenção: Pede ao teu colega para ativar o Modo Manual (ensino) no Studio.")
    print("A gravação arranca em 5 segundos...")
    time.sleep(5)

    print("A GRAVAR DURANTE 30s! Movam o braço livremente...")

    t0 = time.perf_counter()
    rows = []

    # 3. Ciclo de leitura à cadência máxima
    while time.perf_counter() - t0 < 30:
        code, ang = arm.get_servo_angle()
        check(code, "get_servo_angle")
        rows.append([time.perf_counter() - t0, *ang[:6]])

    # 4. Guardar os resultados em CSV
    with open("registo_juntas.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tempo", "j1", "j2", "j3", "j4", "j5", "j6"])
        writer.writerows(rows)

    # 5. Calcular e reportar a cadência real
    amostras = len(rows)
    cadencia = amostras / 30.0
    print(f"Feito! CSV guardado com {amostras} leituras.")
    print(f"Cadência real obtida: {cadencia:.2f} amostras/s")

finally:
    arm.disconnect()
    print("Robô desligado em segurança.")