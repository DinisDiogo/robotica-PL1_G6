from xarm.wrapper import XArmAPI

def check(code, what):
    """Garante que qualquer comando que falhe para a execução imediatamente."""
    if code != 0:
        raise RuntimeError(f"{what} falhou: code={code}")

def safe_session(ip, boundary, tcp_load):
    """
    Inicia a ligação, limpa erros, aplica limites e devolve o objeto XArmAPI.
    Os valores (ip, boundary, tcp_load) vêm dos scripts que chamam esta função.
    """
    arm = XArmAPI(ip, is_radian=False)
    
    # 1. Limpar erros e ativar braço
    arm.clean_warn()
    arm.clean_error()
    check(arm.motion_enable(True), "motion_enable")
    
    # 2. Definir a carga para o braço não cair
    check(arm.set_tcp_load(tcp_load, [0, 0, 30]), "tcp_load")
    
    # 3. Aplicar rigorosamente as velocidades do Modo Reduzido
    arm.set_reduced_max_joint_speed(54)
    arm.set_reduced_max_tcp_speed(150)
    
    # 4. Aplicar a fronteira
    arm.set_reduced_tcp_boundary(boundary)
    
    # 5. Ativar modo reduzido e passar a Controlo de Posição (Modo 0, Estado 0)
    arm.set_reduced_mode(True)
    check(arm.set_mode(0), "set_mode")
    check(arm.set_state(0), "set_state")
    
    return arm
