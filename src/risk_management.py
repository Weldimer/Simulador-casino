# ============================================================
# MOTOR DE GESTIÓN DE RIESGOS Y CIRCUIT BREAKER (V7)
# Electrónica PERLACO - Simulador de Casino
# ============================================================

class RiskManagementEngine:
    def __init__(self, initial_balance=100.0, max_loss_pct=0.25, target_win_pct=0.30, max_allowed_streak=5):
        self.initial_balance = float(initial_balance)
        self.current_balance = self.initial_balance
        self.max_loss_limit = self.initial_balance * (1 - max_loss_pct)
        self.target_win_limit = self.initial_balance * (1 + target_win_pct)
        self.max_allowed_streak = max_allowed_streak
        self.loss_streak = 0
        self.status = "ACTIVE"
        self.audit_log = []

    def evaluate_spin(self, stake, win):
        """Evalúa los guardrails de riesgo en cada tirada individual."""
        if self.status != "ACTIVE":
            return {"status": self.status, "action": "HALT"}

        if win:
            self.current_balance += stake
            self.loss_streak = 0
        else:
            self.current_balance -= stake
            self.loss_streak += 1

        # Comprobar Stop-Loss absoluto
        if self.current_balance <= self.max_loss_limit:
            self.status = "STOP_LOSS_TRIGGERED"
            action = "HALT"
        # Comprobar Stop-Win
        elif self.current_balance >= self.target_win_limit:
            self.status = "STOP_WIN_TRIGGERED"
            action = "HALT"
        # Comprobar racha crítica de derrotas (Circuit Breaker)
        elif self.loss_streak >= self.max_allowed_streak:
            self.status = "CIRCUIT_BREAKER_ACTIVE"
            action = "WARNING_REDUCE_EXPOSURE"
        else:
            action = "CONTINUE"

        log_entry = {
            "balance": self.current_balance,
            "loss_streak": self.loss_streak,
            "status": self.status,
            "action": action
        }
        self.audit_log.append(log_entry)
        return log_entry

if __name__ == "__main__":
    print("Módulo de Riesgos cargado correctamente.")