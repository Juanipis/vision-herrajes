"""Interface for controlling the PLC expellers ("expulsores").

This module wraps the snap7 client and exposes a plain Python class that can
be safely imported even if the PLC library or hardware are unavailable.

Simulation can be toggled with the module-level ``SIMULATION_ENABLED`` flag or
by passing ``simulate=True`` when building an :class:`Expulsores` instance.  In
simulation mode, PLC calls are replaced with log messages so the GUI keeps
working without hardware.
"""

from __future__ import annotations

from typing import Dict, Optional

import snap7  # type: ignore
from snap7.type import Areas  # type: ignore
from snap7.util import set_bool  # type: ignore


# Set this to False when the PLC is available.
SIMULATION_ENABLED = True

# Default PLC endpoint.
PLC_DEFAULT_IP = "192.168.0.4"

# Default family-to-motor mapping, kept at module level to document intent.
DEFAULT_LABEL_TO_MOTOR: Dict[str, int] = {
    "anillo": 2,
    "ocho": 1,
    "dobleanillo": 3,
    "gancho": 4,
    "CLASE_APAGADO_1": 0,
    "CLASE_APAGADO_2": 0,
    "CLASE_APAGADO_3": 0,
    "APAGAR": 0,
}


class Expulsores:
    """Control the PLC expellers (motors 1-4) or simulate them."""

    def __init__(
        self,
        plc_ip: str = PLC_DEFAULT_IP,
        rack: int = 0,
        slot: int = 1,
        mapping: Optional[Dict[str, int]] = None,
        *,
        simulate: Optional[bool] = None,
    ) -> None:
        self.PLC_IP = plc_ip
        self.RACK = rack
        self.SLOT = slot
        self.MAPEO_ETIQUETA_A_MOTOR = dict(mapping or DEFAULT_LABEL_TO_MOTOR)
        self._mapping_lower = {
            key.lower(): value for key, value in self.MAPEO_ETIQUETA_A_MOTOR.items()
        }

        requested_sim = SIMULATION_ENABLED if simulate is None else bool(simulate)
        self.simulate = requested_sim or snap7 is None
        self.connected = False
        self.client = None
        self._last_motor = 0

        if self.simulate:
            print("[Expulsores] Simulation enabled – PLC actions will be logged only")
            return

        self.client = snap7.client.Client()  # type: ignore[attr-defined]
        try:
            self.client.connect(self.PLC_IP, self.RACK, self.SLOT)
            self.connected = bool(self.client.get_connected())
            if self.connected:
                print(f"[Expulsores] Connected to PLC at {self.PLC_IP}")
            else:
                self._fallback_to_simulation("connection not established")
        except Exception as exc:  # pragma: no cover - hardware specific
            self._fallback_to_simulation(str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def activar_motor(self, motor_activo: int) -> bool:
        """Turn on a single motor (1-4) or all off when ``motor_activo`` is 0."""

        if self.simulate:
            self._last_motor = motor_activo
            print(f"[Expulsores] (sim) activar_motor -> {motor_activo}")
            return True

        if not self.connected or self.client is None:
            print("[Expulsores] PLC client not connected; skipping write")
            return False

        try:
            memory = self.client.read_area(Areas.MK, 0, 0, 1)  # type: ignore[arg-type]
            for bit in range(4):
                state = bit + 1 == motor_activo
                memory = set_bool(memory, 0, bit, state)  # type: ignore[misc]
            self.client.write_area(Areas.MK, 0, 0, memory)  # type: ignore[arg-type]
            self._last_motor = motor_activo
            return True
        except Exception as exc:  # pragma: no cover - hardware specific
            self._fallback_to_simulation(str(exc))
            return False

    def run_prediction(self, prediction_label: Optional[str]) -> int:
        """Map a prediction label to a motor index and activate it."""

        if prediction_label is None:
            self.activar_motor(0)
            return 0

        etiqueta = prediction_label.strip()
        motor = self._mapping_lower.get(etiqueta.lower())
        if motor is None:
            motor = self.MAPEO_ETIQUETA_A_MOTOR.get(etiqueta)

        if motor is None:
            print(f"[Expulsores] Unknown label '{etiqueta}', turning off motors")
            self.activar_motor(0)
            return -1

        self.activar_motor(motor)
        return motor

    def close(self) -> None:
        """Turn motors off and close the PLC client if necessary."""

        if self.simulate:
            self._last_motor = 0
            return

        try:
            if self.connected:
                self.activar_motor(0)
                self.client.disconnect()  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - hardware specific
            pass
        finally:
            self.connected = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fallback_to_simulation(self, reason: str) -> None:
        if self.simulate:
            return
        print(f"[Expulsores] Falling back to simulation mode: {reason}")
        self.simulate = True
        self.connected = False
        self.client = None

    def __enter__(self) -> "Expulsores":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = ["Expulsores", "SIMULATION_ENABLED", "PLC_DEFAULT_IP", "DEFAULT_LABEL_TO_MOTOR"]
