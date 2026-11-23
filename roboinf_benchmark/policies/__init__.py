from ..contracts import EEF, JOINT
from .adapters import Gr00t, LingBotVA, OpenWAM, Pi05

POLICY_NAMES = ("pi_05", "openwam", "gr00t", "lingbot-va")


def load_policy(name, config):
    if name not in POLICY_NAMES or not isinstance(config, dict):
        raise ValueError(f"Expected one of {POLICY_NAMES} and a JSON object")
    allowed = {"host", "port"} | ({"action_mode", "eef_layout"} if name == "openwam" else set())
    if config.keys() - allowed:
        raise ValueError(f"Unsupported {name} configuration fields")
    host = config.get("host", "localhost")
    port = config.get("port", dict(zip(POLICY_NAMES, (8000, 8848, 5555, 10001)))[name])
    if not isinstance(host, str) or not host or any(c in host for c in "/@?#"):
        raise ValueError("Expected a hostname without a URL or credentials")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Expected a valid port")
    if name == "pi_05":
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        return Pi05(WebsocketClientPolicy(host=host, port=port))
    if name == "gr00t":
        from gr00t.policy.server_client import PolicyClient
        return Gr00t(PolicyClient(host=host, port=port))
    if name == "openwam":
        from benchmarks.robotwin.openwam2robotwin_interface import ModelClient
        from benchmarks.utils import action_conversion
        mode = config.get("action_mode", "ee")
        layout = config.get("eef_layout", "physical")
        if mode not in ("ee", "qpos") or layout not in ("physical", "legacy_wxyz_values_as_xyzw"):
            raise ValueError("Unsupported OpenWAM action mode or pose representation")
        if mode == "qpos" and "eef_layout" in config:
            raise ValueError("eef_layout is only valid in ee mode")
        client = ModelClient(host=host, port=port, action_type=mode, send_state=True,
                             state_dim=20 if mode == "ee" else 14)
        return OpenWAM(client, EEF if mode == "ee" else JOINT, action_conversion, layout)
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
    return LingBotVA(WebsocketClientPolicy(host=host, port=port))
