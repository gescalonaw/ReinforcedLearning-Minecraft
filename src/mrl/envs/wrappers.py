"""Wrappers de MineRL compartidos por los 4 agentes.

Los cuatro agentes ven exactamente el mismo entorno; lo único que cambia es la
representación de la cámara:

    mode="discrete" -> DiscreteCamera: acción = (combo de teclas, giro fijo de cámara)
                       action_space = MultiDiscrete([8, 9])
    mode="hybrid"   -> HybridCamera:   acción = (combo de teclas, cámara continua)
                       action_space = Tuple(Discrete(8), Box(-1, 1, (2,)))

Para los agentes de Tianshou (PPO y SAC discretos) se usa `make_agent_env`, que
además aplana:
    - la acción discreta a un solo índice: Discrete(72) = 8 combos de teclas x 9 giros
    - la observación a un solo arreglo uint8 (C, 64, 64)

Pila de wrappers (de adentro hacia afuera):
    MineRL -> ActionRepeat -> PovStack -> DiscreteCamera | HybridCamera
           [-> FlatDiscreteAction -> PovArray   (solo make_agent_env)]
"""
import sys
import time
from collections import deque

import gym
import numpy as np

# Teclas que usa el agente. 2^3 = 8 combinaciones posibles.
KEYS = ("forward", "jump", "attack")
N_KEY_COMBOS = 2 ** len(KEYS)

# Cámara discreta: 9 giros = pitch {-1, 0, +1} x yaw {-1, 0, +1} (multiplicado por camera_deg).
# En MineRL la cámara es [pitch, yaw] en grados (pitch = vertical, yaw = horizontal).
CAMERA_BINS = [(p, y) for p in (-1, 0, 1) for y in (-1, 0, 1)]
N_DISCRETE_ACTIONS = N_KEY_COMBOS * len(CAMERA_BINS)  # 72

TASKS = {
    "treechop": "MineRLTreechop-v0",
    "navigate": "MineRLNavigateDense-v0",
}


def key_combo(index):
    """Índice 0..7 -> {'forward': 0/1, 'jump': 0/1, 'attack': 0/1}."""
    return {k: (int(index) >> i) & 1 for i, k in enumerate(KEYS)}


def to_minerl_action(noop_space, keys, camera):
    """Construye el diccionario de acción que espera MineRL (resto de teclas en 0)."""
    action = noop_space.noop()
    for k, v in keys.items():
        action[k] = v
    action["camera"] = np.asarray(camera, dtype=np.float32)
    return action


class ActionRepeat(gym.Wrapper):
    """Repite cada acción `repeat` frames y suma la recompensa.

    Las teclas se mantienen apretadas todos los frames, pero la cámara se aplica
    solo en el primero: así un giro de 15° es un giro de 15°, no de 60°.

    Si MineRL pierde la conexión con Minecraft (info contiene "error"), termina el
    episodio y lo marca con info["minerl_error"] = True para poder contarlo.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        camera = np.asarray(action["camera"], dtype=np.float32)
        total_reward = 0.0
        for i in range(self.repeat):
            sub_action = dict(action)
            sub_action["camera"] = camera if i == 0 else np.zeros(2, dtype=np.float32)
            obs, reward, done, info = self.env.step(sub_action)
            total_reward += reward
            if "error" in info:
                info = dict(info)
                info["minerl_error"] = True
                done = True
            if done:
                break
        return obs, total_reward, done, info


def _compass(obs):
    """Ángulo de la brújula normalizado a [-1, 1] (solo en Navigate)."""
    if "compass" in obs:
        c = obs["compass"]
        angle = c["angle"] if isinstance(c, dict) else c
    elif "compassAngle" in obs:
        angle = obs["compassAngle"]
    else:
        return 0.0
    return float(np.asarray(angle).reshape(-1)[0]) / 180.0


class PovStack(gym.ObservationWrapper):
    """Observación = últimos `n_frames` frames apilados + vector de extras.

    pov: uint8 (3 * n_frames, 64, 64), canales primero (como espera PyTorch).
    vec: float32 (1,) con la brújula si use_compass, si no (0,).
    """

    def __init__(self, env, n_frames=4, use_compass=False):
        super().__init__(env)
        self.n_frames = n_frames
        self.use_compass = use_compass
        self.frames = deque(maxlen=n_frames)
        n_vec = 1 if use_compass else 0
        self.observation_space = gym.spaces.Dict({
            "pov": gym.spaces.Box(0, 255, (3 * n_frames, 64, 64), dtype=np.uint8),
            "vec": gym.spaces.Box(-1.0, 1.0, (n_vec,), dtype=np.float32),
        })

    @staticmethod
    def _frame(obs):
        return np.transpose(obs["pov"], (2, 0, 1)).astype(np.uint8)  # HWC -> CHW

    def _build(self, obs):
        vec = np.array([_compass(obs)], dtype=np.float32) if self.use_compass \
            else np.zeros(0, dtype=np.float32)
        return {"pov": np.concatenate(list(self.frames), axis=0), "vec": vec}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        frame = self._frame(obs)
        for _ in range(self.n_frames):
            self.frames.append(frame)
        return self._build(obs)

    def observation(self, obs):
        self.frames.append(self._frame(obs))
        return self._build(obs)


class DiscreteCamera(gym.ActionWrapper):
    """Acción = [combo de teclas (0..7), giro de cámara (0..8)]."""

    def __init__(self, env, camera_deg=15.0):
        super().__init__(env)
        self._noop_space = env.action_space
        self.camera_deg = float(camera_deg)
        self.action_space = gym.spaces.MultiDiscrete([N_KEY_COMBOS, len(CAMERA_BINS)])

    def action(self, act):
        keys = key_combo(act[0])
        camera = np.array(CAMERA_BINS[int(act[1])], dtype=np.float32) * self.camera_deg
        return to_minerl_action(self._noop_space, keys, camera)


class HybridCamera(gym.ActionWrapper):
    """Acción = (combo de teclas (0..7), cámara continua en [-1, 1]^2).

    La cámara se escala a [-max_camera_deg, max_camera_deg] grados.
    """

    def __init__(self, env, max_camera_deg=15.0):
        super().__init__(env)
        self._noop_space = env.action_space
        self.max_camera_deg = float(max_camera_deg)
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(N_KEY_COMBOS),
            gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        ))

    def action(self, act):
        key_index, camera = act
        keys = key_combo(key_index)
        camera = np.clip(np.asarray(camera, dtype=np.float32), -1.0, 1.0) * self.max_camera_deg
        return to_minerl_action(self._noop_space, keys, camera)


class FlatDiscreteAction(gym.ActionWrapper):
    """Discrete(72) -> [combo de teclas, giro de cámara] para DiscreteCamera.

    índice = combo * 9 + giro. Así PPO y SAC discretos eligen una sola acción
    entre 72 opciones (lo que esperan las implementaciones estándar).
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.n_camera = int(env.action_space.nvec[1])
        self.action_space = gym.spaces.Discrete(int(np.prod(env.action_space.nvec)))

    def action(self, act):
        act = int(np.asarray(act).reshape(-1)[0])
        return np.array([act // self.n_camera, act % self.n_camera])


class PovArray(gym.ObservationWrapper):
    """Dict(pov, vec) -> un solo arreglo uint8 (C, 64, 64).

    Si hay brújula (Navigate), se agrega como un canal extra constante con el
    ángulo escalado de [-1, 1] a [0, 255]. Así la red es la misma en ambas tareas.
    """

    def __init__(self, env):
        super().__init__(env)
        pov_space = env.observation_space["pov"]
        self.n_vec = env.observation_space["vec"].shape[0]
        channels = pov_space.shape[0] + self.n_vec
        self.observation_space = gym.spaces.Box(0, 255, (channels, 64, 64), dtype=np.uint8)

    def observation(self, obs):
        if self.n_vec == 0:
            return obs["pov"]
        extra = np.empty((self.n_vec, 64, 64), dtype=np.uint8)
        for i, v in enumerate(obs["vec"]):
            extra[i].fill(int(round((float(v) + 1.0) * 127.5)))
        return np.concatenate([obs["pov"], extra], axis=0)


class AutoRestart(gym.Wrapper):
    """Si Minecraft se cae (BrokenPipe, ConnectionReset, etc.), lo vuelve a abrir.

    MineRL 0.4.4 a veces pierde la conexión con Minecraft en entrenamientos largos.
    Sin esto el entrenamiento completo se cae; con esto se cierra el episodio actual
    (queda marcado minerl_error=1 en episodes.csv), se abre un Minecraft nuevo y se sigue.

    make_fn: función sin argumentos que crea el entorno completo.
    """

    def __init__(self, make_fn, max_restarts=50, wait_s=10):
        super().__init__(make_fn())
        self.make_fn = make_fn
        self.max_restarts = max_restarts
        self.wait_s = wait_s
        self.restarts = 0
        self._last_obs = None

    def _rebuild(self, error):
        self.restarts += 1
        print(f"\n[AutoRestart] Minecraft se cayó ({type(error).__name__}: {error}). "
              f"Reiniciando ({self.restarts}/{self.max_restarts})...", file=sys.stderr, flush=True)
        if self.restarts > self.max_restarts:
            raise RuntimeError("Demasiados reinicios de Minecraft") from error
        try:
            self.env.close()
        except Exception:
            pass
        time.sleep(self.wait_s)
        self.env = self.make_fn()

    def reset(self, **kwargs):
        for _ in range(5):
            try:
                self._last_obs = self.env.reset(**kwargs)
                return self._last_obs
            except Exception as e:  # noqa: BLE001  (MineRL lanza errores de socket variados)
                self._rebuild(e)
        raise RuntimeError("No se pudo reiniciar Minecraft después de 5 intentos")

    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)
        except Exception as e:  # noqa: BLE001
            self._rebuild(e)
            return self._last_obs, 0.0, True, {"minerl_error": True}
        self._last_obs = obs
        return obs, reward, done, info


def make_env(task="treechop", mode="discrete", action_repeat=4, n_frames=4,
             camera_deg=15.0, fake=False):
    """Crea el entorno con observación Dict y acción discreta (MultiDiscrete) o híbrida.

    task: "treechop" | "navigate"
    mode: "discrete" | "hybrid"
    camera_deg: tamaño del giro discreto / giro máximo continuo (igual en ambos modos).
    fake: usa un Minecraft simulado (sin MineRL) para probar el código en segundos.
    """
    if fake:
        from .fake import FakeMineRL
        env = FakeMineRL(compass=(task == "navigate"))
    else:
        import minerl  # noqa: F401  (registra los entornos en gym)
        env = gym.make(TASKS[task])
    env = ActionRepeat(env, repeat=action_repeat)
    env = PovStack(env, n_frames=n_frames, use_compass=(task == "navigate"))
    if mode == "discrete":
        env = DiscreteCamera(env, camera_deg=camera_deg)
    elif mode == "hybrid":
        env = HybridCamera(env, max_camera_deg=camera_deg)
    else:
        raise ValueError(f"mode desconocido: {mode}")
    return env


def make_agent_env(task="treechop", action_repeat=4, n_frames=4, camera_deg=15.0,
                   fake=False, seed=None):
    """Entorno para los agentes discretos de Tianshou.

    Observación: Box uint8 (C, 64, 64). Acción: Discrete(72).
    """
    env = make_env(task, "discrete", action_repeat=action_repeat, n_frames=n_frames,
                   camera_deg=camera_deg, fake=fake)
    env = FlatDiscreteAction(env)
    env = PovArray(env)
    if seed is not None:
        env.seed(seed)
    return env
