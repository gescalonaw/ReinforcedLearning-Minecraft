"""Entrena PPO o SAC discreto (Tianshou) en MineRL con cámara discreta.

Ejemplos:
    # Prueba rápida sin Minecraft (segundos): verifica que todo el código corre
    python scripts/train.py --algo ppo --dry-run
    python scripts/train.py --algo sac --dry-run

    # Entrenamiento real
    python scripts/train.py --algo ppo --task treechop --seed 0
    python scripts/train.py --algo sac --task treechop --seed 0 --steps 100000

    # Retomar un entrenamiento cortado (mismo comando + --resume)
    python scripts/train.py --algo ppo --task treechop --seed 0 --resume

    # Cambiar un hiperparámetro
    python scripts/train.py --algo ppo --hp ent_coef=0.02 --hp lr=1e-4

Resultados en runs/<task>/<algo>_discrete/seed<N>/:
    config.json      configuración usada
    episodes.csv     un renglón por episodio de entrenamiento (retorno, largo, paso)
    checkpoint.pth   se guarda al final de cada época (para --resume)
    policy.pth       pesos finales (para evaluate.py)
    events.*         curvas para TensorBoard
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tianshou.data import Collector, VectorReplayBuffer  # noqa: E402
from tianshou.env import DummyVectorEnv, SubprocVectorEnv  # noqa: E402
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer  # noqa: E402
from tianshou.utils import TensorboardLogger  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from mrl.agents import BUILDERS, PRESETS, load_checkpoint, save_checkpoint  # noqa: E402
from mrl.envs import AutoRestart, make_agent_env  # noqa: E402


class EpisodeCSV(gym.Wrapper):
    """Anota cada episodio terminado en un CSV (retorno, largo, errores de MineRL)."""

    def __init__(self, env, path):
        super().__init__(env)
        self.path = path
        self.ret, self.len = 0.0, 0
        new = not os.path.exists(path)
        self.file = open(path, "a", newline="")
        self.writer = csv.writer(self.file)
        if new:
            self.writer.writerow(["episode_return", "episode_length", "minerl_error", "wall_time"])
            self.file.flush()

    def reset(self, **kwargs):
        self.ret, self.len = 0.0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.ret += reward
        self.len += 1
        if done:
            self.writer.writerow([self.ret, self.len, int(info.get("minerl_error", False)),
                                  round(time.time(), 1)])
            self.file.flush()
        return obs, reward, done, info


def parse_hp(items):
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k] = float(v) if any(c in v for c in ".e") else int(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "sac"], required=True)
    p.add_argument("--task", choices=["treechop", "navigate"], default="treechop")
    p.add_argument("--steps", type=int, default=100_000, help="pasos del agente en total")
    p.add_argument("--step-per-epoch", type=int, default=10_000,
                   help="pasos por época (se guarda checkpoint al final de cada una)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--n-envs", type=int, default=1,
                   help="Minecraft en paralelo (cada uno usa ~3 GB de RAM)")
    p.add_argument("--threads", type=int, default=4, help="hilos de PyTorch")
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--camera-deg", type=float, default=15.0)
    p.add_argument("--preset", default=None,
                   help="configuración alternativa de mrl/agents.py (ej. SAC: estable)")
    p.add_argument("--hp", action="append", default=[], metavar="clave=valor",
                   help="sobrescribe un hiperparámetro (ver mrl/agents.py)")
    p.add_argument("--resume", action="store_true", help="retoma desde checkpoint.pth")
    p.add_argument("--dry-run", action="store_true",
                   help="Minecraft simulado y pocos pasos: solo prueba que el código corre")
    args = p.parse_args()

    if args.preset and args.preset not in PRESETS[args.algo]:
        sys.exit(f"Preset desconocido para {args.algo}: {args.preset}. "
                 f"Opciones: {list(PRESETS[args.algo])}")
    preset = PRESETS[args.algo].get(args.preset, {}) if args.preset else {}
    hp = {**preset, **parse_hp(args.hp)}
    if args.dry_run:
        args.steps, args.step_per_epoch = 600, 300
        hp = {**preset, **dict(step_per_collect=128, batch_size=32, buffer_size=2000,
                               start_steps=100), **parse_hp(args.hp)}
        if args.algo == "ppo":
            hp.pop("buffer_size"), hp.pop("start_steps")
        args.logdir = os.path.join(ROOT, "runs_dryrun")

    run_dir = os.path.join(args.logdir, args.task, f"{args.algo}_discrete" + (f"_{args.preset}" if args.preset else ""),
                           f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pth")
    if os.path.exists(ckpt_path) and not args.resume and not args.dry_run:
        sys.exit(f"Ya existe {ckpt_path}. Usa --resume para seguir, u otra --seed / --logdir.")

    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def env_fn(rank):
        def _make():
            env = AutoRestart(lambda: make_agent_env(
                args.task, action_repeat=args.action_repeat,
                camera_deg=args.camera_deg, fake=args.dry_run))
            return EpisodeCSV(env, os.path.join(run_dir, f"episodes{'' if rank == 0 else rank}.csv"))
        return _make

    probe = make_agent_env(args.task, fake=True)  # solo para leer los espacios (no abre Minecraft)
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    action_space = probe.action_space
    probe.close()

    VecEnv = DummyVectorEnv if args.n_envs == 1 else SubprocVectorEnv
    train_envs = VecEnv([env_fn(i) for i in range(args.n_envs)])

    policy, cfg = BUILDERS[args.algo](obs_shape, n_actions, action_space, device="cpu", **hp)

    if args.algo == "ppo":
        buffer = VectorReplayBuffer(cfg["step_per_collect"], args.n_envs)
    else:
        buffer = VectorReplayBuffer(cfg["buffer_size"], args.n_envs, ignore_obs_next=True)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=False)

    config = dict(vars(args), algo_hparams=cfg, obs_shape=list(obs_shape), n_actions=n_actions,
                  library="tianshou==0.4.8")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", json.dumps(config, indent=2))
    logger = TensorboardLogger(writer, train_interval=1, update_interval=100, save_interval=1)

    if args.resume:
        if not os.path.exists(ckpt_path):
            sys.exit(f"No hay checkpoint en {ckpt_path}")
        load_checkpoint(policy, ckpt_path)
        print(f"Retomando desde {ckpt_path}")

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        save_checkpoint(policy, ckpt_path, epoch=epoch, env_step=env_step,
                        gradient_step=gradient_step)
        return ckpt_path

    max_epoch = max(1, args.steps // args.step_per_epoch)
    print(f"[{args.algo.upper()} discreto] tarea={args.task} semilla={args.seed} "
          f"pasos={max_epoch * args.step_per_epoch} acciones={n_actions} obs={tuple(obs_shape)}")
    print(f"Resultados en {run_dir}")
    t0 = time.time()

    common = dict(
        policy=policy, train_collector=train_collector, test_collector=None,
        # +1: Tianshou 0.4.8 corta una época antes (compara epoch >= max_epoch).
        max_epoch=max_epoch + 1, step_per_epoch=args.step_per_epoch, episode_per_test=0,
        batch_size=cfg["batch_size"], step_per_collect=cfg["step_per_collect"],
        save_checkpoint_fn=save_checkpoint_fn, resume_from_log=args.resume,
        logger=logger, test_in_train=False,
    )
    if args.algo == "ppo":
        result = onpolicy_trainer(repeat_per_collect=cfg["repeat_per_collect"], **common)
    else:
        if not args.resume:
            # Exploración inicial: acciones al azar para llenar el buffer antes de aprender.
            train_collector.collect(n_step=cfg["start_steps"], random=True)
        result = offpolicy_trainer(update_per_step=cfg["update_per_step"], **common)

    torch.save(policy.state_dict(), os.path.join(run_dir, "policy.pth"))
    train_envs.close()
    mins = (time.time() - t0) / 60
    print(f"Listo en {mins:.1f} min. Pesos finales: {os.path.join(run_dir, 'policy.pth')}")
    print({k: v for k, v in result.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
