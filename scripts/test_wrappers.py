"""Prueba los wrappers con acciones al azar y graba un video corto.

    python scripts/test_wrappers.py --task treechop --mode discrete
    python scripts/test_wrappers.py --task navigate --mode hybrid --steps 100
    python scripts/test_wrappers.py --agent-env            # entorno de los agentes: Discrete(72)
    python scripts/test_wrappers.py --fake                 # sin Minecraft

Video en out/test_<task>_<mode>.mp4 (último frame, escalado x4).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from mrl.envs import make_agent_env, make_env  # noqa: E402


def last_frame(obs):
    pov = obs["pov"] if isinstance(obs, dict) else obs[:12]
    return np.transpose(pov[-3:], (1, 2, 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["treechop", "navigate"], default="treechop")
    p.add_argument("--mode", choices=["discrete", "hybrid"], default="discrete")
    p.add_argument("--agent-env", action="store_true", help="usa make_agent_env (Discrete(72))")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fake", action="store_true", help="Minecraft simulado")
    args = p.parse_args()

    if args.agent_env:
        env = make_agent_env(args.task, action_repeat=args.action_repeat, fake=args.fake)
        mode = "agent"
        print("Observación:", env.observation_space.shape, env.observation_space.dtype)
    else:
        env = make_env(args.task, args.mode, action_repeat=args.action_repeat, fake=args.fake)
        mode = args.mode
        print("Observación: pov", env.observation_space["pov"].shape,
              "vec", env.observation_space["vec"].shape)
    print("Acciones:", env.action_space)

    obs = env.reset()
    frames, total, t0 = [], 0.0, time.time()
    for t in range(args.steps):
        obs, reward, done, info = env.step(env.action_space.sample())
        total += reward
        if not args.no_video:
            frames.append(last_frame(obs))
        if done:
            print(f"Episodio terminado en el paso {t + 1}")
            obs = env.reset()
    dt = time.time() - t0
    env.close()

    print(f"Recompensa total: {total:.2f}")
    print(f"{args.steps / dt:.1f} pasos del agente/seg  "
          f"(~{args.steps * args.action_repeat / dt:.0f} frames/seg)")

    if frames:
        import imageio
        os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
        path = os.path.join(ROOT, "out", f"test_{args.task}_{mode}.mp4")
        imageio.mimsave(path, [np.kron(f, np.ones((4, 4, 1), dtype=np.uint8)) for f in frames], fps=10)
        print("Video:", path)


if __name__ == "__main__":
    main()
