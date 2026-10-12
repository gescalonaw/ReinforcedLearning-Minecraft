"""Verifica que MineRL funciona: corre acciones aleatorias, mide pasos/seg y graba un video.

Uso (con el entorno conda `minerl` activo):
    python scripts/check_env.py                         # Treechop, 1000 pasos, con ventana (WSLg)
    python scripts/check_env.py --env MineRLNavigateDense-v0 --steps 2000
    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/check_env.py   # sin ventana

El video queda en out/check_<entorno>.mp4 (lo que ve el agente, 64x64 ampliado x4).
"""
import argparse
import os
import time

import gym
import imageio
import minerl  # noqa: F401  (registra los entornos MineRL en gym)
import numpy as np


def random_action(env):
    action = env.action_space.noop()
    for key in ("forward", "jump", "attack"):
        if key in action:
            action[key] = int(np.random.rand() < 0.5)
    action["camera"] = np.random.normal(0.0, 5.0, size=2).astype(np.float32)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MineRLTreechop-v0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", default="out")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Creando {args.env} (Minecraft tarda un par de minutos en arrancar)...")
    env = gym.make(args.env)
    obs = env.reset()

    writer = None
    if not args.no_video:
        path = os.path.join(args.out, f"check_{args.env}.mp4")
        writer = imageio.get_writer(path, fps=20)

    total_reward, episodes = 0.0, 1
    start = time.time()
    for step in range(1, args.steps + 1):
        obs, reward, done, _ = env.step(random_action(env))
        total_reward += reward
        if writer is not None:
            writer.append_data(obs["pov"].repeat(4, axis=0).repeat(4, axis=1))
        if done:
            obs = env.reset()
            episodes += 1
        if step % 250 == 0:
            print(f"  paso {step}: {step / (time.time() - start):.1f} pasos/seg")

    elapsed = time.time() - start
    env.close()
    if writer is not None:
        writer.close()
        print(f"Video: {path}")
    print(f"\n{args.steps} pasos en {elapsed:.1f} s -> {args.steps / elapsed:.1f} pasos/seg")
    print(f"Episodios: {episodes} | recompensa total: {total_reward}")


if __name__ == "__main__":
    main()
