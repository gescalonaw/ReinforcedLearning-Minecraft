"""Evalúa un agente entrenado en mundos fijos (los mismos para todos los agentes).

Por defecto 20 episodios con las semillas de mundo 1000..1019, así PPO y SAC se
comparan en exactamente los mismos mundos. Dos modos de elegir la acción:
    determinista (por defecto): siempre la acción más probable
    --stochastic:               se sortea según las probabilidades (como al entrenar)
Con poco entrenamiento la política es casi uniforme y el modo determinista suele
quedarse pegado repitiendo una sola acción; por eso conviene reportar ambos.

Ejemplos:
    python scripts/evaluate.py --run runs/treechop/ppo_discrete/seed0
    python scripts/evaluate.py --run runs/treechop/sac_discrete/seed0 --episodes 5 --video
    python scripts/evaluate.py --run runs/treechop/ppo_discrete/seed0 --stochastic
    python scripts/evaluate.py --random --task treechop          # línea base: agente al azar
    python scripts/evaluate.py --run runs_dryrun/treechop/ppo_discrete/seed0 --dry-run

Guarda en la carpeta del run:
    eval.csv / eval.json                        modo determinista
    eval_stochastic.csv / eval_stochastic.json  con --stochastic
    eval_ep0.mp4 (con --video) el primer episodio, para ver jugar al agente
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mrl.agents import BUILDERS, greedy_action, load_checkpoint, sample_action  # noqa: E402
from mrl.envs import make_agent_env  # noqa: E402

GOAL_REWARD = 100.0  # Navigate: llegar al diamante da +100 en un paso


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="carpeta del entrenamiento (con config.json y policy.pth)")
    p.add_argument("--weights", default=None, help="por defecto policy.pth, si no checkpoint.pth")
    p.add_argument("--random", action="store_true", help="evalúa un agente al azar (línea base)")
    p.add_argument("--task", choices=["treechop", "navigate"], default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="sortea la acción según la política en vez de tomar la más probable")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-offset", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=None, help="corta episodios largos (opcional)")
    p.add_argument("--video", action="store_true", help="graba el primer episodio")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto la del run)")
    p.add_argument("--dry-run", action="store_true", help="Minecraft simulado")
    args = p.parse_args()

    if not args.random and not args.run:
        p.error("indica --run <carpeta> o --random")

    cfg = {}
    if args.run:
        with open(os.path.join(args.run, "config.json")) as f:
            cfg = json.load(f)
    task = args.task or cfg.get("task", "treechop")
    env = make_agent_env(task, action_repeat=cfg.get("action_repeat", 4),
                         camera_deg=cfg.get("camera_deg", 15.0), fake=args.dry_run)

    policy = None
    if not args.random:
        policy, _ = BUILDERS[cfg["algo"]](env.observation_space.shape, env.action_space.n,
                                          env.action_space, device="cpu", **cfg.get("algo_hparams", {}))
        weights = args.weights or next(
            (os.path.join(args.run, n) for n in ("policy.pth", "checkpoint.pth")
             if os.path.exists(os.path.join(args.run, n))), None)
        if weights is None:
            sys.exit(f"No hay policy.pth ni checkpoint.pth en {args.run}")
        load_checkpoint(policy, weights, with_optims=False)
        policy.eval()
        print(f"Pesos: {weights}")

    torch.manual_seed(args.seed_offset)
    np.random.seed(args.seed_offset)
    mode = "random" if args.random else ("stochastic" if args.stochastic else "deterministic")
    name = "random" if args.random else f"{cfg['algo']}_discrete"
    suffix = "_stochastic" if args.stochastic and not args.random else ""
    out_dir = args.out or args.run or os.path.join(ROOT, "runs", task, "random")
    os.makedirs(out_dir, exist_ok=True)

    rows, frames = [], []
    for ep in range(args.episodes):
        world_seed = args.seed_offset + ep
        env.seed(world_seed)
        obs = env.reset()
        ret, length, success, done, info = 0.0, 0, False, False, {}
        counts = np.zeros(env.action_space.n, dtype=int)
        t0 = time.time()
        while not done:
            if args.random:
                act = env.action_space.sample()
            elif args.stochastic:
                act = sample_action(policy, torch.as_tensor(obs))
            else:
                act = greedy_action(policy, torch.as_tensor(obs))
            counts[int(act)] += 1
            obs, reward, done, info = env.step(act)
            ret += reward
            length += 1
            success |= reward >= GOAL_REWARD
            if args.video and ep == 0:
                frames.append(np.transpose(obs[-3:] if obs.shape[0] % 3 == 0 else obs[-4:-1],
                                           (1, 2, 0)))
            if args.max_steps and length >= args.max_steps:
                break
        row = dict(episode=ep, world_seed=world_seed, episode_return=round(ret, 3),
                   episode_length=length, success=int(success),
                   minerl_error=int(info.get("minerl_error", False)),
                   top_action=int(counts.argmax()),
                   top_action_frac=round(float(counts.max() / max(1, counts.sum())), 3),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(row)

    env.close()

    with open(os.path.join(out_dir, f"eval{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    returns = np.array([r["episode_return"] for r in rows])
    summary = dict(
        agent=name, mode=mode, task=task, episodes=len(rows),
        world_seeds=[r["world_seed"] for r in rows],
        mean_return=float(returns.mean()), std_return=float(returns.std()),
        median_return=float(np.median(returns)),
        mean_length=float(np.mean([r["episode_length"] for r in rows])),
        success_rate=float(np.mean([r["success"] for r in rows])),
        minerl_errors=int(sum(r["minerl_error"] for r in rows)),
        mean_top_action_frac=float(np.mean([r["top_action_frac"] for r in rows])),
    )
    with open(os.path.join(out_dir, f"eval{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "world_seeds"}, indent=2))

    if frames:
        import imageio
        big = [np.kron(fr, np.ones((4, 4, 1), dtype=np.uint8)) for fr in frames]
        path = os.path.join(out_dir, f"eval{suffix}_ep0.mp4")
        imageio.mimsave(path, big, fps=10)
        print(f"Video: {path}")


if __name__ == "__main__":
    main()
