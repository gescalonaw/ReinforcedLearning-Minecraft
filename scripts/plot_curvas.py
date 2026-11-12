"""Gráfico de curvas de entrenamiento: troncos por episodio y entropía de la política.

Lee los registros de TensorBoard de todas las corridas en runs/<tarea>/ y dibuja:
    arriba: recompensa por episodio (train/reward), suavizada
    abajo:  entropía de la política
            PPO: la registra directamente (update/loss/ent)
            SAC: se calcula a partir de α y de la pérdida de α:
                 loss_α = -log α · (H_objetivo - H)   =>   H = H_objetivo + loss_α / log α

Ejemplos:
    python scripts/plot_curvas.py
    python scripts/plot_curvas.py --task treechop --smooth 20 --out results/curvas_treechop.png
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = {"ppo_discrete": "#1f77b4", "sac_discrete": "#d62728",
          "sac_discrete_estable": "#ff7f0e"}


def scalars(acc, tag):
    if tag not in acc.Tags().get("scalars", []):
        return np.array([]), np.array([])
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev], float), np.array([e.value for e in ev], float)


def smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def grad_to_env(cfg):
    """Factor para pasar de 'pasos de gradiente' (eje de update/*) a pasos del entorno."""
    hp = cfg["algo_hparams"]
    if cfg["algo"] == "sac":
        return 1.0 / hp["update_per_step"]
    n_minibatch = math.ceil(hp["step_per_collect"] / hp["batch_size"])
    return hp["step_per_collect"] / (hp["repeat_per_collect"] * n_minibatch)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    p.add_argument("--task", default="treechop")
    p.add_argument("--smooth", type=int, default=10, help="ventana de suavizado (episodios / registros)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs, args.task, "*", "seed*"))
                      if os.path.exists(os.path.join(d, "config.json")))
    if not run_dirs:
        raise SystemExit(f"No hay corridas en {os.path.join(args.runs, args.task)}")

    fig, (ax_r, ax_h) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    styles = ["-", "--", ":", "-."]
    n_actions, targets = None, {}

    for d in run_dirs:
        variant, seed = d.split(os.sep)[-2], d.split(os.sep)[-1]
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        n_actions = cfg.get("n_actions", 72)
        acc = EventAccumulator(d, size_guidance={"scalars": 0})
        acc.Reload()
        color = COLORS.get(variant, None)
        ls = styles[int(seed.replace("seed", "")) % len(styles)]
        label = f"{variant} {seed}"

        x, y = scalars(acc, "train/reward")
        if len(y):
            ys = smooth(y, args.smooth)
            ax_r.plot(x[len(x) - len(ys):], ys, ls, color=color, label=label, lw=1.6)

        f = grad_to_env(cfg)
        if cfg["algo"] == "ppo":
            gx, h = scalars(acc, "update/loss/ent")
        else:
            gx, a = scalars(acc, "update/alpha")
            _, la = scalars(acc, "update/loss/alpha")
            n = min(len(a), len(la))
            gx, a, la = gx[:n], a[:n], la[:n]
            target = cfg["algo_hparams"]["target_entropy_scale"] * math.log(n_actions)
            targets[variant] = target
            log_a = np.log(np.clip(a, 1e-12, None))
            ok = np.abs(log_a) > 0.05          # cerca de α = 1 la fórmula no es confiable
            gx, h = gx[ok], target + la[ok] / log_a[ok]
            h = np.clip(h, 0, math.log(n_actions))
        if len(h):
            hs = smooth(h, args.smooth)
            ax_h.plot(gx[len(gx) - len(hs):] * f, hs, ls, color=color, label=label, lw=1.6)

    hmax = math.log(n_actions or 72)
    ax_h.axhline(hmax, color="gray", lw=1, ls=":")
    tr = ax_h.get_yaxis_transform()   # x en fracción del eje, y en datos
    ax_h.text(0.01, hmax + 0.05, f"máximo (al azar) = ln {n_actions} = {hmax:.2f}", color="gray", fontsize=9, transform=tr)
    for variant, t in targets.items():
        ax_h.axhline(t, color=COLORS.get(variant, "k"), lw=0.8, ls=":", alpha=0.7)
        ax_h.text(0.01, t + 0.05, f"objetivo {variant} = {t:.2f}", color=COLORS.get(variant, "k"), fontsize=9,
                  transform=tr)

    ax_r.set_ylabel("troncos por episodio")
    ax_r.set_title(f"MineRL {args.task}: entrenamiento (cámara discreta)")
    ax_r.grid(alpha=0.3)
    ax_r.legend(fontsize=8, ncol=2)
    ax_h.set_ylabel("entropía de la política (nats)")
    ax_h.set_xlabel("pasos del entorno")
    ax_h.set_ylim(0, hmax + 0.4)
    ax_h.grid(alpha=0.3)
    fig.tight_layout()

    out = args.out or os.path.join(ROOT, "results", f"curvas_{args.task}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Gráfico:", out)


if __name__ == "__main__":
    main()
