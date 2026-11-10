"""PPO y SAC discretos, tomados de Tianshou 0.4.8 (no los implementamos nosotros).

    PPO:          tianshou.policy.PPOPolicy           (Schulman et al., 2017)
    SAC discreto: tianshou.policy.DiscreteSACPolicy   (Christodoulou, 2019)

Aquí solo se arman: la red (NatureCNN) + las cabezas actor/crítico de Tianshou +
los hiperparámetros. Ambos usan el mismo entorno (Discrete(72)) y el mismo codificador.

Los hiperparámetros por defecto siguen los ejemplos de Atari de Tianshou y CleanRL
(ver docs/ENTRENAMIENTO.md); se pueden cambiar desde scripts/train.py.
"""
import math

import torch
from tianshou.policy import DiscreteSACPolicy, PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

from .networks import NatureCNN

PPO_DEFAULTS = dict(
    lr=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    step_per_collect=1024,   # pasos que junta antes de cada actualización
    repeat_per_collect=4,    # épocas de PPO sobre esos datos
    batch_size=256,
)

SAC_DEFAULTS = dict(
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    target_entropy_scale=0.89,  # entropía objetivo = 0.89 * ln(72)  (CleanRL)
    buffer_size=50_000,         # ~2.4 GB con observaciones 12x64x64 uint8
    start_steps=5_000,          # pasos con acciones al azar antes de aprender
    step_per_collect=4,
    update_per_step=0.25,       # 1 actualización cada 4 pasos del entorno
    batch_size=64,
)


def _logits_categorical(logits):
    return torch.distributions.Categorical(logits=logits)


def build_ppo(obs_shape, n_actions, action_space, device="cpu", **hp):
    cfg = {**PPO_DEFAULTS, **hp}
    encoder = NatureCNN(obs_shape[0], device=device).to(device)  # compartido actor/crítico
    actor = Actor(encoder, n_actions, softmax_output=False, device=device).to(device)
    critic = Critic(encoder, device=device).to(device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=cfg["lr"], eps=1e-5)
    policy = PPOPolicy(
        actor, critic, optim, _logits_categorical,
        discount_factor=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        eps_clip=cfg["eps_clip"],
        vf_coef=cfg["vf_coef"],
        ent_coef=cfg["ent_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        reward_normalization=False,
        advantage_normalization=True,
        value_clip=False,
        deterministic_eval=True,   # en evaluación elige la acción más probable
        action_space=action_space,
    )
    return policy, cfg


def build_sac(obs_shape, n_actions, action_space, device="cpu", **hp):
    cfg = {**SAC_DEFAULTS, **hp}
    c = obs_shape[0]
    # Actor y críticos con codificadores separados (como el ejemplo Atari de Tianshou).
    actor = Actor(NatureCNN(c, device=device), n_actions,
                  softmax_output=False, device=device).to(device)
    critic1 = Critic(NatureCNN(c, device=device), last_size=n_actions, device=device).to(device)
    critic2 = Critic(NatureCNN(c, device=device), last_size=n_actions, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg["actor_lr"])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfg["critic_lr"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfg["critic_lr"])

    # α (temperatura) ajustado automáticamente hacia una entropía objetivo.
    target_entropy = cfg["target_entropy_scale"] * math.log(n_actions)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=cfg["alpha_lr"])

    policy = DiscreteSACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        alpha=(target_entropy, log_alpha, alpha_optim),
        reward_normalization=False,
        estimation_step=1,
        action_space=action_space,
    )
    return policy, cfg


def optimizers(policy):
    """Diccionario nombre -> optimizador, para guardar y retomar checkpoints."""
    if isinstance(policy, DiscreteSACPolicy):
        return {
            "actor_optim": policy.actor_optim,
            "critic1_optim": policy.critic1_optim,
            "critic2_optim": policy.critic2_optim,
            "alpha_optim": policy._alpha_optim,
        }
    return {"optim": policy.optim}


def save_checkpoint(policy, path, **extra):
    data = {"policy": policy.state_dict(),
            "optims": {k: o.state_dict() for k, o in optimizers(policy).items()}, **extra}
    if isinstance(policy, DiscreteSACPolicy):
        data["log_alpha"] = policy._log_alpha.detach().cpu()
    torch.save(data, path)


def load_checkpoint(policy, path, device="cpu", with_optims=True):
    data = torch.load(path, map_location=device)
    state = data["policy"] if "policy" in data else data
    policy.load_state_dict(state)
    if with_optims and "optims" in data:
        for k, o in optimizers(policy).items():
            if k in data["optims"]:
                o.load_state_dict(data["optims"][k])
    if isinstance(policy, DiscreteSACPolicy) and "log_alpha" in data:
        with torch.no_grad():
            policy._log_alpha.copy_(data["log_alpha"].to(device))
        policy._alpha = policy._log_alpha.detach().exp()
    return data


@torch.no_grad()
def greedy_action(policy, obs):
    """Acción determinista (la más probable del actor) para una sola observación."""
    policy.eval()
    logits, _ = policy.actor(obs[None])
    return int(logits.argmax(-1)[0])


@torch.no_grad()
def sample_action(policy, obs):
    """Acción estocástica: se sortea según las probabilidades del actor (como al entrenar)."""
    policy.eval()
    logits, _ = policy.actor(obs[None])
    return int(torch.distributions.Categorical(logits=logits).sample()[0])


BUILDERS = {"ppo": build_ppo, "sac": build_sac}
