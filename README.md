# ReinforcedLearning-Minecraft

Proyecto del ramo **Aprendizaje por Refuerzo** (Universidad de Concepción), sobre el entorno [MineRL](https://minerl.readthedocs.io/en/v0.4.4/).

Integrantes: Gabriela Escalona Weldt · Bastián Zapata

## Tema

**Cámara discreta vs. continua en Minecraft: comparación de PPO y SAC con políticas híbridas en MineRL**

> ¿Cómo cambia el desempeño de PPO y SAC en MineRL según se represente la cámara como saltos fijos (discreta) o como un ángulo libre (continua)?

En MineRL las teclas son discretas, pero la cámara es continua (un ángulo en grados). Casi todos los trabajos previos la discretizan en giros fijos. Comparamos esa opción con una política **híbrida**: teclas discretas + cámara continua.

|  | Cámara discreta (9 giros de ±15°) | Cámara continua (híbrida) |
|---|---|---|
| **PPO** | Agente 1 | Agente 2 |
| **SAC** | Agente 3 | Agente 4 |

- Agentes 1 y 3: algoritmos tomados de [Tianshou](https://github.com/thu-ml/tianshou) (PPO y SAC discreto).
- Agentes 2 y 4: versiones híbridas desarrolladas por nosotros (teclas categóricas + cámara gaussiana 2D).

**Tareas:** `MineRLTreechop-v0` (principal, juntar troncos) y `MineRLNavigateDense-v0` (llegar a un punto con brújula).

**Hipótesis**

- **H1.** La cámara continua mejora más a SAC que a PPO, porque SAC fue diseñado para acciones continuas.
- **H2.** La ventaja de la cámara continua es mayor en Navigate, donde apuntar con precisión importa más.
- **H3.** Con cámara discreta, PPO aprende de forma más estable que SAC discreto.

## Instalación

MineRL 0.4.4 requiere Python 3.9, Java 8 y un parche para compilar Minecraft. La guía completa, probada en Windows con WSL2 (Ubuntu 24.04), está en [docs/INSTALL_MINERL.md](docs/INSTALL_MINERL.md).

```bash
conda env create -f environment.yml
conda activate minerl
# 1) MineRL desde el código fuente: docs/INSTALL_MINERL.md, paso 5
# 2) librerías de entrenamiento (PyTorch CPU + Tianshou):
pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.2.2+cpu" "tianshou==0.4.8" tensorboard
```

## Uso

```bash
python scripts/check_env.py                          # ¿funciona MineRL?
python scripts/test_wrappers.py --agent-env          # ¿funciona el entorno de los agentes?
python scripts/train.py --algo ppo --dry-run         # ¿corre el entrenamiento? (sin Minecraft)
python scripts/train.py --algo sac --task treechop --seed 0     # entrenar
python scripts/evaluate.py --run runs/treechop/sac_discrete/seed0   # evaluar
tensorboard --logdir runs                            # curvas de aprendizaje
```

Guía paso a paso del entrenamiento: [docs/ENTRENAMIENTO.md](docs/ENTRENAMIENTO.md).

## Estructura

```
├── docs/
│   ├── INSTALL_MINERL.md     instalación de MineRL en WSL
│   └── ENTRENAMIENTO.md      entrenar y evaluar PPO / SAC discretos
├── src/mrl/
│   ├── envs/wrappers.py      entorno compartido por los 4 agentes
│   ├── envs/fake.py          Minecraft simulado (pruebas)
│   ├── networks.py           CNN para las imágenes
│   └── agents.py             PPO y SAC discreto (Tianshou)
├── scripts/
│   ├── check_env.py          prueba de MineRL + pasos/seg + video
│   ├── test_wrappers.py      prueba del entorno con acciones al azar
│   ├── train.py              entrenamiento
│   └── evaluate.py           evaluación en mundos fijos
├── environment.yml           entorno conda
└── .gitignore                data/, runs/, out/, modelos y videos no se suben
```

## Cómo trabajamos

- `main` siempre funciona. Cada uno trabaja en su rama y hace merge cuando algo corre.
- El código se trabaja dentro de WSL (`~/ReinforcedLearning-Minecraft`), no en una carpeta de Windows ni de OneDrive.
- Los datos, modelos y videos quedan fuera de Git (ver `.gitignore`). Los resúmenes de resultados van en `results/`.
- Bastián: PPO. Gabriela: SAC.

## Referencias

- Guss, W. H. et al. (2019). *MineRL: A Large-Scale Dataset of Minecraft Demonstrations*. IJCAI.
- Milani, S. et al. (2020). *Retrospective Analysis of the 2019 MineRL Competition on Sample Efficient Reinforcement Learning*. NeurIPS Competition Track.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
- Christodoulou, P. (2019). *Soft Actor-Critic for Discrete Action Settings*. arXiv:1910.07207.
- Fan, Z. et al. (2019). *Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space*. IJCAI.
- Delalleau, O. et al. (2019). *Discrete and Continuous Action Representation for Practical RL in Video Games*. AAAI Workshop.
- Weng, J. et al. (2022). *Tianshou: A Highly Modularized Deep Reinforcement Learning Library*. JMLR.
