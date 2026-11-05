# Entrenar PPO y SAC discretos

Esta guía cubre la primera etapa del proyecto: entrenar los dos agentes de **cámara discreta** (PPO y SAC) en MineRL. Los algoritmos **no los implementamos nosotros**: vienen de la librería [Tianshou 0.4.8](https://github.com/thu-ml/tianshou/tree/v0.4.8). Nuestro código solo arma el entorno, la red y los scripts para entrenar y evaluar.

| Agente | Clase de Tianshou | Paper |
|---|---|---|
| PPO discreto | `tianshou.policy.PPOPolicy` | Schulman et al., 2017 |
| SAC discreto | `tianshou.policy.DiscreteSACPolicy` | Christodoulou, 2019 |

**¿Por qué Tianshou 0.4.8?** MineRL 0.4.4 obliga a usar Python 3.9 y gym 0.19. Tianshou 0.4.8 funciona con esa versión de gym y trae PPO y SAC discreto bajo la misma interfaz (mismo recolector, mismo buffer y mismos registros), lo que hace justa la comparación. Stable-Baselines3 no tiene SAC discreto, y sus versiones nuevas exigen gym 0.21.

---

## 0. Qué ve y qué hace el agente

Los dos agentes usan exactamente el mismo entorno (`make_agent_env` en `src/mrl/envs/wrappers.py`):

- **Observación:** los últimos 4 frames RGB de 64×64 apilados → arreglo `uint8` de forma `(12, 64, 64)`. En Navigate se agrega un canal 13 con el ángulo de la brújula.
- **Acción:** un número entre 0 y 71 (`Discrete(72)`) = 8 combinaciones de teclas (`forward`, `jump`, `attack`) × 9 giros de cámara (arriba/abajo/izquierda/derecha/diagonales/nada, de 15°).
- **Repetición de acción:** cada acción dura 4 frames de Minecraft. La cámara gira solo en el primero.
- **Red:** la CNN "Nature" de DQN (`src/mrl/networks.py`), la misma de los ejemplos de Atari de Tianshou.

```
MineRL ─► ActionRepeat(4) ─► PovStack(4 frames) ─► DiscreteCamera(15°) ─► FlatDiscreteAction ─► PovArray ─► agente
```

---

## 1. Instalar las librerías (una sola vez)

Dentro de WSL, con el entorno `minerl` que ya tiene MineRL funcionando:

```bash
conda activate minerl
cd ~/ReinforcedLearning-Minecraft
pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.2.2+cpu" "tianshou==0.4.8" tensorboard
```

> Se instala PyTorch **solo CPU** porque la GPU es AMD (y en WSL no hay CUDA). Ocupa ~200 MB en vez de ~2.5 GB.

Revisa que no se haya movido gym ni numpy:

```bash
python -c "import gym, numpy, torch, tianshou; print(gym.__version__, numpy.__version__, torch.__version__, tianshou.__version__)"
# esperado: 0.19.0  1.x  2.2.2+cpu  0.4.8
```

Si pip actualizó numpy a 2.x, vuelve atrás con `pip install "numpy<2"`.

---

## 2. Prueba rápida sin Minecraft (1 minuto)

`--dry-run` usa un Minecraft simulado (imágenes al azar) y pocos pasos. No aprende nada útil: solo comprueba que el código corre de punta a punta.

```bash
python scripts/train.py --algo ppo --dry-run
python scripts/train.py --algo sac --dry-run
python scripts/evaluate.py --run runs_dryrun/treechop/ppo_discrete/seed0 --dry-run --episodes 2
```

Si los tres terminan sin error, está todo listo.

---

## 3. Probar el entorno real

```bash
python scripts/test_wrappers.py --agent-env --steps 200
```

Abre Minecraft, juega 200 pasos al azar con las 72 acciones, imprime los pasos por segundo y deja un video en `out/test_treechop_agent.mp4`.

---

## 4. Entrenar

```bash
# PPO (Bastián)
python scripts/train.py --algo ppo --task treechop --seed 0

# SAC (Gabriela)
python scripts/train.py --algo sac --task treechop --seed 0
```

Por defecto son **100 000 pasos del agente** (= 400 000 frames de Minecraft) en 10 épocas de 10 000 pasos. Al final de cada época se guarda `checkpoint.pth`.

**Duración estimada:** sin aprender, el entorno da ~10 pasos/seg. Con el aprendizaje en CPU, estima **~4 a 6 horas por corrida**; SAC es algo más lento porque actualiza la red durante la recolección. Mide el primer run y ajusta el plan.

Mientras corre, **no cierres la ventana de Minecraft** (si se cierra, el entrenamiento se cae) y evita que el PC entre en suspensión.

Para que siga corriendo aunque cierres la terminal, usa `tmux`:

```bash
sudo apt install tmux        # una vez
tmux new -s sac              # abre una sesión
conda activate minerl && cd ~/ReinforcedLearning-Minecraft
python scripts/train.py --algo sac --task treechop --seed 0
# Ctrl+b y luego d  → te sales y sigue corriendo
tmux attach -t sac           # para volver a verla
```

### Plan de corridas

2 algoritmos × 2 tareas × 3 semillas = **12 corridas**. Primero Treechop (la principal):

| | Treechop | Navigate |
|---|---|---|
| PPO (Bastián) | `--seed 0`, `1`, `2` | `--task navigate --seed 0`, `1`, `2` |
| SAC (Gabriela) | `--seed 0`, `1`, `2` | `--task navigate --seed 0`, `1`, `2` |

### Dónde quedan los resultados

```
runs/<tarea>/<algo>_discrete/seed<N>/
├── config.json      configuración exacta usada (hiperparámetros incluidos)
├── episodes.csv     retorno y largo de cada episodio de entrenamiento
├── checkpoint.pth   último checkpoint (para retomar)
├── policy.pth       pesos finales
└── events.*         curvas de TensorBoard
```

`runs/` no se sube a GitHub (está en `.gitignore`). Para compartir resultados, copien `eval.json` y `episodes.csv` a una carpeta `results/` y súbanla.

### Si se corta: retomar

Mismo comando + `--resume`. Sigue desde la última época guardada:

```bash
python scripts/train.py --algo sac --task treechop --seed 0 --resume
```

> En SAC el buffer de experiencias no se guarda (pesaría ~2.4 GB), así que al retomar parte con el buffer vacío. Es normal ver una baja por un rato.

### Cambiar hiperparámetros

```bash
python scripts/train.py --algo ppo --hp ent_coef=0.02 --hp lr=1e-4
python scripts/train.py --algo sac --hp buffer_size=30000
```

Los valores por defecto están en `src/mrl/agents.py` (`PPO_DEFAULTS` y `SAC_DEFAULTS`).

---

## 5. Ver las curvas de aprendizaje

En otra terminal:

```bash
conda activate minerl
cd ~/ReinforcedLearning-Minecraft
tensorboard --logdir runs
```

Abre <http://localhost:6006> en el navegador de Windows. La curva principal es `train/reward` (en Treechop = troncos por episodio). Con `runs` como carpeta, TensorBoard muestra todas las corridas juntas para compararlas.

---

## 6. Evaluar

Cada agente se evalúa en **los mismos 20 mundos** (semillas 1000 a 1019) con la política determinista (siempre la acción más probable):

```bash
python scripts/evaluate.py --run runs/treechop/ppo_discrete/seed0
python scripts/evaluate.py --run runs/treechop/sac_discrete/seed0 --video   # + video del 1er episodio
python scripts/evaluate.py --random --task treechop                          # línea base: agente al azar
```

Deja `eval.csv` (un renglón por episodio) y `eval.json` (resumen) en la carpeta del run.

| Tarea | Métrica principal | En `eval.json` |
|---|---|---|
| Treechop | troncos por episodio | `mean_return`, `std_return` |
| Navigate | % de episodios que llegan a la meta | `success_rate` |

Un episodio de Treechop dura hasta 2000 pasos (~3–4 min), así que 20 episodios toman ~1 hora. Para algo rápido usa `--episodes 5`.

---

## 7. Memoria (importante)

El PC tiene ~7.7 GB disponibles dentro de WSL. Lo que se usa:

| | RAM aprox. |
|---|---|
| Minecraft (1 instancia) | ~3 GB |
| Buffer de SAC (50 000 transiciones) | ~2.4 GB |
| PyTorch + redes | ~0.5 GB |

- Corre **una sola corrida a la vez** y deja `--n-envs 1`.
- Si SAC se cae por falta de memoria (`Killed`), baja el buffer: `--hp buffer_size=30000`.
- WSL usa por defecto la mitad de la RAM de Windows. Para darle más, crea `C:\Users\<usuario>\.wslconfig` con:
  ```ini
  [wsl2]
  memory=12GB
  ```
  y reinicia WSL con `wsl --shutdown` desde PowerShell.

---

## 8. Hiperparámetros por defecto

Tomados de los ejemplos de Atari de Tianshou y de CleanRL (Huang et al., 2022), que usan la misma CNN y observaciones de imagen.

**PPO**

| Parámetro | Valor |
|---|---|
| tasa de aprendizaje | 2.5e-4 (Adam) |
| γ / λ de GAE | 0.99 / 0.95 |
| clip ε | 0.2 |
| coef. de valor / entropía | 0.5 / 0.01 |
| norma máx. del gradiente | 0.5 |
| pasos por recolección | 1024 |
| épocas por recolección / minibatch | 4 / 256 |

**SAC discreto**

| Parámetro | Valor |
|---|---|
| tasa de aprendizaje (actor, críticos, α) | 3e-4 (Adam) |
| γ / τ | 0.99 / 0.005 |
| α (temperatura) | automático, entropía objetivo = 0.89 · ln(72) |
| buffer | 50 000 |
| pasos al azar al inicio | 5 000 |
| actualizaciones | 1 cada 4 pasos, minibatch 64 |

---

## 9. Problemas comunes

| Síntoma | Solución |
|---|---|
| `could not detect X Server` | Revisa `echo $DISPLAY` (debe ser `:0`) y `docs/INSTALL_MINERL.md`. |
| `Ya existe .../checkpoint.pth` | Usa `--resume` para seguir esa corrida, o cambia `--seed`. |
| `Killed` sin más mensaje | Falta RAM: ver sección 7. |
| Minecraft se cierra a mitad | Retoma con `--resume`. La columna `minerl_error` de `episodes.csv` cuenta las caídas. |
| numpy 2.x después de instalar | `pip install "numpy<2"` |

---

## Archivos

```
src/mrl/
├── envs/wrappers.py   entorno: repetición de acción, pila de frames, cámara discreta/híbrida, Discrete(72)
├── envs/fake.py       Minecraft simulado para --dry-run
├── networks.py        CNN Nature (codificador de imágenes)
└── agents.py          arma PPO y SAC discreto de Tianshou + hiperparámetros
scripts/
├── train.py           entrenar
├── evaluate.py        evaluar en mundos fijos
└── test_wrappers.py   probar el entorno con acciones al azar
```
