# ReinforcedLearning-Minecraft

Proyecto del ramo **Aprendizaje por Refuerzo** (Universidad de Concepción), sobre el entorno [MineRL](https://minerl.readthedocs.io/en/v0.4.4/).

Integrantes: Gabriela Escalona Weldt · Bastián Zapata

> **Tema:** *por definir.*

## Instalación

MineRL 0.4.4 requiere Python 3.9, Java 8 y un parche para compilar Minecraft. La guía completa, probada en Windows con WSL2 (Ubuntu 24.04), está en [docs/INSTALL_MINERL.md](docs/INSTALL_MINERL.md).

Resumen:

```bash
conda env create -f environment.yml
conda activate minerl
# luego instalar MineRL desde el código fuente: docs/INSTALL_MINERL.md, paso 5
```

## Verificar que funciona

```bash
conda activate minerl
python scripts/check_env.py
```

Debe abrir Minecraft, imprimir los pasos por segundo y dejar un video en `out/`.

## Estructura

```
├── docs/                  # guías e informes
├── scripts/               # scripts ejecutables
│   └── check_env.py       # prueba de MineRL + pasos/seg + video
├── environment.yml        # entorno conda
└── .gitignore             # data/, out/, checkpoints/ y videos no se suben
```

## Cómo trabajamos

- `main` siempre funciona. Cada uno trabaja en su rama y hace merge cuando algo corre.
- El código se trabaja dentro de WSL (`~/ReinforcedLearning-Minecraft`), no en una carpeta de Windows ni de OneDrive.
- Los datos, modelos y videos quedan fuera de Git (ver `.gitignore`).
