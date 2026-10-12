# Instalación de MineRL 0.4.4 en Windows (WSL2)

Pasos que funcionaron el 12 de octubre de 2026 en Windows con WSL2 y **Ubuntu 24.04**. Están en el orden en que hay que hacerlos, con las correcciones ya incluidas.

**Resultado:** `MineRLTreechop-v0` arranca y entrega observaciones `(64, 64, 3)`.

---

## 0. Requisitos

- Windows 10 u 11 con **WSL2** y Ubuntu instalado.

  En PowerShell, `wsl -l -v` debe mostrar Ubuntu con `VERSION 2`. Si no lo tienes, ejecuta como administrador `wsl --install -d Ubuntu-22.04` y reinicia.
- Abre Ubuntu con `wsl` en PowerShell, o desde el menú Inicio.
- Trabaja siempre en tu carpeta de Ubuntu (`~`), **no** en `/mnt/c/...`. Ahí todo corre mucho más lento.

## 1. Dependencias del sistema

```bash
sudo apt update
sudo apt install -y openjdk-8-jdk xvfb ffmpeg libgl1 libgl1-mesa-dri git wget unzip \
    x11-xserver-utils libxxf86vm1 libxrandr2 libxtst6 libxi6 libxcursor1
```

> **Importante:** `x11-xserver-utils` instala `xrandr`. Sin él, Minecraft falla al iniciar con el error *"MineRL could not detect an X Server, Monitor, or Virtual Monitor"*, aunque sí haya pantalla. Ver el problema 3 más abajo.

Si `openjdk-8-jdk` no aparece en apt, no importa: se instala con conda en el paso 3.

## 2. Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash && exec bash
```

## 3. Entorno con Python 3.9 y Java 8

MineRL 0.4.4 **no funciona con Python 3.10 o superior**.

Usamos el canal `conda-forge`. Los canales por defecto de Anaconda exigen aceptar sus términos de servicio (`CondaToSNonInteractiveError`).

```bash
conda create -n minerl -c conda-forge --override-channels python=3.9 -y
conda activate minerl
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Verifica Java:

```bash
java -version
```

Debe decir **1.8**. Si no, instálalo dentro del entorno:

```bash
conda install -c conda-forge openjdk=8 -y
java -version
```

## 4. Herramientas de build antiguas

El `gym` 0.19 que usa MineRL no se instala con `setuptools` ni `wheel` recientes.

```bash
pip install "pip<24" setuptools==65.5.0 wheel==0.38.4
```

## 5. Instalar MineRL desde el código fuente, con el arreglo de MixinGradle

`pip install minerl==0.4.4` **falla**. El build de Minecraft (`./gradlew ... shadowJar`) pide la dependencia `com.github.SpongePowered:MixinGradle:dcfaf61`, que ya no existe en ningún repositorio ([minerl#744](https://github.com/minerllabs/minerl/issues/744)).

La solución es usar una copia del `.jar` alojada en GitHub y apuntar `build.gradle` a ella.

```bash
cd ~
git clone --branch v0.4.4 --depth 1 https://github.com/minerllabs/minerl.git
mkdir -p ~/mixin-repo
git clone --depth 1 https://github.com/verityw/MixinGradle-dcfaf61.git ~/mixin-repo/MixinGradle-dcfaf61

F=~/minerl/minerl/Malmo/Minecraft/build.gradle
sed -i "s|maven { url 'https://jitpack.io' }|maven { url 'file:$HOME/mixin-repo' }\n        maven { url 'https://jitpack.io' }|" $F
sed -i "s|com.github.SpongePowered:MixinGradle:dcfaf61|MixinGradle-dcfaf61:MixinGradle:dcfaf61|" $F
grep -n "mixin-repo\|MixinGradle" $F
```

El `grep` debe mostrar estas líneas (la ruta usa tu usuario):

```
5:        maven { url 'file:/home/<usuario>/mixin-repo' }
20:        classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
95:    maven { url 'file:/home/<usuario>/mixin-repo' }
```

Instala (tarda **15–30 min** porque compila Minecraft):

```bash
cd ~/minerl
java -version          # debe decir 1.8
pip install . 2>&1 | tee ~/install.log
```

Debe terminar con `Successfully installed ... minerl-0.4.4`.

## 6. Bajar numpy a la versión 1

pip instala numpy 2, que no es compatible con MineRL 0.4.4 ni con gym 0.19. Opencv también se instala en una versión pensada para numpy 2.

```bash
pip install "numpy<2" "opencv-python<4.11"
```

## 7. Probar

En Windows 11, WSL ya trae pantalla propia (WSLg). Verifícalo:

```bash
echo $DISPLAY          # debe mostrar :0
```

Prueba sin `xvfb`. Se abre una ventana de Minecraft en Windows:

```bash
cd ~
python -c "import gym, minerl; env = gym.make('MineRLTreechop-v0'); obs = env.reset(); print('OK', obs['pov'].shape); env.close()"
```

Resultado esperado: `OK (64, 64, 3)`.

Los avisos `Box bound precision lowered by casting to float32` y `RuntimeWarning: 'minerl.utils.process_watcher'...` son normales.

### Sin ventana (para entrenar varias horas)

```bash
xvfb-run -a -s "-screen 0 1024x768x24" python -c "import gym, minerl; env = gym.make('MineRLTreechop-v0'); obs = env.reset(); print('OK', obs['pov'].shape); env.close()"
```

*(Pendiente de confirmar en este computador.)* Si falla, prueba anteponer `LIBGL_ALWAYS_SOFTWARE=1`, o entrena con WSLg.

---

## Problemas que aparecieron y su solución

| # | Error | Causa | Solución |
|---|---|---|---|
| 1 | `CondaToSNonInteractiveError: Terms of Service have not been accepted` | Los canales por defecto de Anaconda exigen aceptar sus términos | Usar `-c conda-forge --override-channels` (paso 3) |
| 2 | `CalledProcessError: Command '['./gradlew', '-g', 'run/gradle', 'shadowJar']' returned non-zero exit status 1` | Falta `MixinGradle:dcfaf61` en todos los repositorios | Instalar desde el código fuente con la copia local del `.jar` (paso 5) |
| 3 | `RuntimeError: ERROR! MineRL could not detect an X Server, Monitor, or Virtual Monitor!`, tanto con WSLg como con `xvfb-run` | Mensaje engañoso: MineRL lo lanza cada vez que Java falla en `org.lwjgl.opengl.Display.<clinit>`. Aquí la causa era que faltaba `xrandr`, que LWJGL usa para leer la resolución | `sudo apt install -y x11-xserver-utils libxxf86vm1 libxrandr2 libxtst6 libxi6 libxcursor1` (paso 1) |
| 4 | `ModuleNotFoundError: No module named 'gym'` | MineRL aún no estaba instalado (gym se instala con él) | Completar el paso 5 |

### Si aparece otro error

- **Al instalar:** busca la causa real de Gradle en el log.

  ```bash
  grep -n -A 12 "What went wrong" ~/install.log
  ```

  Si menciona `downloadAssets`, repite `pip install .`: a veces falla solo la primera vez.
- **Al ejecutar:** muestra el error completo de Java, que MineRL normalmente oculta.

  ```bash
  python -c "
  import logging, coloredlogs; coloredlogs.install(logging.DEBUG)
  import gym, minerl
  env = gym.make('MineRLTreechop-v0'); obs = env.reset(); print('OK'); env.close()
  " 2>&1 | grep -B 5 -A 25 "Exception\|Error" | head -80
  ```

## Versiones que quedaron instaladas

- Ubuntu 24.04.1 en WSL2
- Python 3.9 (conda-forge)
- Java 1.8
- minerl 0.4.4 (desde el tag `v0.4.4` con el parche de MixinGradle)
- gym 0.19.0
- numpy < 2
- opencv-python < 4.11
