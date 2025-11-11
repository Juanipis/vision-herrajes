# Vision Herrajes

Sistema de visión por computadora para clasificación automática de herrajes mediante análisis de video y cámara en vivo.

---

## Estructura del Proyecto

```
vision-herrajes/
├── config/                 # Configuraciones (presets de filtros)
│   └── filter_presets.json
├── data/                   # ⚠️ NO en Git - Datos procesados y submodelos
│   ├── processed/         # Frames procesados organizados por familia
│   │   ├── anillo/        # (Carpetas con frames: ANILLO_BUENO_T1_orig_05/, etc.)
│   │   ├── dobleanillo/
│   │   ├── gancho/
│   │   ├── ocho/
│   │   └── manifest.json
│   └── submodels/         # Datasets para submodelos
│       ├── base/
│       │   ├── anillo/
│       │   ├── dobleanillo/
│       │   ├── gancho/
│       │   ├── ocho/
│       │   ├── manifest.json
│       │   └── summary.json
│       └── size/
│           ├── anillo/
│           ├── dobleanillo/
│           ├── gancho/
│           └── ocho/
├── train/                  # ⚠️ NO en Git - Videos de entrenamiento por familia
│   ├── anillo/            # ANILLO_BUENO_T1.mp4, ANILLO_MALO_T1.mp4, etc.
│   ├── dobleanillo/       # DOBLEANILLO_BUENO_T1.mp4, etc.
│   ├── gancho/            # GANCHO_BUENO_T2.mp4, etc.
│   └── ocho/              # OCHO_BUENO_T1.mp4, etc.
├── results/                # ✅ SÍ en Git - Modelos entrenados y manifiestos
│   ├── *.joblib           # Modelos MLP entrenados
│   └── *.json             # Manifiestos y metadatos
├── scripts/                # Scripts de procesamiento
├── src/                    # Código fuente
│   ├── gui/               # Interfaces gráficas
│   ├── features/          # Extracción de características
│   ├── processing/        # Pipeline de procesamiento
│   ├── io/                # Entrada/Salida
│   └── submodels/         # Submodelos especializados
├── logs/                   # Logs del sistema
└── Makefile               # Comandos simplificados (solo Linux/macOS)
```

---

## Instalación

### 1. Crear entorno virtual

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
uv pip install opencv-python pillow av tqdm scikit-image scikit-learn scipy joblib
```

O con pip:
```bash
pip install opencv-python pillow av tqdm scikit-image scikit-learn scipy joblib
```

---

## Estructura de Carpetas NO en Git

### Carpeta `data/`

Se crea automáticamente durante el procesamiento:

```
data/
├── processed/
│   ├── anillo/
│   │   ├── ANILLO_BUENO_T1_orig_05/
│   │   ├── ANILLO_MALO_T1_orig_01/
│   │   └── ...
│   ├── dobleanillo/
│   ├── gancho/
│   ├── ocho/
│   └── manifest.json
└── submodels/
    ├── base/
    │   ├── anillo/
    │   ├── dobleanillo/
    │   ├── gancho/
    │   ├── ocho/
    │   ├── manifest.json
    │   └── summary.json
    └── size/
        ├── anillo/
        ├── dobleanillo/
        ├── gancho/
        └── ocho/
```

### Carpeta `train/`

Debe contener videos organizados por familia:

```
train/
├── anillo/
│   ├── ANILLO_BUENO_T1.mp4
│   ├── ANILLO_MALO_T1.mp4
│   └── ...
├── dobleanillo/
│   ├── DOBLEANILLO_BUENO_T1.mp4
│   └── ...
├── gancho/
│   ├── GANCHO_BUENO_T2.mp4
│   └── ...
└── ocho/
    ├── OCHO_BUENO_T1.mp4
    └── ...
```

**Crear estructura:**

```bash
# Windows
mkdir train
mkdir train\anillo train\dobleanillo train\gancho train\ocho

# Linux/macOS
mkdir -p train/{anillo,dobleanillo,gancho,ocho}
```

Coloca los videos de entrenamiento (.mp4) en las carpetas correspondientes según el tipo de herraje.

---

## Uso

### Clasificación con Videos

**Linux/macOS:**
```bash
make classify PRESET=PHANSALKAR SIZE_PRESET=OTSU
```

**Windows:**
```cmd
python -m src.gui.model_viewer --preset PHANSALKAR --size-preset OTSU
```

### Clasificación con Cámara en Vivo

**Linux/macOS:**
```bash
make classify-camera PRESET=PHANSALKAR SIZE_PRESET=OTSU
```

**Windows:**
```cmd
python -m src.gui.camera_viewer --preset PHANSALKAR --size-preset OTSU
```

**Opciones adicionales:**
- `--camera-index N`: Especifica la cámara a usar (0, 1, 2, etc.)
- `--model path/to/model.joblib`: Usar un modelo específico

---

## ⚠️ ADVERTENCIA - Comandos de Entrenamiento

**NO ejecutar** los siguientes comandos a menos que sepas lo que estás haciendo:
- `make dataset` / `python scripts/build_dataset.py`
- `make train` / `python scripts/train_mlp.py`
- `make submodel-base` / `python scripts/build_submodel_dataset.py`
- `make size-dataset` / `python scripts/build_size_dataset.py`
- `make size-train` / `python scripts/train_size_submodel.py`

**Razones:**
- ⚠️ Pueden sobreescribir el modelo ya entrenado
- 💾 Requieren más de **12 GB** de frames del dataset
- ❌ **NO son necesarios** para ejecutar el modelo de clasificación
- ✅ El modelo ya está entrenado y listo para usar en `results/`

**Para usar el sistema, solo necesitas los comandos de clasificación** (`make classify` o `make classify-camera`).

---

## Comandos Disponibles

### Comandos de Clasificación (Uso Normal)

#### Para Linux/macOS (con Makefile)

| Comando                                                   | Descripción                     |
| --------------------------------------------------------- | ------------------------------- |
| `make classify PRESET=PHANSALKAR SIZE_PRESET=OTSU`        | GUI de clasificación con video  |
| `make classify-camera PRESET=PHANSALKAR SIZE_PRESET=OTSU` | GUI de clasificación con cámara |
| `make evaluate`                                           | Evaluar videos en batch         |

#### Para Windows (comandos Python directos)

| Comando                                                                  | Descripción                     |
| ------------------------------------------------------------------------ | ------------------------------- |
| `python -m src.gui.model_viewer --preset PHANSALKAR --size-preset OTSU`  | GUI de clasificación con video  |
| `python -m src.gui.camera_viewer --preset PHANSALKAR --size-preset OTSU` | GUI de clasificación con cámara |
| `python scripts/batch_classify.py`                                       | Evaluar videos en batch         |

### Comandos Avanzados (Solo para Desarrollo)

⚠️ **No ejecutar estos comandos** (ver advertencia arriba)

#### Para Linux/macOS (con Makefile)

| Comando                               | Descripción                                              |
| ------------------------------------- | -------------------------------------------------------- |
| `make gui`                            | Lanzar GUI de ajuste de filtros                          |
| `make preview VIDEO=path PRESET=name` | Exportar frames de vista previa                          |
| `make dataset PRESET=name`            | ⚠️ Construir dataset balanceado (requiere +12 GB)         |
| `make train`                          | ⚠️ Entrenar clasificador MLP (puede sobreescribir modelo) |

#### Para Windows (comandos Python directos)

| Comando                                           | Descripción                     |
| ------------------------------------------------- | ------------------------------- |
| `python -m src.gui.video_tuner`                   | Lanzar GUI de ajuste de filtros |
| `python scripts/preview_filters.py <video>`       | Exportar frames de vista previa |
| `python scripts/build_dataset.py --preset <name>` | ⚠️ Construir dataset balanceado  |
| `python scripts/train_mlp.py`                     | ⚠️ Entrenar clasificador MLP     |

---
