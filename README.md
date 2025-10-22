# Sistema de Reconocimiento Facial en Tiempo Real 📷🧠

Aplicación en Python que permite registrar personas, reconocer rostros en tiempo real y analizar emociones usando OpenCV y DeepFace, con una interfaz web simple en Streamlit y persistencia en SQLite.

- Reconocimiento facial con embeddings (Facenet, ArcFace o VGG-Face)
- Análisis de emociones (7 clases: angry, disgust, fear, happy, sad, surprise, neutral)
- Ajustes finos para mejorar la detección de “disgust” y suavizado temporal de emociones
- Reportes interactivos (gráficas, últimos eventos) y filtro por emoción
- Administración: alta de personas y eliminación segura (preserva historial)

---

## Requisitos previos ⚙️

- Python 3.10 o superior
- Windows recomendado (probado con cámara local). También funciona en otros SO con Python + OpenCV.
- Cámara web disponible

Dependencias se instalan desde `requirements.txt`.

---

## Instalación 📦

En PowerShell (Windows):

```powershell
# 1) Clonar el repositorio
# git clone https://github.com/<tu-usuario>/<repo>.git
# cd <repo>

# 2) Crear y activar entorno virtual (opcional pero recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Instalar dependencias
pip install -r requirements.txt
```

Notas:
- DeepFace descargará automáticamente algunos modelos la primera vez que se ejecuta (puede tardar varios minutos).
- Si OpenCV reporta errores de DLL en Windows, instala el "Microsoft Visual C++ Redistributable".

---

## Puesta en marcha rápida 🚀

Ejecuta la interfaz web:

```powershell
streamlit run app.py
```

La app abre una interfaz con tres secciones en la barra lateral:

1. Registro: alta de personas y captura de rostros (vía cámara del navegador) para generar embeddings.
2. Detección: lanza una ventana nativa (OpenCV) con reconocimiento en tiempo real y análisis de emociones.
3. Reportes: estadísticas, gráficas por emoción y últimos eventos (con filtro por emoción).

Para cerrar la ventana nativa de detección, presiona la tecla `q`.

---

## Flujo de trabajo 🧭

### 1) Registrar personas
- Completa Nombre, Apellido y Email.
- Usa la cámara para tomar varias capturas (se recomienda ≥ 5, con buena iluminación y rostro centrado).
- El sistema detecta y recorta el rostro más grande de cada captura y guarda el embedding en la tabla `Rostro`.

Administración:
- Puedes eliminar una persona desde la misma página. La eliminación borra sus embeddings y deja en `NULL` el `id_persona` de detecciones históricas, preservando el historial.

### 2) Detección en tiempo real
- Configura los parámetros (umbral de reconocimiento, intervalos, backends, modelos, resolución, FPS, etc.).
- Pulsa “Iniciar” para abrir la ventana nativa. Se verán:
  - Caja del rostro, nombre si hay match y confianza de reconocimiento
  - Emoción dominante y confianza
  - Top-3 emociones (ayuda para calibración)
  - FPS actual
- Pulsa “Detener” o cierra con `q`.

Calibraciones de emoción importantes:
- Ganancia de “disgust” (por defecto 1.6) para mejorar su sensibilidad.
- Suavizado temporal (ventana en frames) para reducir parpadeo.
- Balanceo por prior (opcional) para evitar que una emoción quede subrepresentada.

### 3) Reportes
- Métricas generales (personas registradas, detecciones del día, emoción predominante, tasa de reconocimientos confiables).
- Gráfica de distribución de emociones por persona.
- Últimas detecciones (aplica filtro por emoción si eliges una).

---

## Uso por línea de comandos (opcional) 🖥️

Además de la UI, puedes usar el script nativo `camera_local.py`.

### Registrar desde CLI

```powershell
python camera_local.py register --camera 0 --nombre "Juan" --apellido "Pérez" --email "juan@example.com"
```

Opciones útiles:
- `--person-id <id>`: agrega capturas a una persona existente.
- `--duplicate-threshold 0.8`: umbral de similitud para advertir duplicados.
- `--force`: guarda aunque parezca duplicado.

Presiona `c` para capturar y `q` para salir (mínimo recomendado: 5 capturas).

### Detectar desde CLI

```powershell
python camera_local.py detect --camera 0 --threshold 0.6 --infer-interval-ms 500
```

Parámetros importantes (resumen):
- Reconocimiento:
  - `--threshold`: similitud coseno mínima para considerar un match.
  - `--embed-model`: `ArcFace`, `Facenet` (default), `VGG-Face`.
  - `--detector-backend`: `opencv`, `opencv-dnn`, `retinaface`, `mediapipe`.
- Emociones:
  - `--no-emotion`: deshabilita emociones para máximo FPS.
  - `--emotion-backend`: `opencv`, `retinaface`, `mediapipe`, `skip`.
  - `--emotion-interval-ms`: cada cuánto recalcular emociones.
  - `--emotion-scale`: upscale del recorte de emociones (mejor detalle).
  - `--crop-padding`: padding alrededor del rostro para emociones.
  - `--emo-disgust-gain`: ganancia a “disgust” (default 1.6).
  - `--emo-smooth-frames`: suavizado temporal (frames).
  - `--emo-balance`, `--emo-balance-strength`, `--emo-balance-alpha`.
- Rendimiento:
  - `--frame-width`, `--frame-height`: resolución de la cámara.
  - `--detect-scale` y `--scale`: aceleran detección / embeddings.
  - `--force-mjpg`: reduce carga CPU en algunos drivers.
  - `--target-fps`: intenta fijar FPS.
  - `--box-smooth-alpha`: suavizado exponencial de la caja.
  - `--min-log-interval-ms`: limita frecuencia de escritura a DB.

> Tip: Para usar `opencv-dnn` necesitas colocar los archivos del modelo Caffe en `./models`:
> - `deploy.prototxt`
> - `res10_300x300_ssd_iter_140000.caffemodel`

---

## Base de datos y persistencia 🗄️

- Archivo SQLite: `facial_recognition.db` en la raíz del proyecto.
- Tablas principales:
  - `Persona(id, nombre, apellido, email, fecha_registro_timestamp)`
  - `Rostro(id, id_persona, rostro BLOB)` (embeddings)
  - `Deteccion(id, id_persona NULL, recog_confianza, emocion, emocion_confianza, timestamp)`
  - `DeteccionEmocionDetalle(id, id_deteccion, emocion, confianza)` (distribución completa por evento)
- Migraciones básicas se realizan automáticamente si vienes de un esquema anterior.
- Al eliminar una persona:
  - Se borran sus embeddings en `Rostro`.
  - Se pone `id_persona = NULL` en `Deteccion` para conservar el historial.

---

## Ajustes de rendimiento y calidad ⚡

- Aumentar resolución mejora la calidad de recortes, pero reduce FPS.
- `--force-mjpg` y `--target-fps` pueden reducir latencia en Windows.
- Mantén buena iluminación y encuadre para mejores embeddings y emociones.
- “Disgust” suele infradetectarse: usa la ganancia y, si es necesario, incrementa suavizado/balanceo.

---

## Solución de problemas 🧩

- No abre la cámara: revisa el índice (`--camera`), permisos del SO y que otra app no la esté usando.
- Ventana se cierra inmediatamente: mira la consola para mensajes (DLLs, permisos, drivers).
- Errores DeepFace al inicio: la primera vez descarga pesos; espera a que termine.
- Emojis/acentos extraños en consola: usa PowerShell con UTF-8 o evita caracteres especiales.

---

## Estructura del proyecto 📁

```
app.py                # Interfaz Streamlit (Registro, Detección, Reportes)
camera_local.py       # Motor nativo (OpenCV/DeepFace) y CLI
requirements.txt      # Dependencias del proyecto
facial_recognition.db # (se crea al ejecutar) Base de datos SQLite
models/               # (opcional) Modelos Caffe para opencv-dnn
```

---

## Créditos 🙌

- [OpenCV](https://opencv.org/) para video y visión por computador
- [DeepFace](https://github.com/serengil/deepface) para embeddings y emociones
- [Streamlit](https://streamlit.io/) para la interfaz web
