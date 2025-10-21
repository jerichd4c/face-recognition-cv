# libraries

from datetime import datetime
import io
import sqlite3
import time
import pickle
import hashlib
from typing import Optional, Tuple
import subprocess
import sys
import os
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px

# Reuse core logic from local camera script
from camera_local import (
    extract_embedding,
    analyze_emotion,
    analyze_emotion_full,
    recognize,
    load_person_embeddings,
)

# initial page config
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    layout="wide",
    page_icon="📷"
)

class FacialRecognitionSystem():
    def __init__(self):
        self.init_database()
        # Cache cascade to avoid reloading
        self._cascade = None
            
    #initialize database
    def init_database(self):
        self.conn = sqlite3.connect('facial_recognition.db', check_same_thread=False)
        cursor = self.conn.cursor()

        # DEBUG
        print("Conexion exitosa a la base de datos")

        # tables

        # Person table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Persona (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                apellido TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                fecha_registro_timestamp DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("Tabla persona creada")

        # Face table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Rostro (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_persona INTEGER NOT NULL,
                rostro BLOB NOT NULL,
                FOREIGN KEY (id_persona) REFERENCES Persona (id)
            )
        """)
        print("Tabla rostro creada")

        # Detection table (v2 schema with separated confidences and nullable id_persona)
        # Desired schema:
        # Deteccion(
        #   id INTEGER PK,
        #   id_persona INTEGER NULL,
        #   recog_confianza REAL NULL,
        #   emocion TEXT NOT NULL,
        #   emocion_confianza REAL NOT NULL,
        #   timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #   FOREIGN KEY(id_persona) REFERENCES Persona(id)
        # )

        # Create initial table if it does not exist at all
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Deteccion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_persona INTEGER,
                recog_confianza REAL,
                emocion TEXT NOT NULL,
                emocion_confianza REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (id_persona) REFERENCES Persona (id)
            )
        """)
        print("Tabla deteccion creada o actualizada")

        # Emotion detail table (full distribution per detection)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS DeteccionEmocionDetalle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_deteccion INTEGER NOT NULL,
                emocion TEXT NOT NULL,
                confianza REAL NOT NULL,
                FOREIGN KEY (id_deteccion) REFERENCES Deteccion (id)
            )
            """
        )
        print("Tabla detalle de emociones creada")

        # Migration path from legacy schema if needed
        try:
            cursor.execute("PRAGMA table_info(Deteccion)")
            cols = cursor.fetchall()
            col_names = [c[1] for c in cols]

            legacy_cols = {"id", "id_persona", "emocion", "confianza", "timestamp"}
            desired_cols = {"id", "id_persona", "recog_confianza", "emocion", "emocion_confianza", "timestamp"}

            # If table still has legacy layout, migrate
            if set(col_names) == legacy_cols:
                print("Migrando tabla Deteccion a nuevo esquema...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS Deteccion_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        id_persona INTEGER,
                        recog_confianza REAL,
                        emocion TEXT NOT NULL,
                        emocion_confianza REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (id_persona) REFERENCES Persona (id)
                    )
                """)
                # Move data: legacy 'confianza' corresponds to 'emocion_confianza'; recog_confianza unknown -> NULL
                cursor.execute("""
                    INSERT INTO Deteccion_v2 (id_persona, recog_confianza, emocion, emocion_confianza, timestamp)
                    SELECT id_persona, NULL as recog_confianza, emocion, confianza as emocion_confianza, timestamp FROM Deteccion
                """)
                cursor.execute("DROP TABLE Deteccion")
                cursor.execute("ALTER TABLE Deteccion_v2 RENAME TO Deteccion")
                print("Migración de Deteccion completada")
            else:
                # Ensure desired columns exist; if missing, perform additive migration
                needs_additive = False
                if "recog_confianza" not in col_names:
                    cursor.execute("ALTER TABLE Deteccion ADD COLUMN recog_confianza REAL")
                    needs_additive = True
                if "emocion_confianza" not in col_names:
                    cursor.execute("ALTER TABLE Deteccion ADD COLUMN emocion_confianza REAL NOT NULL DEFAULT 0.0")
                    needs_additive = True
                # Allow id_persona to be NULL is already satisfied in SQLite if defined without NOT NULL
                if needs_additive:
                    print("Esquema Deteccion actualizado con columnas adicionales")
        except Exception as e:
            print(f"Aviso: no se pudo verificar/migrar esquema de Deteccion: {e}")

        self.conn.commit()
        cursor.close()
        print("Tablas listas")

    # Note: All camera, DeepFace and recognition logic has been moved to camera_local.py.

    def get_cascade(self):
        if self._cascade is None:
            from cv2 import data as cv2_data
            self._cascade = cv2.CascadeClassifier(
                cv2_data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        return self._cascade

    def get_all_persons(self):
        """Get all registered persons"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, nombre, apellido, email FROM Persona ORDER BY nombre, apellido")
            return cursor.fetchall()
        except Exception as e:
            st.error(f"Error obteniendo personas: {e}")
            return []

    def get_person_name(self, person_id):
        """Get person name by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT nombre, apellido FROM Persona WHERE id = ?", (person_id,))
            result = cursor.fetchone()
            return f"{result[0]} {result[1]}" if result else "Desconocido"
        except Exception as e:
            st.error(f"Error obteniendo nombre: {e}")
            return "Desconocido"

# initialize system (Streamlit web page)
if 'system' not in st.session_state:
    st.session_state.system = FacialRecognitionSystem()
if 'reg_captures' not in st.session_state:
    # Holds captured registration images as raw bytes
    st.session_state.reg_captures = []
if 'reg_hashes' not in st.session_state:
    st.session_state.reg_hashes = set()
if 'reg_form_ok' not in st.session_state:
    st.session_state.reg_form_ok = False
if 'reg_form_data' not in st.session_state:
    st.session_state.reg_form_data = {"nombre":"", "apellido":"", "email":""}
if 'reg_embs' not in st.session_state:
    # Optional cache: map image hash -> embedding ndarray (to speed up save)
    st.session_state.reg_embs = {}
if 'native_proc_pid' not in st.session_state:
    st.session_state.native_proc_pid = None

def main():
    st.sidebar.title("Sistema de Reconocimiento Facial")
    page = st.sidebar.radio(
        "Navegación",
        options=["Registro", "Detección", "Reportes"],
        index=0,
        help="Selecciona una sección"
    )

    if page == "Registro":
        show_registration_page()
    elif page == "Detección":
        show_detection_page()
    else:
        show_reports_page()
# PAGE FUNCTIONS (REPORTS PAGE)

# emotions chart
def show_emotion_charts():
    st.subheader("Distribucion de emociones")

    try: 
        cursor = st.session_state.system.conn.cursor()  

        # obtain data from table
        cursor.execute("""
            SELECT COALESCE(P.nombre || ' ' || P.apellido, 'Desconocido') as persona,
                   D.emocion,
                   COUNT(*) as conteo
            FROM Deteccion D
            LEFT JOIN Persona P ON D.id_persona = P.id
            GROUP BY persona, emocion
        """)

        data = cursor.fetchall()

        if data:

            df_data = []
            for row in data:
                # normalize column names to use in the chart
                df_data.append({
                    "Persona": row[0],
                    "Emocion": row[1],
                    "Cantidad": row[2]
                })
            df = pd.DataFrame(df_data)

            # emotion graph
            fig = px.bar(df, x='Persona', y='Cantidad', color='Emocion',
                        title='Distribución de Emociones por Persona')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de detecciones para mostrar")
    except Exception as e:
        st.error(f"Error al generar grafico de emociones: {e}")

# general stats
def show_general_stats():
    st.subheader("Estadisticas generales")

    try:
        cursor = st.session_state.system.conn.cursor()

        col1, col2, col3, col4 = st.columns(4)

        # registered persons
        cursor.execute("SELECT COUNT(*) FROM Persona")
        total_personas = cursor.fetchone()[0]

        # detections today (localtime window)
        cursor.execute(
            """
            SELECT COUNT(*) FROM Deteccion
            WHERE timestamp >= datetime('now','localtime','start of day')
              AND timestamp <  datetime('now','localtime','start of day','+1 day')
            """
        )
        detecciones_hoy = cursor.fetchone()[0]

        # most common emotion
        cursor.execute(
            """
            SELECT emocion, COUNT(*) as count
            FROM Deteccion
            GROUP BY emocion
            ORDER BY count DESC
            LIMIT 1
            """
        )
        emocion_data = cursor.fetchone()
        emocion_predominante = emocion_data[0] if emocion_data else "N/A"

        # recognition rate (estimated) using recog_confianza
        cursor.execute("SELECT COUNT(*) FROM Deteccion WHERE recog_confianza IS NOT NULL AND recog_confianza > 0.7")
        reconocimientos_confiables = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Deteccion WHERE recog_confianza IS NOT NULL")
        total_reconocimientos = cursor.fetchone()[0]
        tasa_reconocimiento = (reconocimientos_confiables / total_reconocimientos * 100) if total_reconocimientos > 0 else 0

        # metrics
        with col1:
            st.metric("Total de personas registradas", total_personas)
        with col2:
            st.metric("Detecciones realizadas (hoy)", detecciones_hoy)
        with col3:
            st.metric("Emocion predominante", emocion_predominante)
        with col4:
            st.metric("Tasa de reconocimientos confiables", f"{tasa_reconocimiento:.1f}%")

        # query for detections per hour
        st.subheader("Patrón de detecciones por hora")
        cursor.execute(
            """
            SELECT STRFTIME('%H', timestamp, 'localtime') as hora, COUNT(*) as count
            FROM Deteccion
            GROUP BY hora
            ORDER BY hora
            """
        )

        hora_data = cursor.fetchall()
        if hora_data:
            horas = [f"{int(h[0]):02d}:00" for h in hora_data]
            counts = [h[1] for h in hora_data]
            fig = px.line(
                x=horas,
                y=counts,
                title='Patrón de Detecciones por Hora',
                labels={'x': 'Hora del día', 'y': 'Número de detecciones'}
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error al mostrar estadisticas generales: {e}")


# PAGE: Registro
def show_registration_page():
    st.title("Registro de Personas")
    with st.form("form-registro"):
        c1, c2, c3 = st.columns(3)
        with c1:
            nombre = st.text_input("Nombre", value=st.session_state.reg_form_data.get("nombre", ""))
        with c2:
            apellido = st.text_input("Apellido", value=st.session_state.reg_form_data.get("apellido", ""))
        with c3:
            email = st.text_input("Email", value=st.session_state.reg_form_data.get("email", ""))
        submitted = st.form_submit_button("Confirmar datos")

    if submitted:
        if not (nombre and apellido and email):
            st.error("Completa nombre, apellido y email.")
        else:
            st.success("Datos validados. Captura imágenes a continuación.")
            st.session_state.reg_form_ok = True
            st.session_state.reg_form_data = {"nombre": nombre, "apellido": apellido, "email": email}

    st.markdown("---")
    st.subheader("Capturas de rostro")
    st.caption("Toma al menos 5 fotos con buena nitidez, iluminación y el rostro centrado.")

    c1, c2 = st.columns([2,1])
    with c1:
        img_upload = st.camera_input("Tomar foto", disabled=not st.session_state.reg_form_ok)
        col_btn = st.columns(1)[0]
        with col_btn:
            if st.button("Limpiar capturas"):
                st.session_state.reg_captures = []
                st.session_state.reg_hashes = set()
                st.session_state.reg_embs = {}
        if img_upload is not None:
            raw = img_upload.getvalue()
            h = hashlib.sha256(raw).hexdigest()
            if h not in st.session_state.reg_hashes:
                st.session_state.reg_captures.append(raw)
                st.session_state.reg_hashes.add(h)
                st.success("Captura agregada")
    with c2:
        st.write(f"Capturas acumuladas: {len(st.session_state.reg_captures)}")
        save_now = st.button("Guardar en base de datos", disabled=(not st.session_state.reg_form_ok or len(st.session_state.reg_captures) == 0))

    # Preview de thumbnails
    if st.session_state.reg_captures:
        cols = st.columns(3)
        for i, raw in enumerate(st.session_state.reg_captures[-6:]):
            with cols[i % 3]:
                st.image(Image.open(io.BytesIO(raw)), caption=f"Captura #{i+1}", use_container_width=True)

    # Guardado
    if save_now:
        try:
            conn = st.session_state.system.conn
            cur = conn.cursor()
            # Crear/obtener persona
            cur.execute("SELECT id FROM Persona WHERE email = ?", (st.session_state.reg_form_data["email"],))
            row = cur.fetchone()
            if row:
                person_id = row[0]
            else:
                cur.execute(
                    "INSERT INTO Persona(nombre, apellido, email) VALUES(?,?,?)",
                    (st.session_state.reg_form_data["nombre"], st.session_state.reg_form_data["apellido"], st.session_state.reg_form_data["email"])
                )
                person_id = cur.lastrowid

            saved = 0
            cascade = st.session_state.system.get_cascade()
            for raw in st.session_state.reg_captures:
                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    frame_rgb = np.array(img)
                    # Detect and crop largest face for consistent embeddings
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    if len(faces) > 0:
                        (x, y, w, h) = max(faces, key=lambda b: b[2]*b[3])
                        # Clamp within bounds
                        x0 = max(0, x); y0 = max(0, y)
                        x1 = min(frame_rgb.shape[1], x + w)
                        y1 = min(frame_rgb.shape[0], y + h)
                        crop_rgb = frame_rgb[y0:y1, x0:x1]
                    else:
                        # Fallback to center crop if no face detected
                        h, w = frame_rgb.shape[:2]
                        cx0 = int(w*0.2); cy0 = int(h*0.2)
                        cx1 = int(w*0.8); cy1 = int(h*0.8)
                        crop_rgb = frame_rgb[cy0:cy1, cx0:cx1]
                    emb = extract_embedding(crop_rgb)
                    if emb is None:
                        continue
                    blob = pickle.dumps(emb)
                    cur.execute("INSERT INTO Rostro(id_persona, rostro) VALUES(?, ?)", (person_id, blob))
                    saved += 1
                except Exception:
                    continue
            conn.commit()
            st.success(f"Guardado: {saved} capturas para {st.session_state.reg_form_data['nombre']} {st.session_state.reg_form_data['apellido']}")
            # limpiar estado
            st.session_state.reg_captures = []
            st.session_state.reg_hashes = set()
            st.session_state.reg_embs = {}
            st.session_state.reg_form_ok = False
        except Exception as e:
            st.error(f"No se pudo guardar: {e}")


# PAGE: Detección (nativo)
def show_detection_page():
    st.title("Detección en Tiempo Real (Nativo OpenCV)")
    st.caption("Se abrirá una ventana nativa con el video y overlay. Cierra con 'q'.")

    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Umbral reconocimiento (coseno)", 0.5, 0.95, 0.6, 0.01)
        infer_ms = st.slider("Intervalo inferencia (ms)", 100, 2000, 500, 50)
        detect_scale = st.slider("Escala detección rostro", 0.2, 1.0, 0.5, 0.05)
        infer_scale = st.slider("Escala de inferencia (rostro)", 0.3, 1.0, 0.5, 0.05)
        detector_backend = st.selectbox("Detector de rostro (caja verde)", options=['opencv','opencv-dnn','retinaface','mediapipe'], index=0)
        embed_model = st.selectbox("Modelo de embeddings", options=['ArcFace','Facenet','VGG-Face'], index=1)
    with c2:
        emo_backend = st.selectbox("Backend emociones (DeepFace)", options=['opencv','retinaface','mediapipe','skip'], index=0)
        emo_ms = st.slider("Intervalo emociones (ms)", 500, 10000, 3000, 50)
        crop_padding = st.slider("Padding recorte emociones", 0.0, 0.3, 0.15, 0.01)
        emo_scale = st.slider("Escala emociones (upscale)", 1.0, 2.0, 1.2, 0.1)
        emo_smooth = st.slider("Suavizado emociones (frames)", 1, 15, 5, 1)
        emo_disgust_gain = st.slider("Ganancia 'disgusto' (calibración)", 0.5, 3.0, 1.0, 0.1, help="Multiplica la probabilidad de 'disgust' antes de normalizar. Útil si nunca aparece.")
        disable_emotion = st.checkbox("Deshabilitar emociones (máximo FPS)", value=False)
        min_log_ms = st.slider("Intervalo mínimo de guardado (ms)", 1000, 20000, 5000, 100, help="Limita la frecuencia con la que se escriben registros en la base de datos. Si hay una nueva emoción, se guardará aunque no se cumpla este intervalo.")

    st.subheader("Ajustes de cámara")
    colr1, colr2 = st.columns(2)
    with colr1:
        res_option = st.selectbox(
            "Resolución",
            ["640x480 (SD)", "1280x720 (HD)", "1920x1080 (FHD)", "Personalizada"],
            index=1,
            help="Aumentar resolución mejora calidad pero puede reducir FPS"
        )
    with colr2:
        cam_idx = st.number_input("Índice de cámara", min_value=0, max_value=10, value=0, step=1)

    if res_option == "640x480 (SD)":
        cam_w, cam_h = 640, 480
    elif res_option == "1280x720 (HD)":
        cam_w, cam_h = 1280, 720
    elif res_option == "1920x1080 (FHD)":
        cam_w, cam_h = 1920, 1080
    else:
        cwx, cwy = st.columns(2)
        with cwx:
            cam_w = st.number_input("Ancho (px)", min_value=320, max_value=3840, value=1280, step=16)
        with cwy:
            cam_h = st.number_input("Alto (px)", min_value=240, max_value=2160, value=720, step=16)

    # Camera tuning (outside button for proper UI rendering)
    st.subheader("Tuning de rendimiento")
    force_mjpg = st.checkbox('Forzar MJPG (reduce latencia CPU)', value=True, key='mjpg')
    target_fps = st.slider('FPS objetivo (intentar)', 0, 60, 30, 1, key='fps')
    box_alpha = st.slider('Suavizado caja (alpha)', 0.0, 1.0, 0.5, 0.05, key='boxalpha')

    colb1, colb2, colb3 = st.columns([1,1,2])
    can_start = st.session_state.native_proc_pid is None
    with colb1:
        if st.button("Iniciar", disabled=not can_start):
            try:
                cmd = [
                    sys.executable,
                    os.path.join(os.getcwd(), 'camera_local.py'),
                    'detect',
                    '--camera', str(int(cam_idx)),
                    '--threshold', str(float(threshold)),
                    '--infer-interval-ms', str(int(infer_ms)),
                    '--scale', str(float(infer_scale)),
                    '--detect-scale', str(float(detect_scale)),
                    '--frame-width', str(int(cam_w)),
                    '--frame-height', str(int(cam_h)),
                    '--detector-backend', str(detector_backend),
                    '--embed-model', str(embed_model),
                ]
                # Emotions flags only if enabled
                if not disable_emotion and emo_backend != 'skip':
                    cmd += [
                        '--emotion-interval-ms', str(int(emo_ms)),
                        '--emotion-backend', str(emo_backend),
                        '--emo-smooth-frames', str(int(emo_smooth)),
                        '--emotion-scale', str(float(emo_scale)),
                        '--crop-padding', str(float(crop_padding)),
                        '--emo-disgust-gain', str(float(emo_disgust_gain)),
                        '--min-log-interval-ms', str(int(min_log_ms)),
                    ]
                else:
                    cmd += ['--no-emotion']
                # Camera tuning
                if force_mjpg:
                    cmd += ['--force-mjpg']
                if target_fps > 0:
                    cmd += ['--target-fps', str(int(target_fps))]
                cmd += ['--box-smooth-alpha', str(float(box_alpha))]
                creationflags = 0
                if os.name == 'nt' and hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                    creationflags = subprocess.CREATE_NEW_CONSOLE
                proc = subprocess.Popen(cmd, creationflags=creationflags)
                st.session_state.native_proc_pid = proc.pid
                st.success(f"Detección nativa iniciada (PID {proc.pid}). Cierra con 'q' en la ventana o usa Detener.")
            except Exception as e:
                st.error(f"No se pudo iniciar la detección nativa: {e}")
    with colb2:
        stop_enabled = st.session_state.native_proc_pid is not None
        if st.button("Detener", disabled=not stop_enabled):
            try:
                pid = st.session_state.native_proc_pid
                if pid is not None:
                    if os.name == 'nt':
                        subprocess.call(['taskkill', '/PID', str(pid), '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        os.kill(pid, 9)
                    st.session_state.native_proc_pid = None
                    st.info("Detección nativa detenida")
            except Exception as e:
                st.error(f"No se pudo detener: {e}")
    with colb3:
        pid = st.session_state.native_proc_pid
        st.write(f"Estado: {'En ejecución (PID '+str(pid)+')' if pid else 'Detenido'}")


# PAGE: Reportes
def show_reports_page():
    st.title("Reportes")
    show_general_stats()
    st.markdown("---")
    show_emotion_charts()
    st.markdown("---")
    # Ultimas detecciones
    try:
        cur = st.session_state.system.conn.cursor()
        cur.execute(
            """
            SELECT D.timestamp, COALESCE(P.nombre || ' ' || P.apellido, 'Desconocido') as persona,
                   D.recog_confianza, D.emocion, D.emocion_confianza
            FROM Deteccion D
            LEFT JOIN Persona P ON D.id_persona = P.id
            ORDER BY D.timestamp DESC
            LIMIT 100
            """
        )
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "persona", "recog_confianza", "emocion", "emocion_confianza"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay detecciones registradas")
    except Exception as e:
        st.error(f"Error al cargar detecciones: {e}")


if __name__ == "__main__":
    main()