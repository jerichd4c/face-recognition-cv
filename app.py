# libraries

from datetime import datetime
import io
import sqlite3
import time
import pickle
from typing import Optional, Tuple

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

try:
    # Optional dependency: used for real-time camera streaming
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoTransformerBase,
        WebRtcMode,
        RTCConfiguration,
    )
    _WEBRTC_AVAILABLE = True
except Exception:
    _WEBRTC_AVAILABLE = False

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

def main():
    st.sidebar.title("Sistema de Reconocimiento Facial")
    page = st.sidebar.radio(
        "Navegación",
        options=["Registro", "Detección", "Reportes"],
        index=2,
        help="Selecciona una sección"
    )

    if page == "Registro":
        show_registration_page()
    elif page == "Detección":
        show_detection_page()
    else:
        show_reports_page()

# reports page
def show_reports_page():
    st.title("Reportes y estadisticas")
    
    tab1, tab2, tab3 = st.tabs(["Graficos de emociones", "Estadisticas generales", "Historial de detecciones" ])

    with tab1:
        show_emotion_charts()
    with tab2:
        show_general_stats()
    with tab3:
        show_detection_history()

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

        col1 , col2, col3, col4 = st.columns(4)

        # registered persons
        cursor.execute("SELECT COUNT(*) FROM Persona")
        total_personas = cursor.fetchone()[0]

        # detections today
        cursor.execute("SELECT COUNT(*) FROM Deteccion WHERE DATE(timestamp) = DATE('now')")
        detecciones_hoy = cursor.fetchone()[0]

        # most common emotion
        cursor.execute("""
            SELECT emocion, COUNT(*) as count 
            FROM Deteccion 
            GROUP BY emocion 
            ORDER BY count DESC 
            LIMIT 1
        """)
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
        cursor.execute("""
            SELECT STRFTIME('%H', timestamp) as hora, COUNT(*) as count
            FROM Deteccion
            GROUP BY hora
            ORDER BY hora
        """)

    # detection graph per hour

        hora_data = cursor.fetchall()
        if hora_data:
            horas = [f"{int(h[0]):02d}:00" for h in hora_data]
            counts = [h[1] for h in hora_data]


            fig = px.line(x=horas, y=counts, title='Patrón de Detecciones por Hora',
                        labels={'x': 'Hora del día', 'y': 'Número de detecciones'})
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al mostrar estadisticas generales: {e}")

# history
def show_detection_history():
    st.subheader("Historial de detecciones")
    try:
        # filters
        col1, col2, col3 = st.columns(3)

        with col1:
            date_filter = st.date_input("Filtrar por fecha")
        with col2:
            cursor = st.session_state.system.conn.cursor()
            cursor.execute("SELECT id, nombre || ' ' || apellido FROM Persona")
            personas = cursor.fetchall()
            persona_options = ["Todas"] + [p[1] for p in personas]
            person_filter = st.selectbox("Filtrar por persona", persona_options)
        with col3:
            emotion_options = ["Todas", "happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
            emotion_filter = st.selectbox("Filtrar por emoción", emotion_options)

        # build query
        query = """
            SELECT D.timestamp, P.nombre, P.apellido, D.emocion, D.emocion_confianza, D.recog_confianza
            FROM Deteccion D
            LEFT JOIN Persona P ON D.id_persona = P.id
            WHERE 1=1
        """
        params = []

        # apply filters
        if date_filter:
            query += " AND DATE(D.timestamp) = ?"
            params.append(date_filter.strftime('%Y-%m-%d'))

        if person_filter != "Todas":
            query += " AND P.nombre || ' ' || P.apellido = ?"
            params.append(person_filter)

        if emotion_filter != "Todas":
            query += " AND D.emocion = ?"
            params.append(emotion_filter)

        query += " ORDER BY D.timestamp DESC"

        cursor = st.session_state.system.conn.cursor()
        cursor.execute(query, params)
        detections = cursor.fetchall()

        # display data
        if detections:
            df = pd.DataFrame(detections, columns=['Fecha/Hora', 'Nombre', 'Apellido', 'Emoción', 'Conf. Emoción', 'Conf. Reconocimiento'])

            def fmt_pct(x):
                try:
                    v = float(x) if x is not None else 0.0
                    return f"{v*100:.1f}%"
                except Exception:
                    return ""

            df['Conf. Emoción'] = df['Conf. Emoción'].apply(fmt_pct)
            df['Conf. Reconocimiento'] = df['Conf. Reconocimiento'].apply(lambda x: fmt_pct(x) if x is not None else "")
            st.dataframe(df, use_container_width=True)

            # save button CSV file
            if st.button("Guardar reporte en CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Descargar CSV",
                    data=csv,
                    file_name=f"detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No hay datos para mostrar con los filtros seleccionados")

    except Exception as e:
        st.error(f"Error al mostrar historial de detecciones: {e}")

# ============ REGISTRO ============
def _calc_quality(face_bgr: np.ndarray, full_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> dict:
    # Sharpness via Laplacian variance
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    # Brightness as mean intensity
    brightness = float(gray.mean())
    # Face ratio vs frame
    x,y,w,h = bbox
    H, W = full_bgr.shape[:2]
    face_ratio = (w*h) / float(W*H + 1e-6)
    return {"sharpness": sharpness, "brightness": brightness, "face_ratio": face_ratio}


def _detect_largest_face(bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    cascade = st.session_state.system.get_cascade()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return None
    (x,y,w,h) = max(faces, key=lambda b: b[2]*b[3])
    return int(x), int(y), int(w), int(h)


def _bgr_from_upload(upload_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(upload_bytes)).convert('RGB')
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def show_registration_page():
    st.title("Registro de Personas")

    with st.form("persona_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input("Nombre")
            email = st.text_input("Email")
        with col2:
            apellido = st.text_input("Apellido")
            force = st.checkbox("Forzar guardado si hay posible duplicado", value=False)
        submitted = st.form_submit_button("Confirmar datos")

    if submitted:
        if not (nombre and apellido and email):
            st.error("Completa nombre, apellido y email.")
        else:
            st.success("Datos validados. Captura imágenes a continuación.")

    st.markdown("---")
    st.subheader("Capturas de rostro")
    st.caption("Toma al menos 5 fotos con buena nitidez, iluminación y el rostro centrado.")

    c1, c2 = st.columns([2,1])
    with c1:
        img_upload = st.camera_input("Tomar foto")
        if img_upload is not None:
            # Add to session buffer
            st.session_state.reg_captures.append(img_upload.getvalue())
    with c2:
        if st.button("Limpiar capturas"):
            st.session_state.reg_captures = []
        st.write(f"Capturas acumuladas: {len(st.session_state.reg_captures)}")

    # Preview + quality
    if st.session_state.reg_captures:
        store = load_person_embeddings(st.session_state.system.conn)
        # find person id by email if exists (to allow add more)
        cursor = st.session_state.system.conn.cursor()
        cursor.execute("SELECT id FROM Persona WHERE email=?", (email,))
        row = cursor.fetchone()
        existing_pid = row[0] if row else None
        dup_threshold = st.slider("Umbral de duplicado (similitud coseno)", 0.5, 0.95, 0.8, 0.01)
        qual_cols = st.columns(3)
        for i, raw in enumerate(st.session_state.reg_captures[-6:]):
            bgr = _bgr_from_upload(raw)
            bbox = _detect_largest_face(bgr)
            with qual_cols[i % 3]:
                st.image(Image.open(io.BytesIO(raw)), caption=f"Captura #{i+1}", use_container_width=True)
                if bbox is None:
                    st.warning("No se detectó rostro")
                    continue
                x,y,w,h = bbox
                face = bgr[y:y+h, x:x+w]
                # Resize for embedding speed
                face_small = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (0,0), fx=0.5, fy=0.5)
                emb = extract_embedding(face_small)
                qm = _calc_quality(face, bgr, (x,y,w,h))
                st.write(f"Nitidez: {qm['sharpness']:.0f}")
                st.write(f"Brillo: {qm['brightness']:.0f}")
                st.write(f"Cobertura: {qm['face_ratio']*100:.1f}%")
                if emb is None:
                    st.error("No se pudo extraer embedding")
                    continue
                # duplicate check against other persons
                best_score = 0.0
                best_pid = None
                for pid, lst in store.items():
                    if existing_pid is not None and pid == existing_pid:
                        continue
                    if not lst:
                        continue
                    sims = [float(np.dot(emb, e)) for e in lst]
                    s = max(sims) if sims else 0.0
                    if s > best_score:
                        best_score, best_pid = s, pid
                if best_pid is not None and best_score >= dup_threshold:
                    st.warning(f"Posible duplicado (sim={best_score:.3f}) con persona id {best_pid}")
                else:
                    st.success("Sin duplicados aparentes")

    st.markdown("---")
    can_save = (nombre and apellido and email and len(st.session_state.reg_captures) >= 5)
    save_btn = st.button("Guardar registro (mín. 5 capturas)", disabled=not can_save)
    if save_btn:
        try:
            conn = st.session_state.system.conn
            cur = conn.cursor()
            # Create/find person
            cur.execute("SELECT id FROM Persona WHERE email=?", (email,))
            row = cur.fetchone()
            if row:
                person_id = row[0]
            else:
                cur.execute("INSERT INTO Persona(nombre, apellido, email) VALUES(?,?,?)", (nombre, apellido, email))
                conn.commit()
                person_id = cur.lastrowid
            # Load embeddings of others for duplicate enforcement
            store = load_person_embeddings(conn)
            saved = 0
            dup_threshold = 0.8
            for raw in st.session_state.reg_captures:
                bgr = _bgr_from_upload(raw)
                bbox = _detect_largest_face(bgr)
                if bbox is None:
                    continue
                x,y,w,h = bbox
                face = cv2.cvtColor(bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                face_small = cv2.resize(face, (0,0), fx=0.5, fy=0.5)
                emb = extract_embedding(face_small)
                if emb is None:
                    continue
                # duplicate check
                best_score = 0.0
                best_pid = None
                for pid, lst in store.items():
                    if pid == person_id:
                        continue
                    if not lst:
                        continue
                    sims = [float(np.dot(emb, e)) for e in lst]
                    s = max(sims) if sims else 0.0
                    if s > best_score:
                        best_score, best_pid = s, pid
                if best_pid is not None and best_score >= dup_threshold and not force:
                    # skip saving this one
                    continue
                cur.execute("INSERT INTO Rostro(id_persona, rostro) VALUES(?,?)", (person_id, sqlite3.Binary(pickle.dumps(emb))))
                conn.commit()
                saved += 1
            st.success(f"Registro guardado. Embeddings almacenados: {saved}")
            st.session_state.reg_captures = []
        except Exception as e:
            st.error(f"Error al guardar registro: {e}")


# ============ DETECCIÓN ============
def show_detection_page():
    st.title("Detección en Tiempo Real")
    if not _WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc no está instalado. Agrega 'streamlit-webrtc' y 'av' a requirements.txt")
        st.stop()

    st.caption("Video con overlay de nombre, emoción y confidencias. Presiona el botón para iniciar/detener.")

    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Umbral reconocimiento (coseno)", 0.5, 0.95, 0.6, 0.01)
        infer_ms = st.slider("Intervalo inferencia (ms)", 100, 2000, 500, 50)
        detect_scale = st.slider("Escala detección rostro", 0.2, 1.0, 0.5, 0.05)
    with c2:
        emo_backend = st.selectbox("Backend emociones", options=['opencv','retinaface','mediapipe','skip'], index=0)
        emo_ms = st.slider("Intervalo emociones (ms)", 500, 3000, 1500, 50)
        crop_padding = st.slider("Padding recorte emociones", 0.0, 0.3, 0.15, 0.01)

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    import av  # ensure available for VideoFrame conversions

    class Transformer(VideoTransformerBase):
        def __init__(self) -> None:
            super().__init__()
            # Separate DB connection for transformer thread
            self.conn = sqlite3.connect('facial_recognition.db', check_same_thread=False)
            self.cur = self.conn.cursor()
            self.store = load_person_embeddings(self.conn)
            # Instantiate cascade locally (do not rely on session_state in worker thread)
            from cv2 import data as cv2_data
            self.cascade = cv2.CascadeClassifier(
                cv2_data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.last_ts = 0.0
            self.last_emo_ts = 0.0
            self.label = 'Desconocido'
            self.recog_conf = 0.0
            self.emotion = 'neutral'
            self.emo_conf = 0.0
            self.last_box = None
            self.top3 = []  # list of (emo, conf)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            now = time.time()
            # Run heavy inference at configured cadence
            if (now - self.last_ts) >= (infer_ms/1000.0):
                self.last_ts = now
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ds = float(detect_scale)
                    small = cv2.resize(gray, (0,0), fx=ds, fy=ds)
                    faces = self.cascade.detectMultiScale(small, 1.1, 5, minSize=(max(30, int(60*ds)), max(30, int(60*ds))))
                    pid = None
                    if len(faces) > 0:
                        xs, ys, ws, hs = max(faces, key=lambda b: b[2]*b[3])
                        x = int(xs/ds); y = int(ys/ds); w = int(ws/ds); h = int(hs/ds)
                        pad = float(crop_padding)
                        if pad > 0:
                            px = int(w*pad); py = int(h*pad)
                            x0 = max(0, x-px); y0 = max(0, y-py)
                            x1 = min(img.shape[1], x+w+px); y1 = min(img.shape[0], y+h+py)
                        else:
                            x0,y0,x1,y1 = x,y,x+w,y+h
                        self.last_box = (x0,y0,x1-x0,y1-y0)
                        face_rgb = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                        small_rgb = cv2.resize(face_rgb, (0,0), fx=0.5, fy=0.5)
                        emb = extract_embedding(small_rgb)
                        if emb is not None:
                            pid, rconf = recognize(emb, self.store, threshold)
                            self.recog_conf = float(rconf)
                            if pid is not None:
                                self.cur.execute("SELECT nombre, apellido FROM Persona WHERE id=?", (pid,))
                                row = self.cur.fetchone()
                                if row:
                                    self.label = f"{row[0]} {row[1]}"
                                else:
                                    self.label = 'Desconocido'
                            else:
                                self.label = 'Desconocido'
                        # Emotions at its own cadence
                        if (now - self.last_emo_ts) >= (emo_ms/1000.0) and emo_backend != 'skip':
                            e_label, e_conf, e_dist = analyze_emotion_full(face_rgb, backend=emo_backend)
                            self.emotion = e_label
                            self.emo_conf = float(e_conf)
                            # Compute top-3
                            items = sorted(e_dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
                            self.top3 = items
                            self.last_emo_ts = now
                            # Persist detection with detail
                            self.cur.execute(
                                "INSERT INTO Deteccion(id_persona, recog_confianza, emocion, emocion_confianza, timestamp) VALUES(?,?,?,?,?)",
                                (pid, self.recog_conf if pid is not None else None, self.emotion, self.emo_conf, datetime.now())
                            )
                            det_id = self.cur.lastrowid
                            try:
                                for emo, confv in e_dist.items():
                                    self.cur.execute(
                                        "INSERT INTO DeteccionEmocionDetalle(id_deteccion, emocion, confianza) VALUES(?,?,?)",
                                        (det_id, emo, float(confv))
                                    )
                            except Exception:
                                pass
                            self.conn.commit()
                    else:
                        self.last_box = None
                except Exception:
                    # keep last known state on errors
                    pass
            # Always draw the latest overlay
            if self.last_box is not None:
                x,y,w,h = self.last_box
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, f"Persona: {self.label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(img, f"Conf. recog: {self.recog_conf*100:.1f}%  Emocion: {self.emotion} ({self.emo_conf*100:.1f}%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            # Top-3 emotions line
            if self.top3:
                txt = "  ".join([f"{k}:{v*100:.0f}%" for k, v in self.top3])
                cv2.putText(img, f"Top-3: {txt}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def __del__(self):
            try:
                self.cur.close()
                self.conn.close()
            except Exception:
                pass

    webrtc_streamer(
        key="detect-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=Transformer,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

if __name__ == "__main__":
    main()