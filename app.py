# libraries

from datetime import datetime
import sqlite3
import streamlit as st
import os
import pickle
import numpy as np
import time
import pandas as pd
import plotly.express as px
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# initial page config
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    layout="wide",
    page_icon="📷"
)

class FacialRecognitionSystem():
    def __init__(self):
        self.init_database()
        self.load_face_models()
        self.detection_active = False
        self.cap = None
            
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

        # Detection table

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Deteccion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_persona INTEGER NOT NULL,
                emocion TEXT NOT NULL,
                confianza REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                FOREIGN KEY (id_persona) REFERENCES Persona (id)
            )
        """)
        print("Tabla deteccion creada")

        self.conn.commit()
        cursor.close()
        print( "Tablas creadas")

    # load facial recognition and embedding model

    def load_face_models(self):

        try:
            if os.path.exists('face_embeddings.pk1'):
                with open('face_embeddings.pk1', 'rb') as f:
                    self.face_embeddings = pickle.load(f)
            else:
                self.face_embeddings = {}

            if os.path.exists("persona_embeddings.pk1"):
                with open("persona_embeddings.pk1", "rb") as f:
                    self.persona_embeddings = pickle.load(f)
            else:
                self.persona_embeddings = {}

        except Exception as e:
            st.error(f"Error al cargar modelos de reconocimiento facial: {e}")
            self.known_embeddings = {}
            self.persona_embeddings = {}

    # save face embeddings

    def save_face_embeddings(self):

        try: 
            with open('face_embeddings.pk1', 'wb') as f:
                pickle.dump(self.face_embeddings, f)

            with open('persona_embeddings.pk1', 'wb') as f:
                pickle.dump(self.persona_embeddings, f)

        except Exception as e:
            st.error(f"Error al guardar embeddings: {e}")

    # extract face embeddings

    def extract_embeddings(self, image_array):

        try: 
            # uses DeepFace
            result = DeepFace.represent(
                img_path = image_array,
                model_name = "Facenet",
                enforce_detection = False,
                detector_backend = 'opencv'
            )
            if result:
                return result[0]["embedding"]
            return None
        except Exception as e:
            st.error(f"Error al extraer embeddings: {e}")
            return None

    # analyze emotions

    def analyze_emotions(self, image_array):

        try:
            analysis = DeepFace.analyze(
            img_path = image_array,
            actions = ['emotion'],
            enforce_detection = False,
            detector_backend = 'opencv'
            )
            if analysis:
                emotion = analysis[0]["dominant_emotion"]
                confidence = analysis[0]["emotion"][emotion] / 100
                return emotion, confidence
            return "neutral", 0.0 
        except Exception as e:
            st.error(f"Error al analizar emociones: {e}")
            return "neutral", 0.0
    
    # recognize face

    def recognize_face(self, embedding, threshold=0.6): 

        try: 
            best_match = None
            best_score = 0

            for person_id, stored_embedding in self.persona_embeddings.items():
                
                # calc similarity

                similarity = cosine_similarity([embedding], [stored_embedding])[0][0]

                if similarity > best_score and similarity > threshold:
                    best_score = similarity
                    best_match = person_id

            return best_match, best_score

        except Exception as e:
            st.error(f"Error al reconocer rostro: {e}")
            return None, 0 

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

# initialize system (st web page)
if 'system' not in st.session_state:
    st.session_state.system = FacialRecognitionSystem()

# session variables
if 'capture_active' not in st.session_state:
    st.session_state.capture_active = False
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'current_person_id' not in st.session_state:
    st.session_state.current_person_id = None
if 'captured_embeddings' not in st.session_state:
    st.session_state.captured_embeddings = 0
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None

def main():
    st.sidebar.title("Sistema de Reconocimiento Facial")
    page = st.sidebar.radio(
        "Navegacion",
        ["Registro", "Deteccion en Tiempo Real", "Reportes y estadisticas"]
    )

    if page == "Registro":
        show_registration_page()
    elif page == "Deteccion en Tiempo Real":
        show_detection_page()
    elif page == "Reportes y estadisticas":
        show_reports_page()

# registration page
def show_registration_page():
    st.title("Registro de personas")

    col1, col2 = st.columns([1,1])

    with col1: 
        st.subheader("Selección/Registro de Persona")

        # option 1: Select existing person
        personas = st.session_state.system.get_all_persons()
        personas_options = ["Nueva Persona"] + [f"{p[1]} {p[2]} ({p[3]})" for p in personas]

        selected_person = st.selectbox("Seleccionar persona existente:", personas_options)

        if selected_person != "Nueva Persona":
            # extract person ID
            persona_id = next(p[0] for p in personas if f"{p[1]} {p[2]} ({p[3]})" == selected_person)
            st.session_state.current_person_id = persona_id
            st.success(f"Persona seleccionada: {selected_person}")

        # option 2: Register new person
        st.subheader("O registrar nueva persona:")
        with st.form("registration_form"):
            nombre = st.text_input("Nombre")
            apellido = st.text_input("Apellido")
            email = st.text_input("Email")

            submitted = st.form_submit_button("Registrar Nueva Persona")

            if submitted:
                if nombre and apellido and email:
                    person_id = register_person(nombre, apellido, email)
                    if person_id:
                        st.session_state.current_person_id = person_id
                        st.session_state.captured_embeddings = 0
                        st.success(f"Persona registrada exitosamente (ID: {person_id})")
                        st.rerun()
                else:
                    st.error("Por favor complete todos los campos")

    with col2: 
        st.subheader("Captura Facial")

        if st.session_state.current_person_id is None:
            st.warning("Primero seleccione o registre una persona")
            return

        # show current person and captured embeddings
        persona_nombre = st.session_state.system.get_person_name(st.session_state.current_person_id)
        st.info(f"Persona actual: **{persona_nombre}**")
        st.info(f"Embeddings capturados: **{st.session_state.captured_embeddings}/5**")

        # camera selection
        camera_index = st.selectbox("Seleccionar cámara", [0, 1, 2], key="capture_cam")

        # start/Stop preview buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("🟢 Iniciar Vista Previa") and not st.session_state.capture_active:
                st.session_state.capture_active = True
                st.session_state.camera_index = camera_index
                st.rerun()

        with col_stop:
            if st.button("🔴 Detener Vista Previa") and st.session_state.capture_active:
                st.session_state.capture_active = False
                st.session_state.last_frame = None
                st.success("Vista previa detenida")
                st.rerun()

        # show preview if active
        if st.session_state.capture_active:
            display_camera_preview()

            # button to capture embedding
            if st.button("📸 Capturar Rostro Actual", type="primary"):
                capture_current_frame()

    # search person
    st.divider()
    st.subheader("Buscar persona por email")
    email = st.text_input("Escribe el email de la persona a buscar", key="search_email")
    if st.button("Buscar persona"):
        show_registered_person(email)

def display_camera_preview():
    """Mostrar vista previa de la cámara en tiempo real"""
    try:
        cap = cv2.VideoCapture(st.session_state.camera_index)
        if not cap.isOpened():
            st.error("No se pudo abrir la cámara")
            return

        # read a frame
        ret, frame = cap.read()
        if ret:
            # convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = rgb_frame

            # show frame
            st.image(rgb_frame, caption="Vista previa - Posicione su rostro y haga clic en 'Capturar Rostro'", 
                    use_column_width=True)

            # quality indicator
            embedding = st.session_state.system.extract_embeddings(rgb_frame)
            if embedding is not None:
                st.success("✅ Rostro detectado - Listo para capturar")
            else:
                st.warning("⚠️ No se detectó un rostro claro")

        cap.release()

        # auto-refresh for real-time preview
        time.sleep(0.1)
        st.rerun()

    except Exception as e:
        st.error(f"Error en vista previa: {e}")

def capture_current_frame():
    """Capturar el frame actual y extraer embedding"""
    if st.session_state.last_frame is None:
        st.error("No hay frame disponible para capturar")
        return

    try:
        # extract embedding
        embedding = st.session_state.system.extract_embeddings(st.session_state.last_frame)

        if embedding is not None:
            # store embedding in database
            cursor = st.session_state.system.conn.cursor()
            embedding_blob = pickle.dumps(embedding)
            cursor.execute(
                "INSERT INTO Rostro (id_persona, rostro) VALUES (?, ?)",
                (st.session_state.current_person_id, embedding_blob)
            )
            st.session_state.system.conn.commit()

            # update in-memory embeddings
            st.session_state.system.persona_embeddings[st.session_state.current_person_id] = embedding
            st.session_state.system.save_face_embeddings()

            st.session_state.captured_embeddings += 1
            st.success(f"✅ Embedding {st.session_state.captured_embeddings}/5 capturado exitosamente")

            if st.session_state.captured_embeddings >= 5:
                st.balloons()
                st.success("🎉 ¡Captura facial completada! Puede continuar con más capturas o cambiar de persona.")
        else:
            st.error("No se pudo extraer embedding. Asegúrese de que el rostro sea visible y esté bien iluminado.")

    except Exception as e:
        st.error(f"Error capturando embedding: {e}")

# show registered person
def show_registered_person(email):
    cursor = st.session_state.system.conn.cursor()
    cursor.execute("SELECT * FROM Persona WHERE email=?", (email,))
    person = cursor.fetchone()

    if person:
        st.write("Persona encontrada")
        st.write(f"Nombre: {person[1]}")
        st.write(f"Apellido: {person[2]}")
        st.write(f"Email: {person[3]}")
    else:
        st.write("Persona no encontrada")

# register person
def register_person(nombre, apellido, email):
    try: 
        cursor = st.session_state.system.conn.cursor()
        
        # verify email with id
        cursor.execute("SELECT id from Persona WHERE email = ?", (email,))
        if cursor.fetchone():
            st.error("Email ya registrado")
            return None
        
        # insert new person 
        cursor.execute("INSERT INTO Persona (nombre, apellido, email) VALUES (?, ?, ?)", (nombre, apellido, email))
        person_id = cursor.lastrowid
        st.session_state.system.conn.commit()
        return person_id
    
    except Exception as e:
        st.error(f"Error al registrar persona: {e}")
        return None

# detection page
def show_detection_page():
     st.title("Deteccion en tiempo real")

     col1, col2 = st.columns([2,1])

     with col1:
        st.subheader("Vista de camara")

        # detection controls
        col_controls = st.columns(3)
        with col_controls[0]:
            camera_select = st.selectbox("Camara", [0, 1, 2], key="detection_cam")
        with col_controls[1]:
            if st.button("🟢 Iniciar Detección") and not st.session_state.detection_active:
                st.session_state.detection_active = True
                st.session_state.detection_camera = camera_select
                st.rerun()
        with col_controls[2]:
            if st.button("🔴 Detener Detección") and st.session_state.detection_active:
                st.session_state.detection_active = False
                st.session_state.last_detection = None
                st.success("Detección detenida")
                st.rerun()

        # show real-time detection if active
        if st.session_state.detection_active:
            display_real_time_detection()

     with col2:
        st.subheader("Informacion de Deteccion")
        display_realtime_info()

        # last detection
        st.subheader("Ultima detecciones hechas")
        display_recent_detections()

def display_real_time_detection():
    """Mostrar detección en tiempo real sin usar hilos"""
    try:
        cap = cv2.VideoCapture(st.session_state.detection_camera)
        if not cap.isOpened():
            st.error("Error al abrir la cámara para detección")
            return

        # read a frame
        ret, frame = cap.read()
        if ret:
            # process frame for recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, detection_info = process_frame_for_detection(rgb_frame)

            # show processed frame
            st.image(processed_frame, caption="Detección en Tiempo Real", use_column_width=True)

            # update detection information
            st.session_state.last_detection = detection_info

            # show information
            display_detection_info(detection_info)

        cap.release()

        # auto-refresh for real-time preview
        time.sleep(0.1)
        st.rerun()

    except Exception as e:
        st.error(f"Error en detección: {e}")

# process frame
def process_frame_for_detection(frame):

    try: 
        embedding = st.session_state.system.extract_embeddings(frame)
        detection_info = {
            'persona': 'Desconocido',
            'confianza': 0,
            'emocion': 'neutral',
            'emocion_confianza': 0,
            'timestamp': datetime.now()
        }
    
        if embedding is not None:
            # identify person
            person_id , confidence = st.session_state.system.recognize_face(embedding)

            if person_id:

                cursor = st.session_state.system.conn.cursor()
                cursor.execute(
                    "SELECT nombre, apellido FROM Persona WHERE id = ?", (person_id,)
                )
                persona_data = cursor.fetchone()

                if persona_data: 
                    detection_info['persona'] = f"{persona_data[0]} {persona_data[1]}"
                    detection_info['confianza'] = confidence

                    # detect emotion - CORREGIDO: usar analyze_emotions en lugar de detect_emotion
                    emotion, emocion_conf = st.session_state.system.analyze_emotions(frame)

                    detection_info['emocion'] = emotion
                    detection_info['emocion_confianza'] = emocion_conf

                    # save in sql
                    save_detection_record(person_id, emotion, emocion_conf)
            else:
                # person not recognized but analyze emotion
                emotion, emocion_conf = st.session_state.system.analyze_emotions(frame)
                detection_info['emocion'] = emotion
                detection_info['emocion_confianza'] = emocion_conf
    
        return frame, detection_info

    except Exception as e:
        st.error(f"Error procesando frame: {e}")
        return frame, {
            'persona': 'Error',
            'confianza': 0,
            'emocion': 'neutral',
            'emocion_confianza': 0,
            'timestamp': datetime.now()
        }

# display detection info
def display_detection_info(info):
    """Mostrar información de detección"""
    st.info(f"**Persona:** {info['persona']}")
    if info['confianza'] > 0:
        st.info(f"**Confianza reconocimiento:** {info['confianza']:.2%}")
    st.info(f"**Emoción:** {info['emocion']}")
    st.info(f"**Confianza emoción:** {info['emocion_confianza']:.2%}")
    st.info(f"**Hora:** {info['timestamp'].strftime('%H:%M:%S')}")

# save record
def save_detection_record(person_id, emotion, confidence):

    try: 
        cursor = st.session_state.system.conn.cursor()
        cursor.execute(
            "INSERT INTO Deteccion (id_persona, emocion, confianza) VALUES (?, ?, ?)",
            (person_id, emotion, confidence)
        )
        st.session_state.system.conn.commit()
    except Exception as e:
        st.error(f"Error al guardar deteccion: {e}")

# display real time info

def display_realtime_info():
    """Mostrar información en tiempo real"""
    if st.session_state.last_detection:
        info = st.session_state.last_detection
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Persona Detectada", info['persona'])
            st.metric("Confianza", f"{info['confianza']:.2%}")
        
        with col2:
            st.metric("Emoción", info['emocion'])
            st.metric("Confianza Emoción", f"{info['emocion_confianza']:.2%}")
    else:
        st.info("Esperando detección...")

# display recent detections
def display_recent_detections():
    
    try: 
        cursor = st.session_state.system.conn.cursor()
        cursor.execute("""
            SELECT COALESCE(P.nombre, 'Desconocido') as nombre,
                   COALESCE(P.apellido, '') as apellido,
                   D.emocion, D.confianza, D.timestamp
            FROM Deteccion D
            LEFT JOIN Persona P ON D.id_persona = P.id
            ORDER BY D.timestamp DESC
            LIMIT 5
        """)

        detections = cursor.fetchall()

        if detections:
            for det in detections:
                nombre = det[0] or "Desconocido"
                apellido = det[1] or ""
                emocion = det[2] or "Desconocida"
                confianza = det[3]
                # normalize confidence: allow both 0-1 and 0-100 ranges
                conf_percent = None
                try:
                    conf_val = float(confianza)
                    if conf_val <= 1.0:
                        conf_percent = conf_val * 100
                    else:
                        # assume value already in percentage-scale (0-100)
                        conf_percent = conf_val
                except Exception:
                    conf_percent = None

                timestamp = det[4]

                with st.container():
                    st.write(f"**{nombre} {apellido}**")
                    if conf_percent is not None:
                        st.write(f"Emoción: {emocion} ({conf_percent:.1f}%)")
                    else:
                        st.write(f"Emoción: {emocion} (confianza: {confianza})")
                    st.write(f"Hora: {timestamp}")
                    st.divider()

        else:
            st.write("No hay detecciones registradas")
    except Exception as e:
        st.error(f"Error al mostrar detecciones: {e}")
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        
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
            SELECT P.nombre || ' ' || P.apellido as persona,
                   D.emocion,
                   COUNT(*) as conteo
            FROM Deteccion D
            JOIN Persona P ON D.id_persona = P.id
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

        # recognition rate (estimated)
        cursor.execute("SELECT COUNT(*) FROM Deteccion WHERE confianza > 0.7")
        reconocimientos_confiables = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Deteccion")
        total_detecciones = cursor.fetchone()[0]
        
        tasa_reconocimiento = (reconocimientos_confiables / total_detecciones * 100) if total_detecciones > 0 else 0

        # metrics

        with col1:
            st.metric("Total de personas registradas", total_personas)
        with col2:
            st.metric("Total de detecciones realizadas", detecciones_hoy)
        with col3:
            st.metric("Emocion predominante", emocion_predominante)
        with col4:
            st.metric("Porcentaje de emociones", f"{tasa_reconocimiento:.1f}%")

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
        col1, col2, col3 =st.columns(3)

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
        
        # build querys

        query = """
            SELECT D.timestamp, P.nombre, P.apellido, D.emocion, D.confianza
            FROM Deteccion D
            JOIN Persona P ON D.id_persona = P.id
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
            df = pd.DataFrame(detections, columns=['Fecha/Hora', 'Nombre', 'Apellido', 'Emoción', 'Confianza'])

            # normalize confidence values (allow both 0-1 and 0-100 scales)
            def fmt_conf(x):
                try:
                    v = float(x)
                    if v <= 1.0:
                        return f"{v*100:.1f}%"
                    else:
                        return f"{v:.1f}%"
                except Exception:
                    return str(x)

            df['Confianza'] = df['Confianza'].apply(fmt_conf)
            st.dataframe(df, use_container_width=True)

        # save button CSV file

        if detections:
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

if __name__ == "__main__":
    main()