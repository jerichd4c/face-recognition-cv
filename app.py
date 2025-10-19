# libraries

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
                enforce_detection = False
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
            enforce_detection = False
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

                if similarity > best_score:
                    best_score = similarity
                    best_match = person_id

            return best_match, best_score

        except Exception as e:
            st.error(f"Error al reconocer rostro: {e}")
            return None, 0 

# initialize system (st web page)
if 'system' not in st.session_state:
    st.session_state.system = FacialRecognitionSystem()
    st.session_state.capture_active = False
    st.session_state.detection_active = False

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
        st.subheader("Datos personales")

        with st.form("registrarion_form"):
            nombre = st.text_input("Nombre")
            apellido = st.text_input("Apellido")
            email = st.text_input("Email")

            submitted = st.form_submit_button("Registrar persona")

            if submitted:
                if nombre and apellido and email:
                   person_id = register_person(nombre, apellido, email)
                   if person_id:
                       st.session_state.current_person_id = person_id
                       st.success("Persona registrada exitosamente")
                   else:
                       st.error("Error al registrar persona, asegurase de validar todos los campos")

    with col2: 
        st.subheader("Captura facial")

        if 'current_person_id' not in st.session_state:
            st.warning("Primero registre una persona")
            return

        # camera selector
        camera_index = st.selectbox("Seleccione la camara", [0, 1, 2])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Iniciar captura de rostro") and not st.session_state.capture_active:
                st.session_state.capture_active = True
                start_face_capture(camera_index)
            
        with col2:
            if st.button("Finalizar captura de rostro") and st.session_state.capture_active:
                st.session_state.capture_active = False
                st.success("Captura de rostro finalizada")

    # search person
    email = st.text_input("Escribe el email de la persona a buscar")
    if st.button("Buscar persona"):
        show_registered_person(email)

# show registered person
def show_registered_person(email):
    cursor = st.session_state['system'].conn.cursor()
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
        cursor= st.session_state['system'].conn.cursor()
        
        # verify email with id
        cursor.execute("SELECT id from Persona WHERE email = ?", (email,))
        if cursor.fetchone():
            st.error("Email ya registrado")
            return False
        
        # insert new person 
        cursor.execute("INSERT INTO Persona (nombre, apellido, email) VALUES (?, ?, ?)", (nombre, apellido, email) )
        st.session_state['system'].conn.commit()
        return True
    
    except Exception as e:
        st.error(f"Error al registrar persona: {e}")
        return False
    
# capture faces
def start_face_capture(camera_index):
    st.info("Iniciando captural facial...")

    try:

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error("Error al abrir la camara")
            return
        
        st.info("Capturando rostro, mire a la camara")
        placeholder = st.empty()
        embeddings_captured = 0
        max_embeddings = 5

        while st.session_state.capture_active and embeddings_captured < max_embeddings:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
            
            # display frame
            placeholder.image(rgb_frame, caption="Vista de camara - captura de rostro", use_column_width=True)

            # extract embeddings
            embedding = st.session_state['system'].extract_embeddings(rgb_frame)

            if embedding is not None:
                
                # save embedding
                cursor = st.session_state['system'].conn.cursor()
                embedding_blob = pickle.dumps(embedding)
                cursor.execute("INSERT INTO Rostro (id_persona, rostro) VALUES (?, ?)", (st.session_state.current_person_id, embedding_blob))
                st.session_state['system'].conn.commit()

                st.session_state.system.persona_embeddings[st.session_state.current_person_id] = embedding

                st.session_state['system'].save_face_embeddings()

                embeddings_captured += 1
                st.success(f"Embedding {embeddings_captured}/{max_embeddings} capturado exitosamente")

                # short pause betweens caps

                time.sleep(1)
            
            time.sleep(0.1)

        cap.release()
        placeholder.empty()

        if embeddings_captured >= max_embeddings:
            st.success("Captura de rostros finalizada")
            st.session_state_capture_active = False
    except Exception as e:
        st.error(f"Error al capturar rostros: {e}")


# detection page
def show_detection_page():
     st.title("Deteccion en tiempo real")

     col1, col2 = st.columns([2,1])

     with col1:
        st.subheader("Vista de camara")

        # detection controls
        col_controls = st.columns(3)
        # detection controls
        with col_controls[0]:
            camera_select = st.selectbox("Camara", [0, 1, 2], key="detection_cam")
        with col_controls[1]:
            if st.button("Iniciar Detección") and not st.session_state.detection_active:
                st.session_state.detection_active = True
                start_real_time_detection(camera_select)
        with col_controls[2]:
            if st.button("Detener Detección") and st.session_state.detection_active:
                st.session_state.detection_active = False  
                st.success("Detección detenida") 

    #placeholder for video

        video_placeholder= st.empty()
        info_placeholder = st.empty()

        if st.session_state.detection_active:
            st.session_state.video_placeholder = video_placeholder
            st.session_state.info_placeholder = info_placeholder

     with col2:
        st.subheader("Informacion de Deteccion")
        display_realtime_info()

        # last detection
        st.subheader("Ultima detecciones hechas")
        display_recent_detections()

# start real-time detection
def start_real_time_detection(camera_index):
    
    #start on a separate thread
    def detection_thread():
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Error al abrir la camara")
                return
            
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                processed_frame, detection_info = process_frame_for_detection(rgb_frame)

                # display frame

                if hasattr(st.session_state, 'video_placeholder'):
                    st.session_state.video_placeholder.image(
                        processed_frame,
                        caption="Deteccion en tiempo real",
                        use_column_width=True
                    )

                # update info

                if hasattr(st.session_state, 'info_placeholder'):
                    with st.session_state.info_placeholder.container():
                        display_detection_info(detection_info)
                
                time.sleep(0.1)
            
            cap.release()

        except Exception as e:
            st.error(f"Error en deteccion en tiempo real: {e}")

# process frame
def process_frame_for_detection(frame):

    try: 
        embedding = st.session_state['system'].extract_embeddings(frame)
        detection_info = {
            'persona': 'Desconocido',
            'confianza': 0,
            'emocion': 'neutral',
            'emocion_confianza': 0 
        }
    
        if embedding is not None:
            # identify person
            person_id , confidence = st.session_state['system'].recognize_face(embedding)

            if person_id:

                cursor = st.session_state.system.conn.cursor()
                cursor.execute(
                    "SELECT nombre, apellido FROM Persona WHERE id = ?", (person_id,)
                )
                persona_data = cursor.fetchone()

                if persona_data: 
                    detection_info['persona'] = f"{persona_data[0]} {persona_data[1]}"
                    detection_info['confianza'] = confidence

                    # detect emotion
                    emotion, emocion_conf = st.session_state['system'].detect_emotion(frame)

                    detection_info['emocion'] = emotion
                    detection_info['emocion_confianza'] = emocion_conf

                    # save in sql
                    save_detection_record(person_id, emotion, emocion_conf)
    
        return frame, detection_info

    except Exception as e:
        st.error(f"Error procesando frame: {e}")
        return frame, {
            'persona': 'Error',
            'confianza': 0,
            'emocion': 'neutral',
            'emocion_confianza': 0
        }

# display detection info
def display_detection_info(info):
    """Mostrar información de detección"""
    st.info(f"**Persona:** {info['persona']}")
    if info['confianza'] > 0:
        st.info(f"**Confianza:** {info['confianza']:.2%}")
    st.info(f"**Emoción:** {info['emocion']}")
    st.info(f"**Confianza emoción:** {info['emocion_confianza']:.2%}")

# save record
def save_detection_record(person_id, emotion, confidence):

    try: 
        cursor= st.session_state['system'].conn.cursor()
        cursor.execute(
            "INSERT INTO Deteccion (id_persona, emocion, confianza) VALUES (?, ?, ?)",
            (person_id, emotion, confidence)
        )
        st.session_state['system'].conn.commit()
    except Exception as e:
        st.error(f"Error al guardar deteccion: {e}")

# display real time info

def display_realtime_info():
    """Mostrar información en tiempo real"""
    if hasattr(st.session_state, 'last_detection'):
        info = st.session_state.last_detection
        st.metric("Persona Detectada", info['persona'])
        st.metric("Confianza", f"{info['confianza']:.2%}")
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

    # mock data

    emotions_data = {
        'Persona': ['Juan Pérez', 'María García', 'Carlos López', 'Ana Martínez'],
        'Feliz': [45, 30, 25, 40],
        'Triste': [15, 25, 30, 20],
        'Enojado': [10, 15, 20, 5],
        'Sorprendido': [20, 20, 15, 25],
        'Neutral': [10, 10, 10, 10]
    }

    df= pd.DataFrame(emotions_data)

    fig = px.bar(df, x='Persona', y=['Feliz', 'Triste', 'Enojado', 'Sorprendido', 'Neutral'],
                 title="Distribucion de emociones por persona",
                 labels={"value": "Porcentaje", "variable": "Emocion" })

    st.plotly_chart(fig, use_container_width=True)

# general stats
def show_general_stats():
    st.subheader("Estadisticas generales")

    col1 , col2, col3, col4 = st.columns(4)

    # example 

    with col1:
        st.metric("Total de personas registradas", "25")
    with col2:
        st.metric("Total de detecciones realizadas", "156")
    with col3:
        st.metric("Emocion predominante", "Feliz")
    with col4:
        st.metric("Porcentaje de emociones", "94.2%")
    
    # detection graph per hour

    st.subheader("Detecciones por hora")
    hours = list(range(24))
    detections = [5, 3, 2, 1, 1, 2, 8, 15, 20, 18, 16, 14, 16, 15, 12, 10, 8, 12, 15, 14, 10, 8, 6, 4]

    fig = px.line(x=hours, y=detections, title='Patrón de Detecciones por Hora',
                  labels={'x': 'Hora del día', 'y': 'Número de detecciones'})
    st.plotly_chart(fig, use_container_width=True)

# history
def show_detection_history():
    st.subheader("Historial de detecciones")
    
    # filters
    col1, col2, col3 =st.columns(3)

    with col1:
        date_filter = st.date_input("Filtrar por fecha")
    with col2:
        person_filter = st.selectbox("Filtrar por persona", ["Todas", "Juan Pérez", "María García", "Carlos López"])
    with col3:
        emotion_filter = st.selectbox("Filtrar por emoción", ["Todas", "Feliz", "Triste", "Enojado", "Sorprendido"])
    
    # save button CSV file

    if st.button("Guardar reporte en CSV"):
        st.success("Reporte guardado exitosamente (TO DO)")

    # detection table (mock data)

    detection_data = {
        'Fecha/Hora': ['2024-01-15 08:30:15', '2024-01-15 09:15:22', '2024-01-15 10:05:47'],
        'Persona': ['Juan Pérez', 'María García', 'Carlos López'],
        'Emoción': ['Feliz', 'Neutral', 'Sorprendido'],
        'Confianza': ['95.2%', '88.7%', '92.1%']
    }
    
    df = pd.DataFrame(detection_data)
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()

# TO DO LIST 

# - implement openCV for face detection (ex: captureFaces, complete show_detection_page)
# - implement facial embedding extraction
# - connect detection module with deepFace
# - delete mock data and replace with real data (datasets, etc) and use it on reports page
# - do logic for save as CSV