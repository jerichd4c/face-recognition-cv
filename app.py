# libraries

import sqlite3
import streamlit as st
import os
import pickle
import numpy as np
import time

class FacialRecognitionSystem():
    def __init__(self):
        self.init_database()
            
    def init_database(self):
        #initialize database
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

# initial page config
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    layout="wide",
    page_icon="📷"
)

# initialize system (st web page)
if 'system' not in st.session_state:
    st.session_state.system = FacialRecognitionSystem()

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
        #show_real_time_detection_page()
    elif page == "Reportes y estadisticas":
        show_registration_page()
        #show_reports_page()

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
                   if register_person(nombre, apellido, email):
                       st.success("Persona registrada exitosamente")
                   else:
                       st.error("Error al registrar persona, asegurase de validar todos los campos")

    with col2: 
        st.subheader("Rostro")

        # camera selector
        camera_index = st.selectbox("Seleccione la camara", [0, 1, 2])

        if st.button("Iniciar captura de rostro"):
            capture_face(camera_index) # TO DO

    # DEBUG IN ST 
    email = st.text_input("Escribe el email de la persona a buscar")
    if st.button("Buscar persona"):
        show_registered_person(email)

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
    
# DEBUG FUNCTION: show all registered persons

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
            start_detection = st.button("Iniciar Deteccion")
        with col_controls[1]:
            stop_detection = st.button("Detener Deteccion")
        with col_controls[2]:
            camera_select = st.selectbox("Camara", [0, 1, 2], key="detection_cam")

    #placeholder for video

        video_placeholder= st.empty()

        if start_detection:

            #TO DO: Implement logic

            st.infor("Deteccion iniciada")

            for i in range(10):
                with video_placeholder.container():
                    #create AI image using numpy
                    img = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

                    # simulate detection
                    st.image(img, caption="Vista de cámara en tiempo real", use_column_width=True)

                    # show detection (IMPLEMENT LOGIC)
                    st.info(f"Persona detectada: Carlos (Confianza: 100%)")

                time.sleep(2)

        if stop_detection:
            st.warning("Deteccion detenida")
            video_placeholder.empty()

     with col2:
        st.subheader("Informacion de Deteccion")

        # placeholder
        st.info("Esperando deteccion...")

        # last detection
        st.subheader("Ultima detecciones hechas")
        display_recent_detections()

if __name__ == "__main__":
    main()