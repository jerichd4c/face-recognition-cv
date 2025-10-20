# libraries

from datetime import datetime
import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px

# initial page config
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    layout="wide",
    page_icon="📷"
)

class FacialRecognitionSystem():
    def __init__(self):
        self.init_database()
            
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

def main():
    st.sidebar.title("Panel de Reportes")
    st.sidebar.info("Esta aplicación ahora solo muestra reportes y estadísticas. El registro y la detección en tiempo real se realizan con la cámara local (script).")
    with st.sidebar.expander("Cómo usar la cámara local", expanded=False):
        st.write("Ejecuta estos comandos en PowerShell:")
        st.code("python camera_local.py register --camera 0 --nombre \"Ana\" --apellido \"Gomez\" --email \"ana@example.com\"", language="powershell")
        st.code("python camera_local.py detect --camera 0 --threshold 0.65 --infer-interval-ms 500 --scale 0.5", language="powershell")

    # Solo reportes
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

if __name__ == "__main__":
    main()