# libraries

import sqlite3

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
                emailL TEXT UNIQUE NOT NULL,
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

if __name__ == "__main__":
    FacialRecognitionSystem()