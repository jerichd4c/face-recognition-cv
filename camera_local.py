import argparse
import os
import time
import pickle
import sqlite3
from datetime import datetime
from threading import Thread, Event, Lock

import cv2
import numpy as np
from cv2 import data as cv2_data
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity


DB_PATH = 'facial_recognition.db'


def init_database(conn):
    cur = conn.cursor()
    # Personas
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Persona (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            apellido TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            fecha_registro_timestamp DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Rostros (embeddings)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Rostro (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_persona INTEGER NOT NULL,
            rostro BLOB NOT NULL,
            FOREIGN KEY (id_persona) REFERENCES Persona (id)
        )
        """
    )
    # Detecciones (esquema v2)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Deteccion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_persona INTEGER,
            recog_confianza REAL,
            emocion TEXT NOT NULL,
            emocion_confianza REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_persona) REFERENCES Persona (id)
        )
        """
    )
    # Detalle de emociones por detección (distribución completa)
    cur.execute(
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
    conn.commit()


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.astype(np.float32)


def extract_embedding(image_rgb: np.ndarray) -> np.ndarray | None:
    try:
        rep = DeepFace.represent(
            img_path=image_rgb,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend='opencv'
        )
        if rep:
            emb = normalize(np.asarray(rep[0]["embedding"], dtype=np.float32))
            return emb
        return None
    except Exception:
        return None


def analyze_emotion(image_rgb: np.ndarray, backend: str = 'skip') -> tuple[str, float]:
    try:
        # DeepFace expects images in BGR (OpenCV format). If we get RGB, convert.
        if image_rgb is not None and image_rgb.ndim == 3:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_rgb
        ana = DeepFace.analyze(
            img_path=image_bgr,
            actions=['emotion'],
            enforce_detection=False if backend == 'skip' else True,
            detector_backend=backend
        )
        if ana:
            emotion = ana[0]['dominant_emotion']
            conf = float(ana[0]['emotion'][emotion]) / 100.0
            return emotion, conf
        return 'neutral', 0.0
    except Exception:
        return 'neutral', 0.0


def analyze_emotion_full(image_rgb: np.ndarray, backend: str = 'skip') -> tuple[str, float, dict[str, float]]:
    """Return dominant emotion, its confidence, and full distribution (0..1 floats)."""
    try:
        if image_rgb is not None and image_rgb.ndim == 3:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_rgb
        ana = DeepFace.analyze(
            img_path=image_bgr,
            actions=['emotion'],
            enforce_detection=False if backend == 'skip' else True,
            detector_backend=backend
        )
        if ana:
            dist_percent = ana[0]['emotion']  # values 0..100
            # Convert to 0..1 floats
            dist = {k: float(v)/100.0 for k, v in dist_percent.items()}
            emotion = ana[0]['dominant_emotion']
            conf = float(dist.get(emotion, 0.0))
            return emotion, conf, dist
        return 'neutral', 0.0, {k: 0.0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}
    except Exception:
        return 'neutral', 0.0, {k: 0.0 for k in ['angry','disgust','fear','happy','sad','surprise','neutral']}


def load_person_embeddings(conn) -> dict[int, list[np.ndarray]]:
    cur = conn.cursor()
    cur.execute("SELECT id_persona, rostro FROM Rostro")
    embs: dict[int, list[np.ndarray]] = {}
    for pid, blob in cur.fetchall():
        try:
            emb = pickle.loads(blob)
            embs.setdefault(pid, []).append(normalize(emb))
        except Exception:
            continue
    return embs


def recognize(emb: np.ndarray, store: dict[int, list[np.ndarray]], thr: float) -> tuple[int | None, float]:
    best_id, best = None, 0.0
    for pid, lst in store.items():
        if not lst:
            continue
        try:
            sims = [cosine_similarity([emb], [e])[0][0] for e in lst]
        except Exception:
            sims = [float(np.dot(emb, e)) for e in lst]
        s = max(sims) if sims else 0.0
        if s >= thr and s > best:
            best, best_id = s, pid
    return best_id, float(best)


def register_mode(args):
    conn = sqlite3.connect(DB_PATH)
    init_database(conn)
    cur = conn.cursor()

    # Create or get person
    person_id = None
    if args.person_id is not None:
        cur.execute("SELECT id FROM Persona WHERE id=?", (args.person_id,))
        row = cur.fetchone()
        if not row:
            print("[ERROR] Persona no encontrada por id.")
            return
        person_id = args.person_id
    else:
        if not (args.nombre and args.apellido and args.email):
            print("[ERROR] Debes proveer nombre, apellido y email si no usas --person-id")
            return
        # Check duplicate email
        cur.execute("SELECT id FROM Persona WHERE email=?", (args.email,))
        if cur.fetchone():
            print("[ERROR] Email ya registrado. Usa --person-id existente o cambia el email.")
            return
        cur.execute("INSERT INTO Persona(nombre, apellido, email) VALUES(?,?,?)", (args.nombre, args.apellido, args.email))
        conn.commit()
        person_id = cur.lastrowid
        print(f"[OK] Persona creada con id {person_id}")

    # Load embeddings and cascade
    store = load_person_embeddings(conn)
    cascade = cv2.CascadeClassifier(os.path.join(cv2_data.haarcascades, 'haarcascade_frontalface_default.xml'))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    captured = 0
    print("[INFO] Registro: presiona 'c' para capturar, 'q' para salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturas: {captured}/5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, "[c] capturar  [q] salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.imshow('Registro - Camara', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and len(faces) > 0:
            # take largest face
            (x, y, w, h) = max(faces, key=lambda b: b[2]*b[3])
            face_rgb = rgb[y:y+h, x:x+w]
            # Downscale for speed
            face_small = cv2.resize(face_rgb, (0,0), fx=0.5, fy=0.5)
            emb = extract_embedding(face_small)
            if emb is None:
                print("[WARN] No se pudo extraer embedding. Intenta de nuevo.")
                continue
            # Duplicate check against other people
            best_other_id, best_score = None, 0.0
            for pid, lst in store.items():
                if pid == person_id:
                    continue
                try:
                    sims = [cosine_similarity([emb], [e])[0][0] for e in lst]
                except Exception:
                    sims = [float(np.dot(emb, e)) for e in lst]
                s = max(sims) if sims else 0.0
                if s > best_score:
                    best_score, best_other_id = s, pid
            if best_other_id is not None and best_score >= args.duplicate_threshold and not args.force:
                print(f"[WARN] Posible duplicado con id {best_other_id} (sim {best_score:.3f}). Usa --force para guardar de todas formas.")
                continue
            # Save embedding
            cur = conn.cursor()
            cur.execute("INSERT INTO Rostro(id_persona, rostro) VALUES(?,?)", (person_id, pickle.dumps(emb)))
            conn.commit()
            store.setdefault(person_id, []).append(emb)
            captured += 1
            print(f"[OK] Embedding guardado ({captured}/5)")
            if captured >= 5:
                print("[OK] Registro completado")
                break

    cap.release()
    cv2.destroyAllWindows()


def detection_mode(args):
    # Load DB and embeddings
    conn_main = sqlite3.connect(DB_PATH)
    init_database(conn_main)
    store = load_person_embeddings(conn_main)

    # Shared state between threads
    stop_event = Event()
    result_lock = Lock()
    result = {
        'label': 'Desconocido',
        'recog_conf': 0.0,
        'emotion': 'neutral',
        'emo_conf': 0.0,
        'box': None
    }

    # Low-latency camera open (prefer DirectShow on Windows)
    try:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return
    # Set camera resolution to reduce processing load
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.frame_height))
    except Exception:
        pass
    # Try to minimize internal buffering
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    latest_frame_lock = Lock()
    latest_frame = {'frame': None}

    # Warm-up models to avoid first-time stutters
    try:
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        _ = extract_embedding(dummy)
        if not args.no_emotion:
            _ = analyze_emotion(dummy, backend=args.emotion_backend)
    except Exception:
        pass

    def grabber_loop():
        # Continuously grab frames as fast as possible to keep them fresh
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with latest_frame_lock:
                latest_frame['frame'] = frame

    def inference_loop():
        # Do heavy work on a fixed interval without blocking display
        cascade = cv2.CascadeClassifier(os.path.join(cv2_data.haarcascades, 'haarcascade_frontalface_default.xml'))
        # Separate DB connection for this thread
        conn_thr = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur_thr = conn_thr.cursor()
        last_ts = 0.0
        last_emo_ts = 0.0
    # To reduce flicker, only update label/emotion when we have new values
        while not stop_event.is_set():
            now = time.time()
            if (now - last_ts) < (args.infer_interval_ms / 1000.0):
                time.sleep(0.005)
                continue
            last_ts = now
            # Get latest frame snapshot
            with latest_frame_lock:
                frame = latest_frame['frame']
            if frame is None:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downscaled detection for speed
                if 0.2 <= args.detect_scale <= 1.0:
                    ds = float(args.detect_scale)
                else:
                    ds = 0.5
                small_gray = cv2.resize(gray, (0, 0), fx=ds, fy=ds)
                faces_small = cascade.detectMultiScale(small_gray, 1.1, 5, minSize=(max(30, int(60*ds)), max(30, int(60*ds))))
                # Defaults are None to preserve previous values if nothing updates
                label = None
                recog_conf = None
                emotion = None
                emo_conf = None
                box = None
                pid = None
                if len(faces_small) > 0:
                    (xs, ys, ws, hs) = max(faces_small, key=lambda b: b[2]*b[3])
                    # Scale back to original frame coords
                    x = int(xs / ds)
                    y = int(ys / ds)
                    w = int(ws / ds)
                    h = int(hs / ds)
                    # Apply padding around face box for better emotion context
                    pad = float(args.crop_padding)
                    if pad > 0:
                        px = int(w * pad)
                        py = int(h * pad)
                        x0 = max(0, x - px)
                        y0 = max(0, y - py)
                        x1 = min(rgb.shape[1], x + w + px)
                        y1 = min(rgb.shape[0], y + h + py)
                    else:
                        x0, y0, x1, y1 = x, y, x + w, y + h
                    box = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
                    face_rgb = rgb[y0:y1, x0:x1]
                    small = cv2.resize(face_rgb, (0, 0), fx=args.scale, fy=args.scale)
                    emb = extract_embedding(small)
                    if emb is not None:
                        pid, rconf = recognize(emb, store, args.threshold)
                        recog_conf = float(rconf)
                        if pid is not None:
                            cur_thr.execute("SELECT nombre, apellido FROM Persona WHERE id=?", (pid,))
                            row = cur_thr.fetchone()
                            if row:
                                label = f"{row[0]} {row[1]}"
                    # Emotion analysis on its own cadence and independently of recognition
                    if not args.no_emotion and (now - last_emo_ts) >= (args.emotion_interval_ms / 1000.0):
                        # Optionally use a larger face for emotion accuracy
                        emo_face = face_rgb
                        if args.emotion_scale and args.emotion_scale != 1.0:
                            emo_face = cv2.resize(face_rgb, (0, 0), fx=args.emotion_scale, fy=args.emotion_scale)
                        e_label, e_conf, e_dist = analyze_emotion_full(emo_face, backend=args.emotion_backend)
                        emotion, emo_conf = e_label, float(e_conf)
                        last_emo_ts = now
                    # Save detection whenever we computed any of the two signals
                    if (emotion is not None) or (recog_conf is not None):
                        cur_thr.execute(
                            "INSERT INTO Deteccion(id_persona, recog_confianza, emocion, emocion_confianza, timestamp) VALUES(?,?,?,?,?)",
                            (pid, recog_conf if pid is not None else None, emotion if emotion is not None else 'neutral', emo_conf if emo_conf is not None else 0.0, datetime.now())
                        )
                        det_id = cur_thr.lastrowid
                        # Persist full distribution if available
                        try:
                            if 'e_dist' in locals():
                                for emo, confv in e_dist.items():
                                    cur_thr.execute(
                                        "INSERT INTO DeteccionEmocionDetalle(id_deteccion, emocion, confianza) VALUES(?,?,?)",
                                        (det_id, emo, float(confv))
                                    )
                        except Exception:
                            pass
                        conn_thr.commit()
                # Publish results
                with result_lock:
                    if label is not None:
                        result['label'] = label
                    if recog_conf is not None:
                        result['recog_conf'] = float(recog_conf)
                    if emotion is not None:
                        result['emotion'] = emotion
                    if emo_conf is not None:
                        result['emo_conf'] = float(emo_conf)
                    result['box'] = box
            except Exception:
                # Avoid crashing the loop on transient errors
                continue
        try:
            cur_thr.close()
            conn_thr.close()
        except Exception:
            pass

    # Start threads
    t_grab = Thread(target=grabber_loop, daemon=True)
    t_inf = Thread(target=inference_loop, daemon=True)
    t_grab.start()
    t_inf.start()

    print("[INFO] Detección: presiona 'q' para salir")
    # Display loop is lightweight: overlay last results and show
    fps_ts = time.time()
    fps = 0.0
    while True:
        with latest_frame_lock:
            frame = latest_frame['frame']
        if frame is None:
            # Wait a bit for first frame
            if stop_event.is_set():
                break
            time.sleep(0.005)
            continue
        # Draw last detection box/text without blocking
        with result_lock:
            r = dict(result)
        if r['box'] is not None:
            x, y, w, h = r['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Persona: {r['label']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Conf. recog: {r['recog_conf']*100:.1f}%  Emocion: {r['emotion']} ({r['emo_conf']*100:.1f}%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        # FPS overlay
        now = time.time()
        dt = now - fps_ts
        if dt > 0:
            # simple moving estimate
            fps = 0.9*fps + 0.1*(1.0/dt) if fps > 0 else (1.0/dt)
        fps_ts = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,0), 2)
        cv2.imshow('Deteccion - Camara', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # Stop threads and release resources
    stop_event.set()
    t_grab.join(timeout=1.0)
    t_inf.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Camara local para registro y deteccion (alta fluidez)')
    sub = parser.add_subparsers(dest='mode', required=True)

    pr = sub.add_parser('register', help='Registrar persona capturando embeddings')
    pr.add_argument('--camera', type=int, default=0, help='Indice de camara (default 0)')
    pr.add_argument('--person-id', type=int, help='ID de persona existente')
    pr.add_argument('--nombre', type=str, help='Nombre (si no hay person-id)')
    pr.add_argument('--apellido', type=str, help='Apellido (si no hay person-id)')
    pr.add_argument('--email', type=str, help='Email (si no hay person-id)')
    pr.add_argument('--duplicate-threshold', type=float, default=0.8, help='Umbral de duplicado (similitud coseno)')
    pr.add_argument('--force', action='store_true', help='Forzar captura si hay duplicado')

    pd = sub.add_parser('detect', help='Deteccion en tiempo real con overlay y guardado en DB')
    pd.add_argument('--camera', type=int, default=0, help='Indice de camara (default 0)')
    pd.add_argument('--threshold', type=float, default=0.6, help='Umbral de reconocimiento (similitud coseno)')
    pd.add_argument('--infer-interval-ms', type=float, default=500.0, help='Intervalo entre inferencias pesadas (ms)')
    pd.add_argument('--emotion-interval-ms', type=float, default=1500.0, help='Intervalo minimo entre analisis de emocion (ms)')
    pd.add_argument('--scale', type=float, default=0.5, help='Factor de escala para inferencia (0.3-1.0)')
    pd.add_argument('--detect-scale', type=float, default=0.5, help='Factor de escala para la deteccion de rostro (0.2-1.0)')
    pd.add_argument('--frame-width', type=int, default=640, help='Ancho de frame de la camara (ej. 640)')
    pd.add_argument('--frame-height', type=int, default=480, help='Alto de frame de la camara (ej. 480)')
    pd.add_argument('--no-emotion', action='store_true', help='Deshabilitar analisis de emociones para maximizar FPS')
    pd.add_argument('--emotion-scale', type=float, default=1.2, help='Factor de escala para el recorte usado en emociones (1.0-1.5)')
    pd.add_argument('--emotion-backend', type=str, default='opencv', choices=['opencv','retinaface','mediapipe','skip'], help='Backend para deteccion de rostro en emociones')
    pd.add_argument('--crop-padding', type=float, default=0.15, help='Padding adicional alrededor del rostro para emociones (0-0.3)')

    args = parser.parse_args()
    if args.mode == 'register':
        register_mode(args)
    elif args.mode == 'detect':
        detection_mode(args)


if __name__ == '__main__':
    main()
