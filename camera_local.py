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
_WARNED_DNN_MISSING = False
EMO_KEYS = ['angry','disgust','fear','happy','sad','surprise','neutral']

def init_database(conn):
    cur = conn.cursor()
    # People table
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
    # Faces (embeddings)
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
    # Detections (v2 schema)
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
    # Emotion details per detection (full distribution)
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


def extract_embedding(image_rgb: np.ndarray, model_name: str = "Facenet") -> np.ndarray | None:
    try:
        rep = DeepFace.represent(
            img_path=image_rgb,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='skip'
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
    # ROI is already cropped; skip re-detection for better stability
        ana = DeepFace.analyze(
            img_path=image_bgr,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='skip'
        )
        if ana:
            dist_percent = ana[0]['emotion']  # values 0..100
            # Normalize keys: lowercase + synonyms
            synonym_map = {
                'anger': 'angry',
                'happiness': 'happy',
                'sadness': 'sad',
                'surprised': 'surprise',
                'fearful': 'fear',
                # Extra variants commonly seen
                'disgusted': 'disgust',
                'contempt': 'disgust',
            }
            norm = {}
            for k, v in dist_percent.items():
                kk = str(k).lower()
                kk = synonym_map.get(kk, kk)
                norm[kk] = float(v) / 100.0
            # Build complete dist with 7 keys
            # Add tiny epsilon to avoid exact zeros causing ties/instability
            eps = 1e-6
            dist = {k: float(norm.get(k, 0.0)) + eps for k in EMO_KEYS}
            # Re-normalize to sum ~1
            s = sum(dist.values()) or 1.0
            dist = {k: v/s for k, v in dist.items()}
            dom = str(ana[0].get('dominant_emotion', 'neutral')).lower()
            dom = synonym_map.get(dom, dom)
            conf = float(dist.get(dom, 0.0))
            return dom, conf, dist
        return 'neutral', 0.0, {k: 0.0 for k in EMO_KEYS}
    except Exception:
        return 'neutral', 0.0, {k: 0.0 for k in EMO_KEYS}


def preprocess_emotion_roi(face_bgr: np.ndarray, upscale: float = 1.2, use_clahe: bool = True) -> np.ndarray:
    """Upscale and apply CLAHE on L channel to enhance subtle expressions. Return RGB array."""
    try:
        bgr = face_bgr
        if upscale and upscale != 1.0:
            bgr = cv2.resize(bgr, (0, 0), fx=float(upscale), fy=float(upscale))
        if use_clahe:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    except Exception:
        try:
            return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return face_bgr


def load_opencv_dnn_face_detector(models_dir: str = "models"):
    proto = os.path.join(models_dir, 'deploy.prototxt')
    caffemodel = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    if os.path.isfile(proto) and os.path.isfile(caffemodel):
        try:
            net = cv2.dnn.readNetFromCaffe(proto, caffemodel)
            return net
        except Exception:
            return None
    return None


def detect_largest_face(frame_bgr: np.ndarray, gray: np.ndarray, backend: str, cascade, dnn_net, detect_scale: float) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (x0,y0,x1,y1) for the largest face or (None, None, None, None) if none."""
    h, w = frame_bgr.shape[:2]
    try:
        if backend == 'opencv-dnn' and dnn_net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_net.setInput(blob)
            detections = dnn_net.forward()
            best = (-1.0, None)
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < 0.5:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x0, y0, x1, y1) = box.astype("int")
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                area = max(0, x1 - x0) * max(0, y1 - y0)
                if area > best[0]:
                    best = (area, (x0, y0, x1, y1))
            return best[1] if best[1] is not None else (None, None, None, None)
        elif backend in ('retinaface', 'mediapipe'):
            # Use DeepFace to extract faces with chosen backend
            try:
                faces = DeepFace.extract_faces(img_path=frame_bgr, detector_backend=backend, enforce_detection=False)
                best = (-1.0, None)
                for f in faces or []:
                    fa = f.get('facial_area') or {}
                    x = int(fa.get('x', 0)); y = int(fa.get('y', 0))
                    w0 = int(fa.get('w', 0)); h0 = int(fa.get('h', 0))
                    x0, y0, x1, y1 = x, y, x + w0, y + h0
                    area = w0 * h0
                    if area > best[0]:
                        best = (area, (max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
                return best[1] if best[1] is not None else (None, None, None, None)
            except Exception:
                pass  # fall back to opencv
    # If opencv-dnn is requested but not available, note it
        if backend == 'opencv-dnn' and dnn_net is None:
            # Only warn once
            global _WARNED_DNN_MISSING
            if not _WARNED_DNN_MISSING:
                print("[WARN] opencv-dnn backend seleccionado pero modelo Caffe no encontrado en ./models; usando Haar cascade.")
                _WARNED_DNN_MISSING = True
    # Default: OpenCV Haar cascade
        ds = float(detect_scale) if 0.2 <= detect_scale <= 1.0 else 0.5
        small = cv2.resize(gray, (0, 0), fx=ds, fy=ds)
    # More permissive to facilitate initial detection
        faces = cascade.detectMultiScale(small, 1.05, 3, minSize=(max(24, int(40*ds)), max(24, int(40*ds))))
    # If no faces at scaled size, retry at full size to improve robustness
        if len(faces) == 0 and ds != 1.0:
            faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
            if len(faces) == 0:
                return (None, None, None, None)
            xs, ys, ws, hs = max(faces, key=lambda b: b[2] * b[3])
            x = int(xs); y = int(ys); w0 = int(ws); h0 = int(hs)
            return (max(0, x), max(0, y), min(w, x + w0), min(h, y + h0))
        xs, ys, ws, hs = max(faces, key=lambda b: b[2] * b[3])
        x = int(xs / ds); y = int(ys / ds); w0 = int(ws / ds); h0 = int(hs / ds)
        return (max(0, x), max(0, y), min(w, x + w0), min(h, y + h0))
    except Exception:
        return (None, None, None, None)


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
    """Return (matched_person_id_or_None, best_similarity).
    - best_similarity is the max cosine similarity across all stored embeddings (regardless of threshold)
    - person_id is only returned if best_similarity >= thr, else None
    """
    best_pid = None
    best_sim = 0.0
    for pid, lst in store.items():
        if not lst:
            continue
        try:
            sims = [cosine_similarity([emb], [e])[0][0] for e in lst]
        except Exception:
            sims = [float(np.dot(emb, e)) for e in lst]
        s = max(sims) if sims else 0.0
        if s > best_sim:
            best_sim = s
            best_pid = pid
    matched = best_pid if best_sim >= thr else None
    return matched, float(best_sim)


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
        'box': None,
        'emo_top3': ''
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
        if args.target_fps and args.target_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(args.target_fps))
        if args.force_mjpg:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
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
        _ = extract_embedding(dummy, model_name=args.embed_model)
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
        dnn_net = load_opencv_dnn_face_detector()
        # Separate DB connection for this thread
        conn_thr = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur_thr = conn_thr.cursor()
        last_ts = 0.0
        last_emo_ts = 0.0
        last_log_ts = 0.0
        last_logged_emotion = None
        # Box smoothing state and emotion smoothing
        sm_box = None  # [x0,y0,x1,y1]
        miss_count = 0
        emo_hist = []
        # Prior for balancing emotions (EMA)
        emo_prior = {k: 1.0/len(EMO_KEYS) for k in EMO_KEYS}
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
                # Face detection (backend configurable)
                x0, y0, x1, y1 = detect_largest_face(frame, gray, args.detector_backend, cascade, dnn_net, args.detect_scale)
                # Defaults are None to preserve previous values if nothing updates
                label = None
                recog_conf = None
                emotion = None
                emo_conf = None
                box = None
                pid = None
                if x0 is not None:
                    # Apply padding around face box for better emotion context
                    w = int(x1 - x0)
                    h = int(y1 - y0)
                    pad = float(args.crop_padding)
                    if pad > 0:
                        px = int(w * pad)
                        py = int(h * pad)
                        x0p = max(0, x0 - px)
                        y0p = max(0, y0 - py)
                        x1p = min(rgb.shape[1], x1 + px)
                        y1p = min(rgb.shape[0], y1 + py)
                    else:
                        x0p, y0p, x1p, y1p = x0, y0, x1, y1
                    # Box smoothing (EMA)
                    if sm_box is None:
                        sm_box = [float(x0p), float(y0p), float(x1p), float(y1p)]
                    else:
                        a = max(0.0, min(1.0, float(args.box_smooth_alpha)))
                        sm_box[0] = a * x0p + (1 - a) * sm_box[0]
                        sm_box[1] = a * y0p + (1 - a) * sm_box[1]
                        sm_box[2] = a * x1p + (1 - a) * sm_box[2]
                        sm_box[3] = a * y1p + (1 - a) * sm_box[3]
                    x0s, y0s, x1s, y1s = [int(v) for v in sm_box]
                    box = (int(x0s), int(y0s), int(x1s - x0s), int(y1s - y0s))
                    miss_count = 0
                    face_rgb = rgb[y0s:y1s, x0s:x1s]
                    # Downscale for embedding speed
                    if args.scale and args.scale != 1.0:
                        small = cv2.resize(face_rgb, (0, 0), fx=args.scale, fy=args.scale)
                    else:
                        small = face_rgb
                    emb = extract_embedding(small, model_name=args.embed_model)
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
                        # Emotion preprocessing and smoothing
                        emo_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                        emo_rgb = preprocess_emotion_roi(emo_bgr, upscale=max(1.0, float(args.emotion_scale)), use_clahe=True)
                        e_label, e_conf, e_dist = analyze_emotion_full(emo_rgb, backend=args.emotion_backend)
                        # Optional calibration: boost 'disgust' probability before smoothing
                        try:
                            gain = max(0.0, float(getattr(args, 'emo_disgust_gain', 1.0)))
                        except Exception:
                            gain = 1.0
                        if 'disgust' in e_dist and gain != 1.0:
                            e_dist['disgust'] = float(e_dist['disgust']) * gain
                            ssum = sum(e_dist.values()) or 1.0
                            e_dist = {k: float(v) / float(ssum) for k, v in e_dist.items()}
                        # Optional balancing: reweight by inverse prior (EMA-based) to uniformize 7 classes
                        try:
                            if bool(getattr(args, 'emo_balance', False)):
                                alpha = max(0.0, min(1.0, float(getattr(args, 'emo_balance_alpha', 0.15))))
                                beta = max(0.0, float(getattr(args, 'emo_balance_strength', 1.0)))
                                # Update prior with current distribution
                                for k in EMO_KEYS:
                                    emo_prior[k] = (1.0 - alpha) * float(emo_prior.get(k, 1.0/len(EMO_KEYS))) + alpha * float(e_dist.get(k, 0.0))
                                # Compute inverse-prior weights
                                eps = 1e-8
                                w = {k: (1.0 / max(emo_prior.get(k, eps), eps))**beta for k in EMO_KEYS}
                                # Apply weights and renormalize
                                e_dist = {k: float(e_dist.get(k, 0.0)) * float(w[k]) for k in EMO_KEYS}
                                ssum = sum(e_dist.values()) or 1.0
                                e_dist = {k: float(v) / float(ssum) for k, v in e_dist.items()}
                        except Exception:
                            pass
                        # If 'disgust' is close to the top class, gently tip it over to avoid being overshadowed
                        try:
                            top_k, top_v = max(e_dist.items(), key=lambda kv: kv[1])
                            dg = float(e_dist.get('disgust', 0.0))
                            margin = 0.05  # 5% margin
                            if top_k != 'disgust' and dg >= (top_v - margin) and dg >= 0.12:
                                # small boost then renormalize
                                e_dist['disgust'] = dg * 1.15
                                ssum = sum(e_dist.values()) or 1.0
                                e_dist = {k: float(v) / float(ssum) for k, v in e_dist.items()}
                        except Exception:
                            pass

                        # Smooth distribution over recent frames
                        if args.emo_smooth_frames and args.emo_smooth_frames > 1:
                            emo_hist.append(dict(e_dist))
                            if len(emo_hist) > int(args.emo_smooth_frames):
                                emo_hist.pop(0)
                            keys = list(e_dist.keys())
                            avg = {k: float(sum(d.get(k, 0.0) for d in emo_hist) / len(emo_hist)) for k in keys}
                            top = max(avg.items(), key=lambda kv: kv[1])
                            emotion, emo_conf, e_dist = top[0], float(top[1]), avg
                        else:
                            emotion, emo_conf = e_label, float(e_conf)
                        last_emo_ts = now
                        # Compose Top-3 string for overlay
                        try:
                            top3 = sorted(e_dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
                            top3_str = " | ".join([f"{k} {v*100:.0f}%" for k, v in top3])
                        except Exception:
                            top3_str = ''
                    # Save detection only if interval elapsed or emotion changed
                    if (emotion is not None) or (recog_conf is not None):
                        allow_log = False
                        if emotion is not None and emotion != last_logged_emotion:
                            allow_log = True
                        elif (now - last_log_ts) * 1000.0 >= float(getattr(args, 'min_log_interval_ms', 5000.0)):
                            allow_log = True
                        if allow_log:
                            cur_thr.execute(
                                "INSERT INTO Deteccion(id_persona, recog_confianza, emocion, emocion_confianza, timestamp) VALUES(?,?,?,?,?)",
                                (pid, float(recog_conf) if recog_conf is not None else 0.0, emotion if emotion is not None else 'neutral', float(emo_conf) if emo_conf is not None else 0.0, datetime.now())
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
                            last_log_ts = now
                            if emotion is not None:
                                last_logged_emotion = emotion
                else:
                    # No detection this cycle: increment miss counter; keep last box for a few cycles to reduce flicker
                    miss_count += 1
                    if miss_count >= 5:
                        sm_box = None
                        box = None
                # Publish results (keep last box up to 4 misses, then clear)
                with result_lock:
                    if label is not None:
                        result['label'] = label
                    if recog_conf is not None:
                        result['recog_conf'] = float(recog_conf)
                    if emotion is not None:
                        result['emotion'] = emotion
                    if emo_conf is not None:
                        result['emo_conf'] = float(emo_conf)
                    if 'top3_str' in locals():
                        result['emo_top3'] = top3_str
                    if box is not None:
                        result['box'] = box
                    else:
                        # If we missed recently, keep previous; after threshold (miss_count>=5) clear
                        if miss_count >= 5:
                            result['box'] = None
                        # else keep previous box
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
        # Top-3 emotions (debug/tuning)
        if r.get('emo_top3'):
            cv2.putText(frame, f"Top3: {r['emo_top3']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,0), 2)
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
    pd.add_argument('--emo-disgust-gain', type=float, default=1.6, help="Ganancia/calibracion para 'disgust' (1.0 = sin cambio)")
    pd.add_argument('--emo-balance', action='store_true', help='Balancear la distribucion de emociones hacia un prior uniforme (EMA)')
    pd.add_argument('--emo-balance-strength', type=float, default=1.0, help='Fuerza del balanceo (beta). 0=sin efecto, 1=balance total')
    pd.add_argument('--emo-balance-alpha', type=float, default=0.15, help='Alpha EMA para el prior de emociones (0-1)')
    # New advanced options
    pd.add_argument('--detector-backend', type=str, default='opencv', choices=['opencv','opencv-dnn','retinaface','mediapipe'], help='Detector de rostro para el recorte principal')
    pd.add_argument('--embed-model', type=str, default='Facenet', choices=['ArcFace','Facenet','VGG-Face'], help='Modelo de embeddings para reconocimiento')
    pd.add_argument('--emo-smooth-frames', type=int, default=5, help='Ventana de suavizado temporal para emociones (frames)')
    pd.add_argument('--force-mjpg', action='store_true', help='Forzar formato MJPG en la camara para bajar latencia CPU')
    pd.add_argument('--target-fps', type=float, default=0.0, help='Intentar fijar FPS objetivo de la camara (puede no tener efecto)')
    pd.add_argument('--box-smooth-alpha', type=float, default=0.5, help='Factor de suavizado exponencial para la caja del rostro (0-1)')
    pd.add_argument('--min-log-interval-ms', type=float, default=5000.0, help='Intervalo minimo entre registros consecutivos en DB (ms), a menos que cambie la emocion')

    args = parser.parse_args()
    if args.mode == 'register':
        register_mode(args)
    elif args.mode == 'detect':
        detection_mode(args)

if __name__ == '__main__':
    main()