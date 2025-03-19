import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import logging

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# YOLO-Logger unterdrücken
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_models(yolo_model_path):
    """
    Lädt das YOLO- und MiDaS-Modell.
    
    :param yolo_model_path: Pfad zum YOLO-Modell
    :return: Tuple (yolo_model, midas)
    """
    try:
        yolo_model = YOLO(yolo_model_path)
        logger.info("YOLO-Modell erfolgreich geladen.")
    except Exception as e:
        logger.error(f"Fehler beim Laden des YOLO-Modells: {e}")
        exit(1)
    
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        if isinstance(midas, torch.nn.Module) and callable(midas.eval):
            midas = midas.eval()
        else:
            logger.error("Fehler: Das geladene MiDaS-Modell unterstützt keine 'eval'-Methode.")
            exit(1)
        logger.info("MiDaS-Modell erfolgreich geladen.")
    except Exception as e:
        logger.error(f"Fehler beim Laden des MiDaS-Modells: {e}")
        exit(1)
    
    return yolo_model, midas

def initialize_camera():
    """
    Initialisiert die Kamera.
    
    :return: cv2.VideoCapture-Objekt
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Fehler: Kamera konnte nicht geöffnet werden.")
        exit(1)
    return cap

def process_frame(frame, yolo_model, midas, transform, device):
    """
    Verarbeitet ein Frame mit YOLO und MiDaS.
    
    :param frame: Eingabeframe
    :param yolo_model: YOLO-Modell
    :param midas: MiDaS-Modell
    :param transform: Transformations-Pipeline
    :param device: Gerät (CPU/GPU)
    :return: Tuple (frame_resized, depth_map, results)
    """
    # Frame für YOLO und MiDaS vorbereiten
    frame_resized = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_tensor = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    
    # MiDaS Depth Prediction
    with torch.no_grad():
        depth_map = midas(img_tensor)
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # YOLO-Erkennung mit Tracking
    results = yolo_model.track(frame_resized, conf=0.5, persist=True)
    
    return frame_resized, depth_map, results

def calculate_distance(depth_map, x1, y1, x2, y2, frame_shape):
    """
    Berechnet die Entfernung basierend auf der Tiefenkarte.
    
    :param depth_map: Tiefenkarte
    :param x1, y1, x2, y2: Koordinaten der Bounding-Box
    :param frame_shape: Form des Frames (Höhe, Breite)
    :return: Entfernung in Metern
    """
    # Skaliere die Bounding-Box-Koordinaten auf die Größe der depth_map (256x256)
    scale_x = 256 / frame_shape[1]
    scale_y = 256 / frame_shape[0]
    x1_scaled = int(x1 * scale_x)
    y1_scaled = int(y1 * scale_y)
    x2_scaled = int(x2 * scale_x)
    y2_scaled = int(y2 * scale_y)

    # Berechne das Center der skalierten Bounding-Box
    center_x = (x1_scaled + x2_scaled) // 2
    center_y = (y1_scaled + y2_scaled) // 2

    # Sicherheit: Stelle sicher, dass die Koordinaten im Bild liegen
    if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
        depth_value = depth_map[center_y, center_x]
        if depth_value < 0.01:
            depth_value = 0.01  # Vermeidung von extrem niedrigen Werten
        distance = 1 / (depth_value + 0.01)  # Umkehrung der Werte
        return distance
    return None

def main():
    # Modelle laden
    yolo_model, midas = load_models("last.pt")
    
    # Gerät festlegen (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = midas.to(device)
    yolo_model.to(device)
    logger.info(f"Modelle auf Gerät verschoben: {device}")
    
    # MiDaS Transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Kamera initialisieren
    cap = initialize_camera()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Fehler: Frame konnte nicht gelesen werden.")
            break
        
        # Frame verarbeiten
        frame_resized, depth_map, results = process_frame(frame, yolo_model, midas, transform, device)
        
        # Personen erkennen und deren Abstand berechnen
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box, cls, track_id in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
                if int(cls) == 0:  # Klasse 0 = Person
                    x1, y1, x2, y2 = map(int, box[:4])
                    distance = calculate_distance(depth_map, x1, y1, x2, y2, frame_resized.shape)
                    
                    if distance is not None:
                        # Handlungsempfehlung basierend auf der Entfernung
                        if distance < 3:
                            action = "STOP! Person zu nah."
                        elif 3 <= distance <= 7:
                            action = "Langsam fahren. Person in der Nähe."
                        else:
                            action = "Normal fahren. Person in sicherer Entfernung."
                        
                        logger.info(f"Person {int(track_id)} erkannt: {distance:.2f} Meter entfernt. {action}")
                    else:
                        logger.warning(f"Person {int(track_id)} - Center außerhalb des Bildes.")
                else:
                    logger.info(f"Objekt {int(track_id)} ist keine Person (Klasse: {int(cls)}).")
        else:
            logger.info("Keine IDs für Tracking vorhanden.")
        
        # YOLO-Tracking-Fenster anzeigen
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Tracking", annotated_frame)
        
        # Tiefenkarte für Visualisierung
        depth_colormap = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imshow("Depth Map", depth_colormap)
        
        # Beenden bei Drücken der 'q'-Taste
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Programm beendet.")
            break
    
    # Ressourcen freigeben
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()