import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import logging

# Logging-Konfiguration: sehen was im Code passiert
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# YOLO-Logger etwas leiser machen, sonst spammt der zu viel
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_models(yolo_model_path):
    """
    Lädt das YOLO- und MiDaS-Modell.
    YOLO erkennt Objekte, MiDaS schätzt die Tiefe (wie weit etwas weg ist).
    
    :param yolo_model_path: Pfad zum YOLO-Modell (z.B. 'last.pt')
    :return: Tuple (yolo_model, midas) – die geladenen Modelle
    """
    try:
        # YOLO-Modell laden
        yolo_model = YOLO(yolo_model_path)
        logger.info("YOLO-Modell erfolgreich geladen. Los geht's!")
    except Exception as e:
        logger.error(f"Fehler beim Laden des YOLO-Modells: {e}")
        exit(1) 
    
    try:
        # MiDaS-Modell laden (für Tiefenschätzung)
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        if isinstance(midas, torch.nn.Module) and callable(midas.eval):
            midas = midas.eval()  # Modell in den "Eval-Modus" versetzen
        else:
            logger.error("Fehler: MiDaS-Modell ist komisch, kann nicht 'eval' machen.")
            exit(1)
        logger.info("MiDaS-Modell erfolgreich geladen. Tiefenschätzung steht bereit!")
    except Exception as e:
        logger.error(f"Fehler beim Laden des MiDaS-Modells: {e}")
        exit(1)  
    
    return yolo_model, midas

def initialize_camera():
    """
    Initialisiert die Kamera. Ohne Kamera geht hier gar nichts.
    
    :return: cv2.VideoCapture-Objekt – Kamera
    """
    for i in range(3):  # Probiere die ersten 3 Kameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            logger.info(f"Kamera {i} erfolgreich geöffnet.")
            return cap
    logger.error("Fehler: Keine Kamera gefunden. Kein Bild, kein Spaß.")
    exit(1)

def process_frame(frame, yolo_model, midas, transform, device):
    """
    Verarbeitet ein Frame mit YOLO und MiDaS.
    YOLO erkennt Objekte, MiDaS schätzt die Tiefe.
    
    :param frame: Das aktuelle Kamerabild
    :param yolo_model: YOLO-Modell
    :param midas: MiDaS-Modell
    :param transform: Transformations-Pipeline (für MiDaS)
    :param device: CPU oder GPU (je nachdem, was verfügbar ist)
    :return: Tuple (frame_resized, depth_map, results) – das verarbeitete Bild, die Tiefenkarte und die YOLO-Ergebnisse
    """
    # Frame für YOLO und MiDaS vorbereiten
    frame_resized = cv2.resize(frame, (640, 480))  # Bild skalieren
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Farbraum von BGR zu RGB ändern
    img_tensor = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)  # Bild in Tensor umwandeln und auf Gerät schieben
    
    # MiDaS Depth Prediction
    with torch.no_grad():  # Keine Gradienten berechnen (spart an Rechenleistung)
        depth_map = midas(img_tensor)  # Tiefenkarte berechnen
    depth_map = depth_map.squeeze().cpu().numpy()  # Tensor in Numpy-Array umwandeln
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)  # Tiefenkarte glätten
    # depth_map = cv2.medianBlur(depth_map, 5)  # Noch mehr Glättung, falls notwendig
    if depth_map.max() != depth_map.min():
       depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    else:
       depth_map = np.zeros_like(depth_map)  # Fallback, falls alle Werte gleich sind
    
    # YOLO-Erkennung mit Tracking
    results = yolo_model.track(frame_resized, conf=0.5, persist=True)  # Objekte erkennen und tracken - conf = Konfidenzschwelle (wie viele Objekte erkannt werden)
    
    return frame_resized, depth_map, results

def calculate_distance(depth_map, x1, y1, x2, y2, frame_shape):
    """
    Berechnet die Entfernung basierend auf der Tiefenkarte.
    Die Tiefenkarte sagt uns, wie weit etwas weg ist.
    
    :param depth_map: Die Tiefenkarte (von MiDaS)
    :param x1, y1, x2, y2: Koordinaten der Bounding-Box (umrandetes Objekt)
    :param frame_shape: Form des Frames (Höhe, Breite)
    :return: Entfernung in Metern (oder None, wenn etwas schiefgeht)
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
        distance = 1.2 / (depth_value + 0.01)  # Umkehrung der Werte - Um genauer zu machen: muss Entfernungsberechnung an die Kalibrierung der Kamera angepasst werden
        return distance
    return None  # Wenn das Center außerhalb des Bildes liegt

def main():
    # Modelle laden
    yolo_model, midas = load_models("last.pt")  # YOLO-Modell von 'last.pt' laden
    
    # Gerät festlegen (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU, falls verfügbar
    midas = midas.to(device)
    yolo_model.to(device)
    logger.info(f"Modelle auf Gerät verschoben: {device}")
    
    # MiDaS Transformation: Bild für MiDaS vorbereiten
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Bild auf 256x256 skalieren
        transforms.ToTensor(),  # Bild in Tensor umwandeln
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisieren
    ])
    
    # Kamera initialisieren
    cap = initialize_camera()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Fehler: Frame konnte nicht gelesen werden. Kamera kaputt?")
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
                        # Entfernung Output
                        if distance < 3:
                            action = "STOP! Person zu nah."
                        elif 3 <= distance <= 7:
                            action = "Langsam fahren. Person in der Nähe."
                        else:
                            action = "Normal fahren. Person in sicherer Entfernung."
                        
                        logger.info(f"Person {int(track_id)} erkannt: {distance:.2f} Meter entfernt. {action}")
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
            logger.info("Programm beendet. Tschüss!")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()