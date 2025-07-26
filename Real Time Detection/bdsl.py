import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# Load saved CNN model
model = tf.keras.models.load_model("cnn_model.keras")

# Your label list from training
CLASSES = ["আজ", "বাসা", "বিয়োগ", "বন্ধু", "দাঁড়ানো", "দাড়াও", "দেশ", "এখানে", "গুণ", "কিছুটা", "কোথায়", "অনুরোধ", "সাহায্য", "সে", "সময়", "সুন্দর", "০", "অ/য", "আ", "ই/ঈ", "উ/ঊ", "র/ঋ", "এ", "ঐ", "ও", "ঔ", "ক", "১", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ/য", "ঝ", "ঞ", "ট", "২", "ঠ", "ড", "ঢ", "ণ/ন", "ত", "থ", "দ", "ধ", "প", "ফ", "৩", "ব/ভ", "ম", "ল", "শ/ষ/স", "হ", "ং", "◌ং", "৪", "৫", "৬", "৭", "৮", "৯", "স্যার", "তারা", "তুমি", "বাঘ", "বৌদ্ধ", "চামড়া", "গির্জা", "হকি", "জেল", "কেরাম", "পিয়ানো", "পুরু", "সমাজকল্যান", "সত্য"]

IMG_SIZE = 64

def preprocess_roi(roi):
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

# Load Bangla font
try:
    bangla_font = ImageFont.truetype("kalpurush.ttf", 32)
except IOError:
    print("Bangla font file 'kalpurush.ttf' not found. Please ensure it's in the script directory.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box from landmarks
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            # Clamp coordinates to image size
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Predict using model
            processed = preprocess_roi(hand_img)
            predictions = model.predict(processed)
            class_index = np.argmax(predictions)
            predicted_label = CLASSES[class_index]

            # Convert OpenCV image to PIL image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Draw Bangla text
            text = f"{predicted_label}"
            # text = f"{predicted_label} ({class_index})"
            draw.text((x_min, y_min - 40), text, font=bangla_font, fill=(255, 0, 0))

            # Convert back to OpenCV image
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # Optional: draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Bangla Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
