import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os


class RealTimeEmotionDetector:
    def __init__(self, model_path):
        """Initialize the real-time emotion detector"""

        # Emotion labels - adjust these based on your training
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Colors for each emotion (BGR format for OpenCV)
        self.colors = {
            'Angry': (0, 0, 255),  # Red
            'Disgust': (0, 128, 0),  # Green
            'Fear': (128, 0, 128),  # Purple
            'Happy': (0, 255, 255),  # Yellow
            'Sad': (255, 0, 0),  # Blue
            'Surprise': (255, 165, 0),  # Orange
            'Neutral': (128, 128, 128)  # Gray
        }

        # Load your trained model
        print("🤖 Loading your trained model...")
        try:
            self.model = load_model("D:/FACIAL_EXPRESSION/best_emotion_model_enhanced_20250727_174841.keras")
            print("✅ Model loaded successfully!")
            print(f"📊 Model input shape: {self.model.input_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try different model file (.keras, .h5)")
            raise

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0

        print("🎯 Real-time emotion detector initialized!")
        print("👤 Looking for faces...")

    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        try:
            # Resize to model input size (48x48)
            face_resized = cv2.resize(face_img, (48, 48))

            # Convert to grayscale if needed
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized

            # Normalize pixel values to [0, 1]
            face_normalized = face_gray.astype('float32') / 255.0

            # Reshape for model input (1, 48, 48, 1)
            face_input = face_normalized.reshape(1, 48, 48, 1)

            return face_input

        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return None

    def predict_emotion(self, face_input):
        """Predict emotion from preprocessed face"""
        try:
            # Make prediction
            predictions = self.model.predict(face_input, verbose=0)

            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Get emotion name
            emotion = self.emotions[predicted_class]

            return emotion, confidence, predictions[0]

        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return "Unknown", 0.0, None

    def draw_results(self, frame, x, y, w, h, emotion, confidence, all_predictions=None):
        """Draw detection results on frame"""

        # Get color for this emotion
        color = self.colors.get(emotion, (255, 255, 255))

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Prepare text
        label = f"{emotion}: {confidence:.2f}"

        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw text background
        cv2.rectangle(frame, (x, y - text_height - 10),
                      (x + text_width, y), color, -1)

        # Draw text
        cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

        # Draw confidence bar
        bar_width = w
        bar_height = 10
        bar_filled = int(bar_width * confidence)

        # Background bar
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 5 + bar_height), (50, 50, 50), -1)
        # Filled bar
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_filled, y + h + 5 + bar_height), color, -1)

        # Show all emotion probabilities (optional)
        if all_predictions is not None and len(all_predictions) == len(self.emotions):
            y_offset = y + h + 25
            for i, (emo, prob) in enumerate(zip(self.emotions, all_predictions)):
                text = f"{emo}: {prob:.2f}"
                cv2.putText(frame, text, (x + w + 10, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1

        if self.fps_counter % 30 == 0:  # Update every 30 frames
            end_time = time.time()
            self.fps = 30 / (end_time - self.fps_start_time)
            self.fps_start_time = end_time

    def run_detection(self, show_all_emotions=False, min_confidence=0.3):
        """Run real-time emotion detection"""

        print("🚀 Starting real-time emotion detection...")
        print("📹 Initializing webcam...")

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            print("💡 Check if webcam is connected and not used by another app")
            return

        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✅ Webcam initialized successfully!")
        print("\n🎯 CONTROLS:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'a' to toggle all emotions display")
        print("   - Press SPACE to pause/unpause")

        paused = False
        screenshot_count = 0

        try:
            while True:
                if not paused:
                    # Read frame from webcam
                    ret, frame = cap.read()

                    if not ret:
                        print("❌ Error reading from webcam")
                        break

                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)

                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    # Process each detected face
                    for (x, y, w, h) in faces:
                        # Extract face region
                        face_roi = frame[y:y + h, x:x + w]

                        # Preprocess face
                        face_input = self.preprocess_face(face_roi)

                        if face_input is not None:
                            # Predict emotion
                            emotion, confidence, all_preds = self.predict_emotion(face_input)

                            # Only show results above minimum confidence
                            if confidence >= min_confidence:
                                # Draw results
                                self.draw_results(frame, x, y, w, h, emotion, confidence,
                                                  all_preds if show_all_emotions else None)
                            else:
                                # Draw face rectangle with low confidence indicator
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                                cv2.putText(frame, "Low Confidence", (x, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

                    # Update FPS
                    self.update_fps()

                # Draw FPS and info
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if paused:
                    cv2.putText(frame, "PAUSED - Press SPACE to continue", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show frame
                cv2.imshow('Real-Time Emotion Detection', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_path = f'emotion_detection_screenshot_{screenshot_count}.jpg'
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
                elif key == ord('a'):
                    show_all_emotions = not show_all_emotions
                    print(f"🔄 All emotions display: {'ON' if show_all_emotions else 'OFF'}")
                elif key == ord(' '):  # Space bar
                    paused = not paused
                    print(f"⏸️ {'Paused' if paused else 'Resumed'}")

        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("🧹 Cleanup completed")


# ✅ USAGE FUNCTION
def run_webcam_emotion_detection():
    """Main function to run webcam emotion detection"""

    print("🎯 REAL-TIME EMOTION DETECTION SYSTEM")
    print("=" * 50)

    # Find your trained model
    print("🔍 Looking for your trained model...")

    # Possible model file patterns
    model_patterns = [
        '*enhanced*.keras',
        '*emotion*.keras',
        '*best*.keras',
        '*emotion*.h5',
        '*best*.h5'
    ]

    import glob
    found_models = []

    for pattern in model_patterns:
        models = glob.glob(pattern)
        found_models.extend(models)

    if found_models:
        print("📁 Found trained models:")
        for i, model in enumerate(found_models):
            print(f"   {i + 1}. {model}")

        # Auto-select the most recent or let user choose
        if len(found_models) == 1:
            model_path = found_models[0]
            print(f"✅ Using model: {model_path}")
        else:
            try:
                choice = int(input("\nEnter model number to use: ")) - 1
                model_path = found_models[choice]
                print(f"✅ Selected model: {model_path}")
            except:
                model_path = found_models[0]  # Use first one as default
                print(f"✅ Using default model: {model_path}")
    else:
        # Manual path input
        print("❌ No model files found automatically")
        model_path = input("📝 Enter the full path to your trained model: ").strip()

        if not os.path.exists(model_path):
            print("❌ Model file not found!")
            return

    try:
        # Initialize detector
        detector = RealTimeEmotionDetector(model_path)

        # Run detection
        detector.run_detection(
            show_all_emotions=False,  # Set to True to see all emotion probabilities
            min_confidence=0.3  # Minimum confidence threshold
        )

    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        print("\n🛠️ TROUBLESHOOTING:")
        print("   1. Make sure your model file exists")
        print("   2. Check if webcam is connected")
        print("   3. Close other apps using the webcam")
        print("   4. Try different model file formats (.keras vs .h5)")


# 🚀 RUN THE WEBCAM DETECTION
if __name__ == "__main__":
    run_webcam_emotion_detection()
