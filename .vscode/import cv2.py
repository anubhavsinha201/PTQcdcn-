import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Game State Variables
bomb_code = [random.randint(0, 5) for _ in range(4)]
current_digit_index = 0
bomb_defused = False
game_over = False

# Debouncing & Confirmation Variables
current_detected_fingers = -1
last_detected_fingers = -1
hand_steady_start_time = 0
CONFIRMATION_TIME = 2.0  # Seconds to hold the gesture

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

def count_raised_fingers(hand_landmarks):
    """
    Counts raised fingers comparing the fingertip Y-coordinate 
    against the knuckle (PIP joint) Y-coordinate.
    """
    # Landmark IDs for tips and corresponding knuckles
    # Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    
    count = 0
    
    # 1. Check 4 fingers (Lower Y value means higher on screen)
    for tip, knuckle in zip(tips, knuckles):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[knuckle].y:
            count += 1
            
    # 2. Check Thumb (Uses X-coordinate comparison)
    # Compares thumb tip (4) with thumb IP joint (3)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]
    
    # Determine hand orientation (Left vs Right) based on wrist vs middle finger MCP
    # Simplified approach: check if thumb tip is extended outward
    if hand_landmarks.landmark[17].x > hand_landmarks.landmark[5].x: # Palm facing camera (Right Hand)
        if thumb_tip.x > thumb_ip.x: count += 1
    else: # Palm facing camera (Left Hand)
        if thumb_tip.x < thumb_ip.x: count += 1
            
    return count

print(f"BOMB CODE TO DETECT: {bomb_code}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip horizontally for a mirror view, convert to RGB
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    detected_fingers = -1
    
    # Draw hand landmarks and detect count
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)

    # Game Logic UI Overlay
    if not game_over:
        # Target Code Display
        code_str = " ".join([str(x) for x in bomb_code])
        cv2.putText(frame, f"BOMB CODE: {code_str}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Current Target Pointer
        pointer = " " * (current_digit_index * 2) + "^"
        cv2.putText(frame, f"           {pointer}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Live Status
        if detected_fingers != -1:
            cv2.putText(frame, f"Detected: {detected_fingers}", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            target_digit = bomb_code[current_digit_index]
            
            # Debounce and Confirmation Logic
            if detected_fingers == target_digit:
                if last_detected_fingers != detected_fingers:
                    # Timer starts when you first flash the correct number
                    hand_steady_start_time = time.time()
                    last_detected_fingers = detected_fingers
                
                elapsed = time.time() - hand_steady_start_time
                remaining = max(0.0, CONFIRMATION_TIME - elapsed)
                
                # Progress Bar
                bar_width = int((elapsed / CONFIRMATION_TIME) * 200)
                bar_width = min(bar_width, 200)
                cv2.rectangle(frame, (30, 150), (230, 170), (100, 100, 100), -1)
                cv2.rectangle(frame, (30, 150), (30 + bar_width, 170), (0, 255, 0), -1)
                cv2.putText(frame, f"Holding... {remaining:.1f}s", (240, 165), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if elapsed >= CONFIRMATION_TIME:
                    current_digit_index += 1
                    last_detected_fingers = -1  # Reset debounce
                    
                    if current_digit_index >= 4:
                        bomb_defused = True
                        game_over = True
            else:
                # Reset if they show the wrong number or drop the gesture
                last_detected_fingers = -1
                cv2.putText(frame, f"Match Code Digit: {target_digit}", (30, 165), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Show Hand to System Override", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            last_detected_fingers = -1
            
    else:
        # Game Over Screen
        if bomb_defused:
            cv2.rectangle(frame, (50, h//2 - 60), (w - 50, h//2 + 40), (0, 128, 0), -1)
            cv2.putText(frame, "SUCCESS: GRID SECURED", (80, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
        else:
            cv2.rectangle(frame, (50, h//2 - 60), (w - 50, h//2 + 40), (0, 0, 128), -1)
            cv2.putText(frame, "SYSTEM DETONATION", (100, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)

    # Display window
    cv2.imshow("Biometric Grid Override", frame)
    
    # Break loop on 'q' key or ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
