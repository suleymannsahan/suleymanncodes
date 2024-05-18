import cv2
import mediapipe as mp 

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils


while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900, 725))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame)
    

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            

        if len(hand_landmarks.landmark) == 21:
            thumb_tip = hand_landmarks.landmark[4]
            finger_tip = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            ring_finger = hand_landmarks.landmark[16]
            little_tip = hand_landmarks.landmark[20]

            fingers_open = sum(1 for lm in [thumb_tip, finger_tip, middle_finger, ring_finger, little_tip] if lm.y < middle_finger.y)
                
            cv2.putText(frame, f"Fingers: {fingers_open}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord ("q"):
        break

cap.release()
cv2.destroyAllWindows()
