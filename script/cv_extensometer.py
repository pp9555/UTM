import cv2
import numpy as np
import websocket
import json
import time

# --- Configuration ---
WEBSOCKET_URL = "ws://localhost:8080" # Connects to your Node.js relay
PIXELS_TO_MM = 0.1 # Calibration factor (Adjust based on camera distance)
FAKE_STRESS_MULTIPLIER = 5.0 # Just for testing at your desk

# Global variables for mouse click tracking
points = []
tracking = False

# def select_points(event, x, y, flags, param):
#     global points, tracking
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(points) < 2:
#             points.append((x, y))
#             print(f"Point {len(points)} selected: ({x}, {y})")
#         if len(points) == 2:
#             tracking = True

def select_points(event, x, y, flags, param):
    global points # removed 'tracking' from global here
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)} selected: ({x}, {y})")

def main():
    global points, tracking
    
    # Connect to the WebSocket relay
    ws = websocket.WebSocket()
    try:
        ws.connect(WEBSOCKET_URL)
        print("Connected to Node.js Dashboard.")
    except Exception as e:
        print(f"Could not connect to WebSocket: {e}")
        return

    cap = cv2.VideoCapture(0) # 0 for default webcam
    cv2.namedWindow('Virtual Extensometer')
    cv2.setMouseCallback('Virtual Extensometer', select_points)

    print("--- INSTRUCTIONS ---")
    print("1. Point camera at your hand/sample.")
    print("2. Click two points on the screen to track (e.g., two fingertips).")
    print("3. Press 'q' to quit.")

    # Optical Flow Parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_gray = None
    p0 = None
    initial_distance = None

    # Time tracking to control data stream rate (e.g., 10 Hz)
    last_send_time = time.time()
    send_interval = 0.1 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Draw instructions if not tracking
       # Draw instructions if not tracking
        if not tracking:
            cv2.putText(frame, "Click 2 points to start tracking", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            
            old_gray = frame_gray.copy()
            if len(points) == 2:
                    # Initialize points for optical flow
                    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                    # Calculate the baseline L0
                    initial_distance = np.linalg.norm(p0[0] - p0[1])
                    
                    # --- NEW SAFETY CHECK ---
                    if initial_distance < 5: # If points are less than 5 pixels apart
                        cv2.putText(frame, "Points too close! Restart.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        points = [] # Reset points
                    else:
                        tracking = True

        # Execute tracking
        if tracking and old_gray is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # If tracking is successful
            if st[0] == 1 and st[1] == 1:
                # Draw the tracked points and the virtual extensometer line
                a, b = p1[0].ravel()
                c, d = p1[1].ravel()
                pt1 = (int(a), int(b))
                pt2 = (int(c), int(d))
                
                cv2.circle(frame, pt1, 5, (0, 255, 0), -1)
                cv2.circle(frame, pt2, 5, (0, 255, 0), -1)
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                # Calculate Strain
                current_distance = np.linalg.norm(p1[0] - p1[1])
                delta_l = current_distance - initial_distance
                strain = delta_l / initial_distance if initial_distance > 0 else 0
                
                # We fake stress for desk testing (normally this comes from a load cell)
                fake_stress = strain * FAKE_STRESS_MULTIPLIER

                # Display data on screen
                cv2.putText(frame, f"Strain: {strain:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Stream to Node.js Dashboard (limit to 10Hz to avoid overwhelming the browser)
                current_time = time.time()
                if current_time - last_send_time > send_interval:
                    payload = json.dumps({"strain": float(round(strain, 4)), "stress": float(round(fake_stress, 2))})
                    ws.send(payload)
                    last_send_time = current_time

                # Update old frame and points for the next loop
                old_gray = frame_gray.copy()
                p0 = p1.copy()
            else:
                cv2.putText(frame, "TRACKING LOST - Restart", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tracking = False
                points = []

        cv2.imshow('Virtual Extensometer', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    ws.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()