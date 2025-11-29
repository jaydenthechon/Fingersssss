import cv2
import mediapipe as mp
import time

class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def count_fingers(self, hand_landmarks, handedness):
        """
        Count the number of extended fingers on a hand.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: Whether the hand is 'Left' or 'Right'
            
        Returns:
            Number of extended fingers (0-5)
        """
        fingers = []
        
        # Landmark indices
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        
        # Get hand label
        is_right_hand = handedness == "Right"
        
        # Check thumb
        if is_right_hand:
            # For right hand, thumb is extended if tip is to the right of IP joint
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # For left hand, thumb is extended if tip is to the left of IP joint
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Check other 4 fingers
        for id in range(1, 5):
            # Finger is extended if tip is higher than PIP joint (lower y value = higher on screen)
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)
    
    def run(self):
        """Main loop to capture video and detect fingers."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        prev_time = 0
        
        print("Starting finger counter...")
        print("Press 'q' to quit")
        
        while True:
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # Flip the image horizontally for a mirror view
            image = cv2.flip(image, 1)
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = self.hands.process(image_rgb)
            
            # Draw hand landmarks and count fingers
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get hand label (Left or Right)
                    hand_label = handedness.classification[0].label
                    
                    # Count fingers
                    finger_count = self.count_fingers(hand_landmarks, hand_label)
                    
                    # Get hand position for text placement
                    h, w, c = image.shape
                    hand_x = int(hand_landmarks.landmark[0].x * w)
                    hand_y = int(hand_landmarks.landmark[0].y * h)
                    
                    # Display finger count
                    cv2.putText(
                        image, 
                        f"{hand_label}: {finger_count} fingers", 
                        (hand_x - 50, hand_y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            cv2.putText(
                image, 
                f"FPS: {int(fps)}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
            
            # Show the image
            cv2.imshow('Finger Counter', image)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    counter = FingerCounter()
    counter.run()
