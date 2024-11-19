import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from exercise_tracker import ExerciseTracker

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create exercise tracker instance
tracker = ExerciseTracker()

@app.get("/")
async def get_index():
    """Serve a basic HTML page (you'd replace this with your actual frontend)"""
    return HTMLResponse(content="<h1>Exercise Tracker WebSocket Server</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for processing video frames"""
    await websocket.accept()
    
    try:
        while True:
            # Receive video frame data
            data = await websocket.receive_text()
            
            try:
                # If it's a full data URL, split it
                if data.startswith('data:image/jpeg;base64,'):
                    encoded_data = data.split(',')[1]
                else:
                    encoded_data = data
                
                # Decode base64 image
                image_bytes = base64.b64decode(encoded_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Check if frame is valid
                if frame is None or frame.size == 0:
                    print("Failed to decode image")
                    continue
                
                # Resize frame if it's too large
                max_width = 640
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Process frame with ExerciseTracker
                processed_frame = tracker.process_frame(frame)
                
                # Encode processed frame to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Get exercise state
                exercise_state = tracker.get_exercise_state()
                
                # Send processed frame and exercise state back to client
                await websocket.send_json({
                    'image': processed_frame_base64, 
                    'state': exercise_state
                })
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                
    except WebSocketDisconnect:
        print("Client disconnected")

# Optional: If you want to run with uvicorn directly in the script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)