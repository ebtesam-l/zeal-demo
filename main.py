from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import tempfile
import os
from zeal import ZEAL
app = FastAPI()
zeal = ZEAL()  # Initialize your ZEAL class

@app.post("/detect_action")
async def detect_action(
    video: UploadFile = File(...),
    action_name: str = Form(...)
) -> List[dict]:
    """
    Endpoint to detect action intervals in uploaded video
    """
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await video.read()
            temp_video.write(content)
            video_path = temp_video.name
        
        # Detect action intervals
        intervals = zeal.detect_action_intervals(
            video_path=video_path,
            action_name=action_name,
            top_k=3
        )
        
        # Clean up temporary file
        os.unlink(video_path)
        
        return intervals
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"error": str(e)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8800)    
