This project uses a custom-trained YOLOv11 model(best.pt) and DeepSORT for accurate player detection and re-identification in football video(15sec_input_720p.mp4). It assigns consistent player numbers and generates annotated output video, logs, and visualizations to assess tracking quality.

How to Set Up & Run the Football Player Re-identification Project(GOOGLE COLAB):

Step 1: Open Google Colab
Go to https://colab.research.google.com
Create a new notebook

Step 2: Clone the GitHub Repository
Paste the following in the first cell:
!git clone https://github.com/Joshika-K/Reidentification_football_players.git
%cd Reidentification_football_players

Step 3: Install Required Libraries
Install all dependencies needed for your script:
!pip install -r requirements.txt

Step 4: Mount Google Drive (for video access).
Make sure your video file 15sec_input_720p.mp4 is placed in:
/content/drive/MyDrive/15sec_input_720p.mp4

Step 5: Run the Script
!python track_players.py
Make sure your script already contains the correct path:
video_path = "/content/drive/MyDrive/15sec_input_720p.mp4"

Step 6: Analyze the result
The code will automatically download all the files on to you the browser
If it doesn't then, in the new cell please type:
from google.colab import files
files.download("final_output_fixed.mp4")
files.download("final_log_fixed.csv")
files.download("debug_sample_fixed.jpg")

Step 7: Run the code for visualization: 
 !python visualization.py

NOTE
Ensure GPU is enabled via Runtime > Change runtime type > GPU in Colab for faster processing.
The model (best.pt) is auto-downloaded from Google Drive using gdown.
Only the top 22 consistent players are assigned player numbers Player 1 to Player 22.

Features:
Custom YOLOv8 model (best.pt) for detecting football players.
DeepSORT for robust object re-identification across frames.
Input: Football match video (.mp4).
Output: Annotated video with consistent player numbers.
Re-ID Accuracy Metric: Measures ID consistency.
Visualizations:
1) Player tracking consistency (bar chart).
2) Player presence timeline (heatmap).
Runs end-to-end in a single script.

Sample Visualizations
Bar Chart: Player appearance consistency across frames.
Heatmap: Player presence over time (frame vs player matrix).

Files Overview:
File	                       Description
best.pt               	      YOLOv8 model for player detection
track_players.py              Actual reidentification code
final_output_fixed.mp4       	Video with annotated tracking
final_log_fixed.csv	CSV       log of player positions per frame
debug_sample_fixed.jpg	      Debug snapshot from tracking
requirements.txt	            Python dependencies
README.md	                    This documentation file




