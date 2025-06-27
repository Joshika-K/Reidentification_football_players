This project uses a custom-trained model(best.pt) and DeepSORT for accurate player detection and re-identification in football video(15sec_input_720p.mp4). It assigns consistent player numbers and generates annotated output video, logs, and visualizations to assess tracking quality.

How to Set Up & Run the Football Player Re-identification Project(GOOGLE COLAB):

Step 1: Open Google Colab
Go to https://colab.research.google.com
Create a new notebook

Step 2: Install the requirements by running this:
!pip install ultralytics==8.1.34 opencv-python-headless==4.9.0.80 \
             deep_sort_realtime==1.3.1 gdown==4.7.1 matplotlib==3.8.4 \
             torch==2.0.1 torchvision==0.15.2 pillow==9.4.0 numpy==1.24.4

Step 3: Upload the 15sec_input_720p.mp to your drive(MyDrive)
Path should be like:
/content/drive/MyDrive/15sec_input_720p.mp4

Step 4: Copy and paste the track_players.py into the new cell(cloning is not required)
Two files will get downloaded, one the final video and the other csv file.
Allow the browser to download multiple files.

Step 5: Analyze the result(final video and csv file)

Step 6: Copy and paste the visualization.py for visualization
Bar chart and heat map will be displayed

NOTE
Ensure GPU is enabled via Runtime > Change runtime type > GPU in Colab for faster processing.
The model (best.pt) is auto-downloaded from Google Drive using gdown.
Only the top 22 consistent players are assigned player numbers Player 1 to Player 22.

Features:
Custom YOLOv11 model (best.pt) for detecting football players.
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







