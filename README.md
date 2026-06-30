# CricketEye — Ball Tracker

CricketEye is a real-time cricket ball tracking application built with web technologies and Apache Capacitor. It uses your device's camera and computer vision (OpenCV) to track the trajectory of a cricket ball, calculate its speed, and determine its position relative to a drawn line on the pitch.

## Features
- **Real-time Ball Tracking:** Tracks red, white, pink, or green cricket balls using your device's camera.
- **Speed Estimation:** Calculates the approximate speed of the ball based on its trajectory.
- **Line Detection & Side Tracking:** Draw a line manually or use auto-detect to find the pitch line. The app determines which side of the line the ball is on.
- **Customizable Settings:** Adjust ball color, detection sensitivity, and trail length.
- **Cross-Platform:** Runs natively on iOS and Android via Capacitor, or directly in a web browser.

## How It Works (Under the Hood)
The core tracking pipeline runs locally on your device using `opencv.js`:
1. **Camera Input:** The app captures video frames continuously from the device's rear camera.
2. **RGB to HSV Conversion:** Video frames are converted to the HSV (Hue, Saturation, Value) color space. This isolates color from brightness, making tracking robust under varying lighting conditions.
3. **Color Thresholding:** Each pixel is tested against specific HSV ranges for the selected ball color.
4. **Blob Detection:** The application scans a grid to find the densest region of matching pixels, calculating a weighted centroid to find the ball's precise center.
5. **Optical Flow:** The Lucas-Kanade optical flow algorithm tracks the drawn pitch line across frames, ensuring it stays anchored even if the camera moves slightly.
6. **Analytics & Rendering:** The ball's coordinates are stored in a buffer to render a trail and calculate the speed between consecutive frames.

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) and npm installed on your machine.
- **For iOS:** macOS with [Xcode](https://developer.apple.com/xcode/) installed.
- **For Android:** [Android Studio](https://developer.android.com/studio) installed.

### Installation
Navigate to the project directory and install the dependencies:
```bash
npm install
```

### Running on iOS
1. Sync the web assets with the native iOS project:
   ```bash
   npx cap sync ios
   ```
2. Open the project in Xcode:
   ```bash
   npx cap open ios
   ```
3. In Xcode, select your connected iOS device or a simulator, and click the **Run** button.

### Running on Android
1. Sync the web assets with the native Android project:
   ```bash
   npx cap sync android
   ```
2. Open the project in Android Studio:
   ```bash
   npx cap open android
   ```
3. In Android Studio, wait for the Gradle sync to complete, choose your connected Android device or an emulator, and click the **Run** button.

### Running on Web
To test the web app directly in your browser, serve the `www` directory using any local web server. For example:
```bash
npx serve www
```
Then, open the provided localhost URL in your browser. 
*Note: Browser security policies generally require localhost or HTTPS to grant camera access.*
