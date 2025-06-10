export class Capturer {

    constructor() {
        // INIT
        this.capturer = new CCapture({format: "webm", framerate: 60});
        this.isCapturing = false;
        this.DEBUG = false;
    }

    // Start capturing when the animation begins
    startCapture() {
        if (!this.isCapturing && !this.DEBUG) {
            this.capturer.start();
            this.isCapturing = true;
        }
    }

    captureFrame(renderer) {
        this.capturer.capture(renderer.domElement);
    }

    // Stop capturing and save the video
    stopCapture(cameraBasePositions) {
        if (this.isCapturing && !this.DEBUG) {
            this.capturer.stop();
            this.capturer.save();

            // Download cameraBasePositions as JSON
            const blob = new Blob([JSON.stringify(cameraBasePositions, null, 2)], {type: "application/json"});
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = "cameraBasePositions.json";
            link.click();
            this.isCapturing = false;
        }
    }
}