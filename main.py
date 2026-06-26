from PyQt6.QtGui import *
from PyQt6.QtCore import *
import cv2
from UI import UI
from ComputerVisionModules import computerVision
import zmq
import time
import sys

from PyQt6 import QtWidgets


# video instead of posing so i don't lose my mind
VIDEO_PATH = None

latest_frame = None  # Shared buffer for MJPEG


class Interface(UI.Ui_HumonoidRobotControl):
    def __init__(self):
        super().__init__()


    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))


    def start_video_feed(self, ui):
        self.CameraFeed = CameraFeed()
        self.CameraFeed.start()
        self.CameraFeed.ui = ui
        self.CameraFeed.ImageUpdate.connect(ui.ImageUpdateSlot)


    def CancelFeed(self):
        self.CameraFeed.stop()


    """def is_head_on(self):
        return self.Head.isChecked()


    def is_shoulders_on(self):
        return self.Shoulders.isChecked()


    def is_elbows_on(self):
        return self.Elbows.isChecked()


    def is_hands_on(self):
        return self.Hands.isChecked()"""


    def is_black_background_on(self):
        return self.BlackBackground.isChecked()


    def is_head_text_on(self):
        return self.HeadInfo.isChecked()


    def is_upper_body_text_on(self):
        return self.UpperBodyInfo.isChecked()


    def get_mode(self):
        if self.buttonCONTRASTING.isChecked():
            mode = "contrasting"
        elif self.buttonAMPLIFYING.isChecked():
            mode = "amplifying"
        elif self.buttonRMIRRORING.isChecked():
            mode = "reversed mirroring"
        else:
            mode = "mirroring"

        return mode


    def is_auto_checked(self):
        if self.buttonAUTOMATIC.isChecked():
            auto = 1
        else:
            auto = 0

        return auto


    def is_random_checked(self):
        return 1 if self.buttonRANDOM.isChecked() else 0


    def hide_modes(self):
        if self.HideModes.isChecked():
            #self.overlay.setGeometry(self.centralwidget.rect())
            self.overlay.setVisible(True)
            self.overlay.raise_()
            #print("Overlay ON")
        else:
            self.overlay.setVisible(False)
            #print("Overlay OFF")


    def get_slider_value(self):
        slider_value = self.sliderStrength.value()
        return slider_value


    def setup_slider(self):
        """Set up the slider and disable it unless 'AMPLIFYING' is selected"""
        self.sliderStrength.setMinimum(1)
        self.sliderStrength.setMaximum(10)
        self.sliderStrength.setValue(5)  # Default to 5
        self.sliderStrength.setEnabled(False)  # Disabled initially
        self.labelStrengthValue.setText(str(self.sliderStrength.value()))

        self.sliderStrength.valueChanged.connect(self.update_strength_label)

        self.buttonAMPLIFYING.toggled.connect(self.toggle_slider)
        self.overlay.setVisible(False)


    def toggle_slider(self, checked):
        if checked:
            self.sliderStrength.setEnabled(True)
        else:
            self.sliderStrength.setEnabled(False)
            self.sliderStrength.setValue(5)  # Reset to default when disabled
        self.labelStrengthValue.setText(str(self.sliderStrength.value()))


    def update_strength_label(self):
        """Update label with slider value"""
        self.labelStrengthValue.setText(str(self.sliderStrength.value()))


class CameraFeed(QThread):
    ImageUpdate = pyqtSignal(QImage)
    ui = None

    # Event set by a background key-listener thread to skip the countdown
    _skip_event = None

    def _run_countdown(self, Capture, duration=10):
        """
        The countdown lasts 10 seconds.  Press any key in the
        terminal to skip the remaining time immediately.
        """
        import threading, select

        # Background thread that waits for a key press in the terminal
        self._skip_event = threading.Event()
        def _wait_for_key():
            try:
                # Works on Unix: wait until stdin is readable
                import tty, termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                
                try:
                    tty.setcbreak(fd)
                    sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            
            except Exception:
                # Fallback (Windows / no terminal): just wait for input()
                try:
                    input()
                
                except EOFError:
                    return
            self._skip_event.set()

        key_thread = threading.Thread(target=_wait_for_key, daemon=True)
        key_thread.start()

        start_time = time.time()
        print(f"[countdown] Get into position! {duration}s countdown "
              "(press any key in terminal to skip)")

        while self.ThreadActive:
            elapsed = time.time() - start_time
            remaining = duration - elapsed

            if remaining <= 0 or self._skip_event.is_set():
                if self._skip_event.is_set():
                    print("[countdown] Skipped by key press.")
                else:
                    print("[countdown] Countdown finished.")
                break

            ret, frame = Capture.read()
            if not ret:
                continue

            # flip camera during countdown so i don't clash on my bathroom door while i get in position
            preview = cv2.flip(frame, 1)

            # Draw countdown number centred on the frame
            display_sec = int(remaining) + 1
            text = str(display_sec)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 3
            thickness = 6
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cx = (preview.shape[1] - tw) // 2
            cy = (preview.shape[0] + th) // 2
            # Black outline for readability
            cv2.putText(preview, text, (cx, cy), font, scale,
                        (0, 0, 0), thickness + 4, cv2.LINE_AA)
            # White number
            cv2.putText(preview, text, (cx, cy), font, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

            # Small instruction text at the bottom
            hint = "Press any key to skip"
            cv2.putText(preview, hint,
                        (10, preview.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Convert BGR → RGB for Qt display
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                            QImage.Format.Format_RGB888)
            pic = qt_img.scaled(800, 640, Qt.AspectRatioMode.KeepAspectRatio)
            self.ImageUpdate.emit(pic)


    def run(self):
        self.ThreadActive = True

        using_video = VIDEO_PATH is not None

        if using_video:
            Capture = cv2.VideoCapture(VIDEO_PATH)
            if not Capture.isOpened():
                print(f"[video] ERROR: cannot open '{VIDEO_PATH}'")
                return
            
            video_fps = Capture.get(cv2.CAP_PROP_FPS) or 30.0
            frame_delay = 1.0 / video_fps
            print(f"[video] Playing '{VIDEO_PATH}' at {video_fps:.1f} FPS")
        
        else:
            Capture = cv2.VideoCapture(0)
            Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            Capture.set(cv2.CAP_PROP_FPS, 15)
            frame_delay = 0  # no throttle for live camera

        # Countdown — only for live camera (no need to get into position for video)
        if not using_video:
            self._run_countdown(Capture, duration=10)

        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.bind("tcp://*:5555")

        while self.ThreadActive:

            interface_inputs = self.get_inputs_from_interface()
            ret, frame = Capture.read()

            # Loop video when it reaches the end
            if not ret and using_video:
                Capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = Capture.read()

            if ret:
                Image = computerVision.run_computer_vision(frame, interface_inputs, socket)

                global latest_frame #MPJEPG
                latest_frame = Image.copy() #also just to test

                ConvertToQtFormat = QImage(Image.data , Image.shape[1], Image.shape[0] , QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(800, 640, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)


    def stop(self):
        self.ThreadActive = False
        self.quit()


    def get_inputs_from_interface(self):
        interface_input = {
            "Head": True, #ui.is_head_on(),
            "Shoulders": True, #ui.is_shoulders_on(),
            "Elbows": True, #ui.is_elbows_on(),
            "Hands": True, #ui.is_hands_on(),
            "BlackBackground": ui.is_black_background_on(),
            "HeadText": ui.is_head_text_on(),
            "UpperBodyText": ui.is_upper_body_text_on(),
            "mode": ui.get_mode(),
            "aSliderValue": ui.get_slider_value(),
            "auto_mode": ui.is_auto_checked(),
            "random_mode": ui.is_random_checked()
        }

        return interface_input


if __name__ == "__main__":

    from flask import Flask, Response, stream_with_context
    import threading

    app = Flask(__name__)

    import time


    def generate_mjpeg():
        global latest_frame
        while True:
            if latest_frame is not None:
                rgb_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
                ret, jpeg = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 30])

                if ret:
                    frame = jpeg.tobytes()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + f"{len(frame)}".encode() + b"\r\n\r\n" +
                           frame + b"\r\n")
            time.sleep(0.03)


    @app.route('/video_feed')
    def video_feed():
        return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


    from gevent.pywsgi import WSGIServer


    def start_mjpeg_server():
        #app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        http_server = WSGIServer(('0.0.0.0', 5000), app)
        http_server.serve_forever()


    # Start MJPEG server in parallel with Qt app
    server_thread = threading.Thread(target=start_mjpeg_server)
    server_thread.daemon = True
    server_thread.start()

    # Start the OSC bridge that forwards joint angles to Unity (port 9000)
    import osc_bridge
    osc_bridge.start_in_background()

    import sys
    app = QtWidgets.QApplication(sys.argv)
    HumonoidRobotControl = QtWidgets.QMainWindow()
    ui = Interface()
    ui.setupUi(HumonoidRobotControl)
    ui.HideModes.toggled.connect(ui.hide_modes)
    ui.setup_slider() #FIX AFTER
    ui.hide_modes()

    ui.start_video_feed(ui)
    HumonoidRobotControl.show()
    sys.exit(app.exec())