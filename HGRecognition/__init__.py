import logging as log
from os import makedirs
from pathlib import Path

import cv2
import imutils

log.basicConfig(format='[%(levelname)s]-\t%(message)s', level=log.INFO)


class CreateDataset:
    bg = None
    count = 10

    def __init__(self, gesture_name: str, dst_dir: Path,
                 threshold: float = 0.5, img_count: int = 0, image_suffix: str = '',
                 process_img: bool = False, capture_device: int = 0):

        self.device = capture_device
        self.threshold = threshold
        self.gesture = gesture_name
        self.dst = dst_dir
        self.process = process_img
        self.suffix = image_suffix

        if img_count > 0:
            self.count = img_count

        if self.count > 0:
            makedirs(self.dst, exist_ok=True)
            self.dst /= self.gesture
            makedirs(self.dst, exist_ok=True)

    def __run_avg___(self, image, a_weight):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, a_weight)

    def __image_segment__(self, image):
        diff = cv2.absdiff(self.bg.astype("uint8"), image)
        threshold_img = cv2.threshold(diff,
                                      self.threshold,
                                      255,
                                      cv2.THRESH_BINARY)[1]

        (cnt, _) = cv2.findContours(threshold_img.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        if len(cnt) == 0:
            return
        else:
            segmented = max(cnt, key=cv2.contourArea)
            return threshold_img, segmented

    def createImages(self) -> bool:
        a_weight = 0.5
        camera = cv2.VideoCapture(0)
        top, right, bottom, left = 10, 350, 225, 590
        num_frames = 0
        image_num = 0

        start_recording = False

        log.info(f'Press \'s\' to start clicking images')
        log.info(f'Press \'q\' to quit clicking images')
        log.info(f'Please wait while initialize')
        log.info(f'Hold your device stable')

        while True:
            grabbed, frame = camera.read()
            if grabbed:
                frame = imutils.resize(frame, width=700)
                frame = cv2.flip(frame, 1)
                clone = frame.copy()
                roi = frame[top:bottom, right:left]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                if num_frames < 100:
                    self.__run_avg___(gray, a_weight)
                else:
                    hand = self.__image_segment__(gray)
                    if hand is not None:
                        (threshold_img, segmented) = hand
                        cv2.drawContours(
                            clone, [segmented + (right, top)], -1, (0, 0, 255))
                        if start_recording:
                            cv2.imwrite(str(self.dst / (self.suffix + str(image_num) + '.png')), threshold_img)
                            image_num += 1
                            log.info(f'Created image {image_num}')
                        cv2.imshow("Threshold Image", threshold_img)

                cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
                num_frames += 1
                cv2.imshow("Video Feed", clone)
                keypress = cv2.waitKey(1) & 0xFF
                if keypress == ord("q"):
                    log.warning(f'Process terminated by user')
                    break
                elif image_num == self.count:
                    log.info(f'Images created')
                    break

                if keypress == ord("s"):
                    start_recording = True

            else:
                break

        camera.release()
        cv2.destroyAllWindows()

        return True

    def __str__(self):
        return self.gesture

    def __repr__(self):
        return self.gesture

    def collect_details(self) -> dict:
        d = {
            'gesture': self.gesture,
            'saved_dir': self.dst,
            'num_images': self.count,
            'is_processed': self.process,
            'threshold_value': self.threshold,
            'capture_device': self.device
        }
        return d
