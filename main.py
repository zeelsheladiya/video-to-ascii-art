
import cv2
import numpy as np
from numba import jit


@jit(nopython=True)
def to_ascii_art(frame, images, box_height=12, box_width=16):
    height, width = frame.shape
    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = frame[i:i + box_height, j:j + box_width]
            best_match = np.inf
            best_match_index = 0
            for k in range(1, images.shape[0]):
                total_sum = np.sum(np.absolute(np.subtract(roi, images[k])))
                if total_sum < best_match:
                    best_match = total_sum
                    best_match_index = k
            roi[:, :] = images[best_match_index]
    return frame


def generate_ascii_letters():
    images = []
    # letters = "# $%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    letters = " zt "
    # letters = " \\ '(),-./:;[]_`{|}~"
    for letter in letters:
        img = np.zeros((12, 16), np.uint8)
        img = cv2.putText(img, letter, (0, 11),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        images.append(img)
    return np.stack(images)


def startMainFunc(Vidchannel=0):

    vid1 = cv2.VideoCapture(Vidchannel)
    vid1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    images = generate_ascii_letters()

    while(True):

        _, frame = vid1.read()
        frame = cv2.flip(frame, 1)
        # frame = cv2.resize(frame, (600, 400))
        gb = cv2.GaussianBlur(frame, (5, 5), 0)
        can = cv2.Canny(gb, 127, 31)
        # can = cv2.resize(can, (600, 500))
        can2 = cv2.Canny(gb, 127, 31)
        # can2 = cv2.resize(can2, (600, 500))
        ascii_art = to_ascii_art(can, images)
        cv2.imshow('ASCII ART', ascii_art)
        cv2.imshow('Canny edge detection', can2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    vid1.release()

    cv2.destroyAllWindows()


def Main():
    startMainFunc()


if __name__ == "__main__":
    Main()
