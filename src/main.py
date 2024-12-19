import cv2
import numpy as np
import math


class OpticalFlow:

    def __init__(self, alpha, beta, x_pixels, y_pixels, fps):

        self.fps = fps
        self.alpha = alpha * math.pi / 180
        self.beta = beta
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels

    def get_pixel_value(self, high):

        # Вычисление дистанции отображаемой одним пикселем
        distance = high * math.tan(self.alpha) * 2
        pixel_value = distance / self.x_pixels
        return pixel_value

    def _calculate_optical_flow_farneback(self, prev_frame, curr_frame):

        # Преобразуем кадры в оттенки серого
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Вычисляем оптический поток методом Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        return flow

    def run(self, high):

        # Загрузка видеофайла
        cap = cv2.VideoCapture("video.mp4")
        visual = []
        # Получение первого кадра
        ret, prev_frame = cap.read()

        while True:
            # Получение текущего кадра
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Вычисление оптического потока
            flow = self._calculate_optical_flow_farneback(prev_frame, curr_frame)

            # Визуализация потока
            y_sum = 0
            x_sum = 0
            for y in range(0, flow.shape[0], 10):
                for x in range(0, flow.shape[1], 10):
                    y_sum += flow[y, x][1]
                    x_sum += flow[y, x][0]
                    fx, fy = flow[y, x]
                    cv2.arrowedLine(curr_frame, (x, y), (x + int(fx), y + int(fy)), (0, 255, 0), 1)

            # Перобразование потока в скорость
            pixel_value = self.get_pixel_value(high)
            y_vec_pix = y_sum / flow.shape[0]
            x_vec_pix = x_sum / flow.shape[1]
            x_vec = x_vec_pix * pixel_value
            y_vec = y_vec_pix * pixel_value
            speed_vect = math.sqrt(x_vec ** 2 + y_vec ** 2) * self.fps

            # Скорость в км/ч
            print(3600 * speed_vect / 1000)

            # Обновление предыдущего кадра
            visual.append(curr_frame)
            prev_frame = curr_frame

        # Закрытие окна и освобождение ресурсов
        cv2.destroyAllWindows()
        cap.release()


flow = OpticalFlow(30, 30, 762, 432, 30)
flow.run(50)

