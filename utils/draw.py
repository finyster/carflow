# utils/draw.py
import cv2

class Painter:
    def __init__(self, routes):
        """routes: config.yml 裡的 routes（多組 entry/exit 線）"""
        self.routes = routes

    def draw_lines(self, frame):
        """畫出所有 entry/exit 線，標上 route 名稱"""
        for route in self.routes:
            # Entry 線（綠色）
            p1, p2 = tuple(route["entry"]["p1"]), tuple(route["entry"]["p2"])
            cv2.line(frame, p1, p2, (0, 255, 0), 3)
            cv2.putText(frame, f'{route["name"]}_entry', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # Exit 線（紅色）
            p3, p4 = tuple(route["exit"]["p1"]), tuple(route["exit"]["p2"])
            cv2.line(frame, p3, p4, (0, 0, 255), 3)
            cv2.putText(frame, f'{route["name"]}_exit', p3, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    def draw_box(self, frame, bbox, track_id, class_name):
        """畫出追蹤框和ID、類別"""
        x1, y1, x2, y2 = bbox
        color = (0,255,255) if class_name == "car" else (255,0,0) if class_name == "bus" else (0,0,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{class_name} ID:{track_id}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_counts(self, frame, counter):
        """在左上角顯示各方向各車種計數"""
        x0, y0 = 20, 40
        for idx, (route, cnts) in enumerate(counter.items()):
            txt = f"{route}: " + "  ".join([f"{k}={v}" for k, v in cnts.items()])
            cv2.putText(frame, txt, (x0, y0 + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    def put_text(self, frame, text, pos, color=(255,255,255)):
        """在指定位置放文字"""
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def check_cross(self, pt, history=None):
        """
        (進階) 判斷點 pt 是否有經過某條 entry/exit 線。
        可依需求改為回傳哪一組線的名稱。
        """
        pass  # 視實際判線方法設計，可用在進階需求
