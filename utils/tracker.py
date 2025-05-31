import math

class Tracker:
    """
    簡易 Centroid Tracker
    ---------------------------------
    • update(rectangles)    rectangles = [[x,y,w,h], ...]
    • 回傳 [[x,y,w,h,id], ...]
    """
    def __init__(self, max_distance=70):
        self.center_points = {}          # id -> (cx,cy)
        self.id_count      = 0
        self.max_distance  = max_distance

    def update(self, rectangles):
        objects_bbs_ids=[]
        # 每個偵測框
        for (x,y,w,h) in rectangles:
            cx,cy = x+w//2, y+h//2
            matched_id=None
            for oid,pt in self.center_points.items():
                if math.hypot(cx-pt[0], cy-pt[1]) < self.max_distance:
                    matched_id=oid; break
            if matched_id is None:              # 新目標
                matched_id=self.id_count
                self.id_count+=1
            self.center_points[matched_id]=(cx,cy)
            objects_bbs_ids.append([x,y,w,h,matched_id])

        # 移除久未更新 id（此版不做，保持簡單）
        return objects_bbs_ids
