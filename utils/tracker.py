# utils/tracker.py
# Minimal centroid tracker used by CarFlow project.
# Author: <Your Name> · 2025

import math

class Tracker:
    """
    Simple centroid‑based tracker.

    How it works
    ------------
    1.  The caller passes a list of bounding boxes in **(x, y, w, h)** format.
        Each box is assumed to be a detection for the current video frame.
    2.  For every detection we compute its centroid (cx, cy) and search for the
        previously stored centroid that is within ``max_distance`` pixels.
        • If a match is found, we keep the same object ID.  
        • Otherwise a **new object ID** is assigned.
    3.  The tracker returns a list shaped like
        ``[[x, y, w, h, id], …]`` so downstream code can draw the box and
        display a consistent object ID.

    This implementation purposefully stays minimal:
      * **No track termination** – IDs live forever once created.  
      * **No motion prediction / Kalman filter** – a pure distance check.  
      * **No appearance features** – suitable for small‑scale demos.

    Parameters
    ----------
    max_distance : int, default=70
        Maximum allowable Euclidean distance (in pixels) between the centroid of
        a detection and an existing track in order to be considered the same
        object.
    """
    def __init__(self, max_distance=70):
        self.center_points = {}          # Maps object ID → last known centroid (cx, cy)
        self.id_count      = 0
        self.max_distance  = max_distance

    def update(self, rectangles):
        objects_bbs_ids=[]
        # Iterate over every detection in the current frame
        for (x,y,w,h) in rectangles:
            cx,cy = x+w//2, y+h//2
            # Attempt to match this detection to an existing track
            matched_id=None
            for oid,pt in self.center_points.items():
                if math.hypot(cx-pt[0], cy-pt[1]) < self.max_distance:
                    matched_id=oid; break
            if matched_id is None:              # No matching track found → create a NEW object ID
                matched_id=self.id_count
                self.id_count+=1
            self.center_points[matched_id]=(cx,cy)
            objects_bbs_ids.append([x,y,w,h,matched_id])

        # Note: we do not delete stale IDs in this minimal version.
        return objects_bbs_ids
