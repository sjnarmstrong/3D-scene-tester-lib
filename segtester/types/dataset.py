from segtester.types.scene import Scene
from typing import List


class Dataset:
    def __init__(self):
        self.scenes: List[Scene] = []

    def get_scene_with_id(self, id):
        for scene in self.scenes:
            if scene.id == id:
                return scene
        return None
