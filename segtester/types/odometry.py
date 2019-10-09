import copy


class Trajectory:
    def __init__(self, trajectory=None):
        self.trajectory = trajectory

    def get_trajectory_copy(self):
        return copy.deepcopy(self.trajectory)
