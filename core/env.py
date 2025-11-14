
class Env():
    def __init__(self, file_path):
        self.map = []

    @property
    def robot_num():
        """ return robot number """
        return 1

    @property
    def map(self):
        """ grid size 
        @return: tuple of x, y"""
        return 1, 1

    

    def is_obstacle(self, index) -> bool:
        return self.map[index]
