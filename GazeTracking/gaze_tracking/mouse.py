import autopy

class Mouse(object):
    """
    This class tracks the cursor's current position.
    """

    def __init__(self):
        self.coordx = None
        self.coordy = None

    def cursor_position(self):
        x, y = autopy.mouse.location()
        return (x, y)
