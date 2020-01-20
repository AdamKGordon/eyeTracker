import autopy

class Mouse(object):
    """
    This class provides interaction with the cursor.
    """

    def __init__(self):
        self.coordx = None
        self.coordy = None

    """returns the cursor position in (x,y)"""

    def cursor_position(self):
        x, y = autopy.mouse.location()
        return (x, y)

    """"move the cursor to the given position"""

    def cursor_move(self, x, y):
        autopy.mouse.move(x, y)
