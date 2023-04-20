class Feature():
    def __init__(self, name, x=None, y=None):
        self.name = name
        self.x = x if x is not None else []
        self.y = y if y is not None else []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def __str__(self):
        return self.name.upper()
