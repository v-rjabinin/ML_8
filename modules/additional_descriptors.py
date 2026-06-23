class PositiveInt:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        try:
            value = int(value)
        except ValueError:
            raise TypeError(f"'{self.name}' must be an integer")

        if value < 1:
            raise ValueError(f"'{self.name}' must be positive")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)


class NonNegativeInt:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        try:
            value = int(value)
        except ValueError:
            raise TypeError(f"'{self.name}' must be an integer")

        if value < 0:
            raise ValueError(f"'{self.name}' must be non-negative")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)


class PositiveFloat:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        try:
            value = float(value)
        except ValueError:
            raise TypeError(f"'{self.name}' must be a float")

        if not value > 0:
            raise ValueError(f"'{self.name}' must be positive")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)


class Ratio:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        try:
            value = float(value)
        except ValueError:
            raise TypeError(f"'{self.name}' must be a float")

        if not (0 < value < 1):
            raise ValueError(f"'{self.name}' must be in (0.0; 1.0)")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)