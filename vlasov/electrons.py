class Electrons:
    def __init__(self, Te, ne=1, charge=1):
        # Electron temperature
        self.Te = Te
        # Electron charge (a positive number)
        self.charge = charge
        # Electron number density
        self.ne = ne
        # Derivative of electron pressure wrt to electron number density
        self.dpedne = Te
        # Charge density
        self.ene = charge*ne
