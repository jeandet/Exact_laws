from ... import logs

from ...mathematical_tools.derivation import cdiff


class IncGrid():
    """Classe contenant les informations nécessaires sur la grille finale (grille d'échelles)"""

    def __init__(self, original_grid, N, axis, coords, kind):
        self.spatial_grid = original_grid
        self.axis = axis
        self.N = N 
        self.coords = coords
        self.kind = 'logcyl_'+kind
    
    def describ(self):
        tmp = self.spatial_grid.describ().replace('\t','\t\t')
        message =  f"\n\t - original grid :{tmp}"
        message += f"\n\t - axis: {self.axis}"
        message += f"\n\t - N: {self.N}"
        message += f"\n\t - kind: {self.kind}"
        return message

    def check(self, name=''):
        """Display the principal parameter of the grid object"""
        message = self.describ()
        logs.getLogger(__name__).info(message)


