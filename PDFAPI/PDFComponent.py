#!/usr/bin/env python
"""The FitComponent class for PDF refinement.

See the class documentation for more information.
"""

__id__ = "$Id$"


from RefinementAPI.FitComponent import FitComponent
from RefinementAPI.exceptiontools import isIterable, verifyType, isFloatable

class PDFComponent(FitComponent):
    """Class for specifying a single PDF fit componenet.

    PDFComponent is derived from RefinementAPI.FitComponent.FitComponent.

    Attributes
    _calculator --  PDFCalculator instance (default None).  This is set during
                    instantiation. 
    _data       --  PDFData instance (default None). This must be set with the
                    'setPattern' method.
    _model      --  MultiPhase instance (default MultiPhase()). Phases are added
                    to the MultiPhase with the 'addPhase' method. 
    _objective  --  Objective instance (default Chi2Objective). This is set
                    during instantiation, and is not required to be initialized
                    by the client.

    Calling 'refine' will verify the configuration and raise a
    FitConfigurationError if the configuration is incomplete.

    Common Exceptions
    TypeError is thrown when method arguments are of inappropriate type.
    ValueError when method argument does not fall within specified bounds.
    """

    def __init__(self):
        """Instantiate a PDFComponent.

        This sets the attributes detailed in the class documentation to their
        default values.
        """
        FitComponent.__init__(self)
        from phases import _MultiPhase
        self._model = _MultiPhase()
        from PDFCalculator import PDFCalculator
        self.setCalculator(PDFCalculator())

    def setCalculator(self, cal):
        """Set the calculator to be used by this component.

        This also sets _calculator of _objective.
        
        Arguments
        cal     --  An instance of PDFCalculator
        """
        from PDFCalculator import PDFCalculator
        verifyType(cal, PDFCalculator)
        FitComponent.setCalculator(self, cal)
        self._calculator._comp = self
        return

    def addPhase(self, phase):
        """Add a phase.

        Arguments
        phase   --  A Phase instance. See the Phase class for more information.

        Raises
        ValueError when attempting to add a phase that has already been added.
        """
        from phases import Phase
        verifyType(phase, Phase)
        # Error checking done in _MultiPhase
        self._model._comp = self
        self._model.addPhase(phase)
        return

    def getPhase(self, index):
        """Get a phase.
        
        This uses python list notation, so index -n returns the nth phase from
        the end.

        Arguments:
        index  --  index of phase (integer, starting at 0).

        Returns Phase instance

        Raises 
        IndexError if requesting a phase that does not exist
        """
        return self._model.getPhase(index)

    def getNumPhases(self):
        """Get the number of phases."""
        return self._model.getNumPhases()

    def verifyConfiguration(self):
        """Verify this object is ready to be used in a fit.
        
        Raises
        FitConfigurationError if _calculator is None.
        FitConfigurationError if _calculator is not configured properly.
        FitConfigurationError if _data is None.
        FitConfigurationError if _data is not configured properly.
        FitConfigurationError if _objective is not configured properly.
        FitConfigurationError if _model is not configured properly.
        """
        FitComponent.verifyConfiguration(self)
        self._model.verifyConfiguration()
        return

    def getSubRefinables(self):
        """Get a list of subordinate refinables."""
        return [self._data, self._model]

# End of PDFComponent


if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass
