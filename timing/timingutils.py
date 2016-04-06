from __future__ import print_function
import numpy
import numpy.linalg

class fitparalleltimes(object):

    def __init__(self, t, n=None):
        '''Fit run times from parallel calculations.

        t    -- array of evaluation times.
        n    -- number of parallel jobs for each time.  Assume
                [1, 2, ..., len(tcpar)] when not specified.
        '''
        self.t = numpy.asarray(t)
        self.n = numpy.arange(len(self.t)) + 1
        if n is not None:
            self.n = numpy.asarray(n)
        N = len(self.t)
        mx = numpy.ones([N, 3], dtype=float)
        mx[:, 1] = self.n
        mx[:, 2] = 1.0 / self.n
        abc = numpy.linalg.lstsq(mx, self.t)[0]
        self.A = abc[0]
        self.B = abc[1]
        self.C = abc[2]
        self.nbest = numpy.sqrt(self.C / self.B)
        self.tbest = self.tsim(self.nbest)
        return


    def tsim(self, ncpu):
        '''Estimate parallel evaluation time for ncpu parallel jobs.
        '''
        rv = self.A + self.B * ncpu + self.C / ncpu
        return rv


    def plot(self):
        from matplotlib.pyplot import plot
        nx = numpy.linspace(0.9, self.n.max() + 0.5)
        rv = plot(self.n, self.t, 'bo',
                  nx, self.tsim(nx), 'g--',
                  [self.nbest], [self.tbest], '*', markersize=6)
        hcircle = rv[0]
        hcircle.set_markerfacecolor('none')
        hcircle.set_markeredgecolor('blue')
        hstar = rv[-1]
        hstar.set_markersize(12)
        hstar.set_markerfacecolor('none')
        hstar.set_markeredgecolor('red')
        return rv


    def __str__(self):
        rv = "ABC = %g %g %g  tbest = %g  nbest = %g" % (
                self.A, self.B, self.C, self.tbest, self.nbest)
        return rv

# class fitparalleltimes

if __name__ == '__main__':
    fpt1 = fitparalleltimes([
        7.3867149353, 3.98858690262, 3.24399209023, 3.07190990448])
    print("fpt1:", fpt1)
    fpt2 = fitparalleltimes([
        8.0034828186, 3.52259397507, 2.41263222694, 1.81602692604,
        1.63822293282, 1.53632307053, 1.46815896034, 1.39612221718])
    print("fpt2:", fpt2)
