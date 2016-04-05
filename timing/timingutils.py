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


    def __str__(self):
        rv = "ABC = %g %g %g  tbest = %g  nbest = %g" % (
                self.A, self.B, self.C, self.tbest, self.nbest)
        return rv

# class fitparalleltimes

if __name__ == '__main__':
    times1 = numpy.array([
        7.3867149353, 3.98858690262, 3.24399209023, 3.07190990448])
    times2 = numpy.array([
        8.0034828186, 3.52259397507, 2.41263222694, 1.81602692604,
        1.63822293282, 1.53632307053, 1.46815896034, 1.39612221718])
    print fitparalleltimes(times1)
    print fitparalleltimes(times2)
