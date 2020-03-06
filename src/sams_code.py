from diffpy.structure import loadStructure

# load NaCl CIF as diffpy.structure
nacl_stru = loadStructure('NaCl.cif')

# create BondCalculator object
from diffpy.srreal.bondcalculator import BondCalculator
bc = BondCalculator(rmax=3)

#from diffpy.srreal.bondcalculator import BondCalculator # not working but moving on, lets say there is a bondcalc

bond_lengths =10*bc(nacl_stru) # r max = 3 from above


## Coelho Calc basic outline !!!

# Step 1: for each length r, create the "bins" --> intensity
    # create histograph type thing -- different r have a number of times that r appears
        # is it just the number of times r repeats that gives intensity? ****QUESTION
    # split intensity between points just before and after r
        # how do we pick delta r? ****QUESTION

# getting intensities
intensities = zeros(30)
sp = zeros(30)
for b in bond_lengths:
    intensities[b] += 1
    sp[b] +=1

# rebinning --> creating partial stick patterns for rb and ra (points just
# before and just after r)
delta = 2 # i randomly picked this number
r = 0
for i in intensities:
    ra = r+delta
    rb = r-delta
    sppra = i*(ra-r)/(ra-rb)
    spprb = i*(r-rb)/(ra-rb)
    sp[ra] = sppra
    sp[rb] =spprb
    r += 1

# Step 2: calculate number of Gaussians needed for each peak (Ng)
    # based on fp and fa (.6 <= fp/fa <= 1)
        # how do we know fa? isn't knowing fa necessary to getting fp based on the ratio? ****QUESTION
        # if we do know fa, wouldn't we already have the final Gaussian for that r? ****QUESTION
        # if we don't know fa, how do we pick fp? ****QUESTION
        # i am guessing other factors are needed to approx fa, and then get fp based on this approx, maybe?


# Step 3: insert Ng sticks for each bond length
    # no longer inserted on atomic pair basis
    # how do we calculate side sticks? is it using the same formula above? ****QUESTION
    # overall section 4. Laying down sticks --> need clarification



# Step 4: convolution of stick pattern w gauss fp to get Gp
if 6*Np*ln(Np) < 3*Np*fp/delta: # Np is number of data points
    # FDFT used
elif fp/delta > 100:
    # approximation for Gauss(fp) used
else:
    # direct numerical convolution

# Step 5: Gp to G ???