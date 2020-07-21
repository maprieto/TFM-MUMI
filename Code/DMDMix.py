"""
Reads the simulated data from a xdmf/h5 file and applies DMD to it to get a prediction
"""

from paraview.simple import *
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pydmd import DMD
from pydmd import HODMD
from pydmd import MrDMD

# Files to combine at DMD
fileName1 = '_DMDMixTest_sr-f_harm__IC_6_N_400_dt_8.33333e-06_c_300'
fileName2 = '_DMDMixTest_sr-f_harm__IC_6_N_400_dt_7.28863e-06_c_343' # Triangular

rank = 5

period = 1/6000
period = 400/(23*6000)

# tCutoff = 2/32 * period

# ----------------------------------------------------------------------------------------------------------------------

print('Reading data 1...')

# Open files to read them
fileXDMF = XDMFReader(FileNames=['results/'+fileName1+'.xdmf'])
fileH5 = h5py.File('results/'+fileName1+'.h5', 'r')

# Read the time and geometry vectors
xVector1 = np.array(fileH5['Mesh']['0']['mesh']['geometry'].value[:,0])
tVector1 = np.array(fileXDMF.TimestepValues)

# Create the space-time grids
xGrid1, tGrid1 = np.meshgrid(xVector1, tVector1)

# Get sizes
xSize1 = len(xVector1)
tSize1 = len(tVector1)

# Initialize data arrays
uGrid1 = np.empty([tSize1, xSize1])
vGrid1 = np.empty([tSize1, xSize1])
aGrid1 = np.empty([tSize1, xSize1])
uexGrid1 = np.empty([tSize1, xSize1])

# Read data arrays
for sKey in fileH5['VisualisationVector'].keys():
    if (int(sKey) % 4) == 0:
        uGrid1[int(sKey) // 4,:] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 1:
        vGrid1[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 2:
        aGrid1[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 3:
        uexGrid1[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])

dataGrid1 = uGrid1

# ----------------------------------------------------------------------------------------------------------------------

print('Reading data 2...')

# Open files to read them
fileXDMF = XDMFReader(FileNames=['results/'+fileName2+'.xdmf'])
fileH5 = h5py.File('results/'+fileName2+'.h5', 'r')

# Read the time and geometry vectors
xVector2 = np.array(fileH5['Mesh']['0']['mesh']['geometry'].value[:,0])
tVector2 = np.array(fileXDMF.TimestepValues)

# Create the space-time grids
xGrid2, tGrid2 = np.meshgrid(xVector2, tVector2)

# Get sizes
xSize2 = len(xVector2)
tSize2 = len(tVector2)

# Initialize data arrays
uGrid2 = np.empty([tSize2, xSize2])
vGrid2 = np.empty([tSize2, xSize2])
aGrid2 = np.empty([tSize2, xSize2])
uexGrid2 = np.empty([tSize2, xSize2])

# Read data arrays
for sKey in fileH5['VisualisationVector'].keys():
    if (int(sKey) % 4) == 0:
        uGrid2[int(sKey) // 4,:] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 1:
        vGrid2[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 2:
        aGrid2[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 3:
        uexGrid2[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])

dataGrid2 = uGrid2

# ----------------------------------------------------------------------------------------------------------------------

print('Splitting the data in 2...')

tCutoff = period

# First part

# Index of first element in tVector greater than tCutoff in tVector
idxCutoff = next(x for x, val in enumerate(tVector2) if val > tCutoff)
# Cutoff time vector
tVectorHalf1 = tVector1[0:idxCutoff]
print('\ntVector 1 Cut Off length: %i\n' %len(tVectorHalf1))

# Second part

# Index of first element in tVector greater than tCutoff in tVector
idxCutoff = next(x for x, val in enumerate(tVector2) if val > tCutoff)
# Cutoff time vector
tVectorHalf2 = tVector2[idxCutoff:]
print('\ntVector 2 Cut Off length: %i\n' %len(tVectorHalf2))


# Create the space-time grids
xGridHalf1, tGridHalf1 = np.meshgrid(xVector1, tVectorHalf1)
xGridHalf2, tGridHalf2 = np.meshgrid(xVector2, tVectorHalf2)

# Cut off data grids
uGridHalf1 = uGrid1[0:idxCutoff, :]
vGridHalf1 = vGrid1[0:idxCutoff, :]
aGridHalf1 = aGrid1[0:idxCutoff, :]
uexGridHalf1 = uexGrid1[0:idxCutoff, :]

# Cut off data grids
uGridHalf2 = uGrid2[idxCutoff:, :]
vGridHalf2 = vGrid2[idxCutoff:, :]
aGridHalf2 = aGrid2[idxCutoff:, :]
uexGridHalf2 = uexGrid2[idxCutoff:, :]


dataGridHalf1 = uGridHalf1
dataGridHalf2 = uGridHalf2

# Joined data
tVector = np.hstack((tVectorHalf1, tVectorHalf2))
xGrid, tGrid = np.meshgrid(xVector1, tVector)
uGrid = np.vstack((uGridHalf1, uGridHalf2))
vGrid = np.vstack((vGridHalf1, vGridHalf2))
aGrid = np.vstack((aGridHalf1, aGridHalf2))
uexGrid = np.vstack((uexGridHalf1, uexGridHalf2))
dataGrid = np.vstack((dataGridHalf1, dataGridHalf2))


# Compute DMDs
d = 5
print('Computing DMD 1...')
hodmd1 = HODMD(svd_rank=rank, opt=True, d=d)
hodmd1.fit(dataGridHalf1.T)

print('Computing DMD 2...')
hodmd2 = HODMD(svd_rank=rank, opt=True, d=d)
hodmd2.fit(dataGridHalf2.T)

print('Computing mixed DMD...')
hodmd = HODMD(svd_rank=rank, opt=True, d=d)
hodmd.fit(dataGrid.T)

# Adjust times for partials
hodmd1.original_time['dt'] = hodmd1.dmd_time['dt'] = tVectorHalf1[1] - tVectorHalf1[0]
hodmd1.original_time['t0'] = hodmd1.dmd_time['t0'] = tVectorHalf1[0]
hodmd1.original_time['tend'] = hodmd1.dmd_time['tend'] = tVectorHalf1[-1]
hodmd1.dmd_time['tend'] = tVector1[-1]

hodmd2.original_time['dt'] = hodmd2.dmd_time['dt'] = tVectorHalf2[1] - tVectorHalf2[0]
hodmd2.original_time['t0'] = hodmd2.dmd_time['t0'] = tVectorHalf2[0]
hodmd2.original_time['tend'] = hodmd2.dmd_time['tend'] = tVectorHalf2[-1]
hodmd2.dmd_time['t0'] = tVector2[0]

print('Drawing...')

# Partial reconstruction 1
# fig = plt.figure(figsize=(20, 6))
fig = plt.figure(figsize=(8,10))
fig.subplots_adjust(wspace=0.5,hspace=1,top=0.9, bottom=0.1, right=0.95, left=0.05)


plt.subplot(331)
plt.pcolor(xGridHalf1, tGridHalf1, dataGridHalf1)
plt.ylim(0, tVector1[-1])
cbar = plt.colorbar()
plt.title('Given data 1', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('figures/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

plt.subplot(332)
plt.pcolor(xGrid1, tGrid1, hodmd1.reconstructed_data.T.real)
cbar = plt.colorbar()
plt.title('DMD approximation 1', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('figures/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

plt.subplot(333)
plt.pcolor(xGrid1, tGrid1, (dataGrid1 - hodmd1.reconstructed_data.T).real)
cbar = plt.colorbar()
plt.title('Absolute error 1', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))


# Partial reconstruction 2
plt.subplot(334)
plt.pcolor(xGridHalf2, tGridHalf2, dataGridHalf2)
plt.ylim(0, tVector2[-1])
cbar = plt.colorbar()
plt.title('Given data 2', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

plt.subplot(335)
plt.pcolor(xGrid2, tGrid2, hodmd2.reconstructed_data.T.real)
cbar = plt.colorbar()
plt.title('DMD approximation 2', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

plt.subplot(336)
plt.pcolor(xGrid2, tGrid2, (dataGrid2 - hodmd2.reconstructed_data.T).real)
cbar = plt.colorbar()
plt.title('Absolute error 2', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

# Mixed reconstruction
plt.subplot(337)
plt.pcolor(xGrid, tGrid, dataGrid)
plt.ylim(0, tVector[-1])
cbar = plt.colorbar()
plt.title('Given mixed data', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

plt.subplot(338)
plt.pcolor(xGrid, tGrid, hodmd.reconstructed_data.T.real)
cbar = plt.colorbar()
plt.title('DMD mixed approximation', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))

plt.subplot(339)
plt.pcolor(xGrid, tGrid, (dataGrid1 - hodmd.reconstructed_data.T).real)
cbar = plt.colorbar()
plt.title('Absolute error with 1', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()


fig.savefig('DMDResults/DMDMixingHalf_'+fileName1+'+'+fileName2+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))
# plt.show()

print('Done')