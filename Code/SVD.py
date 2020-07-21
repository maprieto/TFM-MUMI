"""
Reads the simulated data from a xdmf/h5 file and applies DMD to it to get a prediction
"""

from paraview.simple import *
import numpy as np
import h5py
import matplotlib.pyplot as plt

from numpy import array
from sklearn.decomposition import TruncatedSVD

# File to apply DMD to
fileName = 'st-fp_IC_6_N_200_dt_1.46e-05_phi_0.5_gam_1.4_sig_100'
fileName = 'sr-fp_IC_6_N_200_dt_1.46e-05_phi_0.5_gam_1.4_sig_100'
# fileName = 'rr-f_IC_4_N_500_dt_5.8309e-06'
# fileName = '_prev_rt-fp_IC_4_N_500_dt_5.8309e-06_phi_0.5_gam_1.4_sig_100'
# fileName = 'rr-f_IC_3_N_500_dt_5.8309e-06' # Triangular
# fileName = 'rr-f_IC_4_N_500_dt_5.8309e-06' # Meseta
# fileName = '_asymetry_st-fp-UmnovaLow_IC_7_N_400_dt_7.28863e-06_phi_0.5_gam_1.4_sig_100' # Mexican Hat
# fileName = '_last_st-fp-UmnovaLow-P-N_N_50_phi_0.36_gam_1.4_sig_27888' # Umnova Low

# Mesh Test
# fileName = 'sr-f_harm__IC_6_N_50_dt_5.8309e-05'
# fileName = 'sr-f_harm__IC_6_N_100_dt_2.91545e-05'
# fileName = 'sr-f_harm__IC_6_N_200_dt_1.45773e-05'
# fileName = 'sr-f_harm__IC_6_N_400_dt_7.28863e-06'

rank = 50

period = 1 / (3 * np.pi * 343)
period = 0.00075

# tCutoff = 2/32 * period

print('Reading data...')

# Open files to read them
fileXDMF = XDMFReader(FileNames=['results/'+fileName+'.xdmf'])
fileH5 = h5py.File('results/'+fileName+'.h5', 'r')

# Read the time and geometry vectors
xVector = np.array(fileH5['Mesh']['0']['mesh']['geometry'].value[:,0])
tVector = np.array(fileXDMF.TimestepValues)

# Create the space-time grids
xGrid, tGrid = np.meshgrid(xVector, tVector)

# Get sizes
xSize = len(xVector)
tSize = len(tVector)

# Initialize data arrays
uGrid = np.empty([tSize, xSize])
vGrid = np.empty([tSize, xSize])
aGrid = np.empty([tSize, xSize])
uexGrid = np.empty([tSize, xSize])

# Read data arrays
for sKey in fileH5['VisualisationVector'].keys():
    if (int(sKey) % 4) == 0:
        uGrid[int(sKey) // 4,:] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 1:
        vGrid[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 2:
        aGrid[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])
    elif (int(sKey) % 4) == 3:
        uexGrid[int(sKey) // 4, :] = (fileH5['VisualisationVector'][sKey].value[:,0])

dataGrid = uGrid


# print('Drawing...')

# # Plot read data
# fig = plt.figure(figsize=(17,6))
# plt.pcolor(xGrid, tGrid, uGrid)
# plt.title('Displacement')
# plt.colorbar()
# plt.show()
# # print('Done')

print('Computing SVD...')

# ----------------------------------------------------------------------------------------------------------------------

# Compute SVD on displacement

# svd
svd = TruncatedSVD(n_components=rank)
svd.fit(np.transpose(dataGrid))
svdReduced = svd.transform(np.transpose(dataGrid))
svdReconstr = np.transpose(svd.inverse_transform(svdReduced))

print(np.shape(svdReconstr))
print(svd.singular_values_)

# svd.fit(np.ones((100, xSize)))
#
# svdReconstr = np.transpose(svd.inverse_transform(svdReduced))
# print(np.shape(svdReconstr))

print('Drawing...')

# Show modes
fig = plt.figure(figsize=(4,3))
fig.subplots_adjust(top=0.8, bottom=0.2)
for r in range(0,rank):
    plt.plot(xVector, svdReduced[:,r])
    plt.title('Modes')
    plt.xlabel('$x$')
plt.show()

# Full reconstruction
fig = plt.figure(figsize=(8,3))
fig.subplots_adjust(wspace=0.5,top=0.8, bottom=0.2, right=0.95, left=0.05)

plt.subplot(131)
plt.pcolor(xGrid, tGrid, dataGrid)
cbar = plt.colorbar()
plt.title('True', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

plt.subplot(132)
plt.pcolor(xGrid, tGrid, svdReconstr)
cbar = plt.colorbar()
plt.title('SVD approximation', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

plt.subplot(133)
plt.pcolor(xGrid, tGrid, (dataGrid-svdReconstr).real)
cbar = plt.colorbar()
plt.title('Absolute error', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/SVD_'+fileName+'_Rank-%i.png'%(rank))
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# CO = [8, 16]
# CO = [1, 2, 4]
# CO = [4]
# CO = [6/6,5/6,4/6,3/6]
# CO = [7/6,8/6,9/6,10/6,11/6,12/6]
CO = [1]
for co in CO:

    # tCutoff = 1/co * period
    tCutoff = co * period
    # tCutoff = period
    # tCutoff = 0.003/4*3

    # Index of first element in tVector greater than tCutoff in tVector
    idxCutoff = next(x for x, val in enumerate(tVector) if val > tCutoff)

    # Cutoff time vector
    tVectorHalf = tVector[0:idxCutoff]
    print('\ntVector Cut Off length: %i\n' %len(tVectorHalf))

    # Create the space-time grids
    xGridHalf, tGridHalf = np.meshgrid(xVector, tVectorHalf)

    # Cut off data grids
    uGridHalf = uGrid[0:idxCutoff, :]
    vGridHalf = vGrid[0:idxCutoff, :]
    aGridHalf = aGrid[0:idxCutoff, :]
    uexGridHalf = uexGrid[0:idxCutoff, :]

    dataGridHalf = uGridHalf

    # ----------------------------------------------------------------------------------------------------------------------

    # blankData = 0*dataGrid
    blankData = np.ones(dataGrid.shape)
    svdBlank = TruncatedSVD(n_components=rank)
    svdBlank.fit(np.transpose(blankData))


    # Compute SVD on displacement
    svdPred = TruncatedSVD(n_components=rank)
    svdPred.fit(np.transpose(dataGridHalf))
    svdPredReduced = svdPred.transform(np.transpose(dataGridHalf))
    svdPredReconstr = np.transpose(svd.inverse_transform(svdPredReduced))
    svdPredReconstr = np.transpose(svdBlank.inverse_transform(svdPredReduced))

    print(np.shape(svdPredReconstr))
    print(svdPred.singular_values_)

    print('Drawing...')

    # Show modes
    fig = plt.figure(figsize=(17,6))
    for r in range(0,rank):
        plt.plot(xVector, svdPredReduced[:,r])
        plt.title('Modes')
    plt.show()

    # # Full reconstruction
    # fig = plt.figure(figsize=(17, 6))
    # plt.subplot(131)
    # plt.pcolor(xGridHalf, tGridHalf, dataGridHalf)
    # plt.ylim(0, tVector[-1])
    # plt.colorbar()
    # plt.title('True')
    #
    # plt.subplot(132)
    # plt.pcolor(xGrid, tGrid, svdPredReconstr)
    # plt.colorbar()
    # plt.title('SVD approximation')
    #
    # plt.subplot(133)
    # plt.pcolor(xGrid, tGrid, (dataGrid - svdPredReconstr).real)
    # plt.colorbar()
    # plt.title('Absolute error')







    # Full reconstruction
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2, right=0.95, left=0.05)

    plt.subplot(131)
    plt.pcolor(xGridHalf, tGridHalf, dataGridHalf)
    plt.ylim(0, tVector[-1])
    cbar = plt.colorbar()
    plt.title('True', pad=20)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplot(132)
    plt.pcolor(xGrid, tGrid, svdPredReconstr)
    cbar = plt.colorbar()
    plt.title('SVD approximation', pad=20)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplot(133)
    plt.pcolor(xGrid, tGrid, (dataGrid - svdPredReconstr).real)
    cbar = plt.colorbar()
    plt.title('Absolute error', pad=20)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    fig.savefig('DMDResults/SVDPred_' + fileName + '_Rank-%i.png' % (rank))
    plt.show()






    # fig.savefig('figures/SVDPred_'+fileName+'_CO-%f_Rank-%i.png'%(tCutoff,rank))
    plt.show()


print('Done')