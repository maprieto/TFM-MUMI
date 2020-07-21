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

# File to apply DMD to
# fileName = 'st-fp_IC_6_N_200_dt_1.46e-05_phi_0.5_gam_1.4_sig_100'
# fileName = 'sr-fp_IC_6_N_200_dt_1.46e-05_phi_0.5_gam_1.4_sig_100'
# fileName = 'sr-fp_IC_6_N_400_dt_7.28863e-06_phi_0.5_gam_1.4_sig_100'
# fileName = 'rr-f_IC_4_N_500_dt_5.8309e-06'
# fileName = '_prev_rt-fp_IC_4_N_500_dt_5.8309e-06_phi_0.5_gam_1.4_sig_100'
# fileName = 'rr-f_IC_3_N_500_dt_5.8309e-06' # Triangular
# fileName = 'rr-f_IC_4_N_500_dt_5.8309e-06' # Meseta
fileName = '_asymetry_st-fp-UmnovaLow_IC_7_N_400_dt_7.28863e-06_phi_0.5_gam_1.4_sig_100' # Mexican Hat
# fileName = '_last_st-fp-UmnovaLow-P-N_N_50_phi_0.36_gam_1.4_sig_27888' # Umnova Low
# fileName = '_fade_sr-fp_IC_6_N_200_dt_1.45773e-05_phi_0.5_gam_1.4_sig_100' # Fade harmonic
fileName = '_fade_st-fp-UmnovaLow_IC_7_N_200_dt_1.45773e-05_phi_0.5_gam_1.4_sig_10000'

# Mesh Test
# fileName = 'sr-f_harm__IC_6_N_50_dt_5.8309e-05'
# fileName = 'sr-f_harm__IC_6_N_100_dt_2.91545e-05'
# fileName = 'sr-f_harm__IC_6_N_200_dt_1.45773e-05'
# fileName = 'sr-f_harm__IC_6_N_400_dt_7.28863e-06'

rank = 0

# period = 1 / (3 * np.pi * 343)
# period = 0.0002

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

# dataGrid = uGrid  + 0.05*(np.random.random(np.shape(uGrid))-0.5*np.ones(np.shape(uGrid)))
dataGrid = uGrid

# print('Drawing...')

# # Plot read data
# fig = plt.figure(figsize=(17,6))
# plt.pcolor(xGrid, tGrid, uGrid)
# plt.title('Displacement')
# plt.colorbar()
# plt.show()
# # print('Done')

print('Computing DMD...')

# ----------------------------------------------------------------------------------------------------------------------

# Compute DMD on displacement
d  = 5
dmd = HODMD(svd_rank=rank, opt=True, d=d)
# dmd = MrDMD(svd_rank=rank, max_level = 3, max_cycles = 1)
dmd.fit(dataGrid.T)

# Show eigenvalues
for eig in dmd.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

dmd.plot_eigs(show_axes=True, show_unit_circle=True)

# Show modes
fig = plt.figure(figsize=(8,3))
fig.subplots_adjust(top=0.8, bottom=0.2)
plt.subplot(121)
for mode in dmd.modes.T:
    plt.plot(xVector, mode.real)
    plt.title('Modes')
    plt.xlabel('$x$')
# fig.savefig('DMDResults/DMDModes_'+fileName+'_Rank-%i_d-%i.png'%(rank,d))
# plt.show()

# Show dynamics
# fig = plt.figure(figsize=(10,6))
plt.subplot(122)
for dynamic in dmd.dynamics:
    plt.plot(tVector, dynamic.real)
    plt.title('Dynamics')
    plt.xlabel('$t$')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
fig.savefig('DMDResults/Modes_'+fileName+'_Rank-%i_d-%i.png'%(rank,d))
plt.show()

#
# # Show each mode's contribution and the addition
# fig = plt.figure(figsize=(17, 6))
# for n, mode, dynamic in zip(range(241, 248), dmd.modes.T, dmd.dynamics):
#     plt.subplot(n)
#     plt.pcolor(xGrid, tGrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
#
# plt.subplot(248)
# plt.pcolor(xGrid, tGrid, dmd.reconstructed_data.T.real)
# plt.colorbar()
# plt.show()
#
# # Show absolute error
# plt.pcolor(xGrid, tGrid, (dataGrid-dmd.reconstructed_data.T).real)
# fig = plt.colorbar()
# plt.show()

print('Drawing...')

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
plt.pcolor(xGrid, tGrid, dmd.reconstructed_data.T.real)
cbar = plt.colorbar()
plt.title('DMD approximation', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

plt.subplot(133)
plt.pcolor(xGrid, tGrid, (dataGrid-dmd.reconstructed_data.T).real)
cbar = plt.colorbar()
plt.title('Absolute error', pad=20)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

fig.savefig('DMDResults/Recons_'+fileName+'_Rank-%i_d-%i.png'%(rank,d))
plt.show()

# # ----------------------------------------------------------------------------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------
# CO = [8, 16]
# # CO = [1, 2, 4]
# # CO = [4]
# # CO = [6/6,5/6,4/6,3/6]
# # CO = [7/6,8/6,9/6,10/6,11/6,12/6]
# # CO = [1]
# # D = [4, 8, 10]
# for co in CO:
# # for d in D:
#
#     tCutoff = 1/co * period
#     # tCutoff = co * period
#     # tCutoff = period
#     # tCutoff = 0.003/4*3
#
#     # Index of first element in tVector greater than tCutoff in tVector
#     idxCutoff = next(x for x, val in enumerate(tVector) if val > tCutoff)
#
#     # Cutoff time vector
#     tVectorHalf = tVector[0:idxCutoff]
#     print('\ntVector Cut Off length: %i\n' %len(tVectorHalf))
#
#     # Create the space-time grids
#     xGrid, tGrid = np.meshgrid(xVector, tVector)
#     xGridHalf, tGridHalf = np.meshgrid(xVector, tVectorHalf)
#
#     # Cut off data grids
#     uGridHalf = uGrid[0:idxCutoff, :]
#     vGridHalf = vGrid[0:idxCutoff, :]
#     aGridHalf = aGrid[0:idxCutoff, :]
#     uexGridHalf = uexGrid[0:idxCutoff, :]
#
#     dataGridHalf = uGridHalf
#
#     # ----------------------------------------------------------------------------------------------------------------------
#
#     # Compute DMD on displacement
#     d = 5
#     hodmd = DMD(svd_rank=rank)#, d=d) # HODMD: exact, opt, d, rank 0
#     # hodmd = MrDMD(svd_rank=rank, max_level = 3, max_cycles = 1)
#     hodmd.fit(dataGridHalf.T)
#
#     # # Show eigenvalues
#     # for eig in hodmd.eigs:
#     #     print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
#     #
#     # hodmd.plot_eigs(show_axes=True, show_unit_circle=True)
#
#     # # Show modes
#     # for mode in hodmd.modes.T:
#     #     plt.plot(xVector, mode.real)
#     #     plt.title('Modes')
#     # plt.show()
#     #
#     # # Show dynamics
#     # for dynamic in hodmd.dynamics:
#     #     plt.plot(tVectorHalf, dynamic.real)
#     #     plt.title('Dynamics')
#     # plt.show()
#
#     hodmd.original_time['dt'] = hodmd.dmd_time['dt'] = tVectorHalf[1] - tVectorHalf[0]
#     hodmd.original_time['t0'] = hodmd.dmd_time['t0'] = tVectorHalf[0]
#     hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = tVectorHalf[-1]
#     hodmd.dmd_time['tend'] = tVector[-1]
#
#     # Show each mode's contribution and the addition
#     # fig = plt.figure(figsize=(17, 6))
#     # for n, mode, dynamic in zip(range(241, 248), hodmd.modes.T, hodmd.dynamics):
#     #     plt.subplot(n)
#     #     plt.pcolor(xGrid, tGrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
#     #
#     # plt.subplot(248)
#     # plt.pcolor(xGrid, tGrid, hodmd.reconstructed_data.T.real)
#     # plt.colorbar()
#     # plt.show()
#     #
#     # # Show absolute error
#     # plt.pcolor(xGrid, tGrid, (dataGrid-dmd.reconstructed_data.T).real)
#     # fig = plt.colorbar()
#     # plt.show()
#
#     print('Drawing...')
#
#     # Partial reconstruction
#     fig = plt.figure(figsize=(20,6))
#     plt.subplot(131)
#     plt.pcolor(xGridHalf, tGridHalf, dataGridHalf)
#     plt.ylim(0,tVector[-1])
#     plt.colorbar()
#     plt.title('Given data')
#
#     plt.subplot(132)
#     plt.pcolor(xGrid, tGrid, hodmd.reconstructed_data.T.real)
#     plt.colorbar()
#     plt.title('DMD approximation')
#
#     plt.subplot(133)
#     plt.pcolor(xGrid, tGrid, (dataGrid-hodmd.reconstructed_data.T).real)
#     plt.colorbar()
#     plt.title('Absolute error with full data')
#
#
#     # fig.savefig('figures/DMDMeshTest_'+fileName+'_CO-%f_Rank-%i_d-%i.png'%(tCutoff,rank,d))
#     plt.show()


# CO = [1,5,100]#, 11, 50]
# CO = [3/6,2/6,1/6,1/12]
CO = [0.005]
for co in CO:

    # tCutoff = tVector[co]
    # tCutoff = co * period
    tCutoff = 0.0046
    # tCutoff = tVector[-1] - co
    # rank = co


    # Index of first element in tVector greater than tCutoff in tVector
    idxCutoff = next(x for x, val in enumerate(tVector) if val >= tCutoff)

    # Cutoff time vector
    tVectorHalf = tVector[0:idxCutoff]
    print('\ntVector Cut Off length: %i\n' %len(tVectorHalf))
    nSnap = len(tVectorHalf)

    # Create the space-time grids
    xGrid, tGrid = np.meshgrid(xVector, tVector)
    xGridHalf, tGridHalf = np.meshgrid(xVector, tVectorHalf)

    # Cut off data grids
    uGridHalf = uGrid[0:idxCutoff, :]
    vGridHalf = vGrid[0:idxCutoff, :]
    aGridHalf = aGrid[0:idxCutoff, :]
    uexGridHalf = uexGrid[0:idxCutoff, :]

    # dataGridHalf = uGridHalf
    dataGridHalf = dataGrid[0:idxCutoff, :]

    # ----------------------------------------------------------------------------------------------------------------------

    # Compute DMD on displacement
    d = 5
    dmd = HODMD(svd_rank=rank, d=d, opt=True)#, opt=True, d=d) # HODMD: exact, opt, d, rank 0
    # hodmd = MrDMD(svd_rank=rank, max_level = 3, max_cycles = 1)
    dmd.fit(dataGridHalf.T)

    # Show modes
    fig = plt.figure(figsize=(8,3))
    fig.subplots_adjust(top=0.8, bottom=0.2)
    plt.subplot(121)
    for mode in dmd.modes.T:
        plt.plot(xVector, mode.real)
        plt.title('Modes')
        plt.xlabel('$x$')

    plt.subplot(122)
    for dynamic in dmd.dynamics:
        plt.plot(tVector, dynamic.real)
        plt.title('Dynamics')
        plt.xlabel('$t$')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.savefig('DMDResults/tanspModes_'+fileName+'_nSnap-%i_Rank-%i_d-%i.png'%(nSnap,rank,d))
    plt.show()

    dmd.original_time['dt'] = dmd.dmd_time['dt'] = tVectorHalf[1] - tVectorHalf[0]
    dmd.original_time['t0'] = dmd.dmd_time['t0'] = tVectorHalf[0]
    dmd.original_time['tend'] = dmd.dmd_time['tend'] = tVectorHalf[-1]
    dmd.dmd_time['tend'] = tVector[-1]

    print('Drawing...')

    # Full reconstruction
    fig = plt.figure(figsize=(8,3))
    fig.subplots_adjust(wspace=0.5,top=0.8, bottom=0.2, right=0.95, left=0.05)

    plt.subplot(131)
    plt.pcolor(xGridHalf, tGridHalf, dataGridHalf)
    plt.ylim(0,tVector[-1])
    cbar = plt.colorbar()
    plt.title('Given data', pad=20)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplot(132)
    plt.pcolor(xGrid, tGrid, dmd.reconstructed_data.T.real)
    cbar = plt.colorbar()
    plt.title('DMD approximation', pad=20)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplot(133)
    plt.pcolor(xGrid, tGrid, (dataGrid-dmd.reconstructed_data.T).real)
    cbar = plt.colorbar()
    plt.title('Absolute error', pad=20)
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    fig.savefig('DMDResults/tanspPred_'+fileName+'_nSnap-%i_Rank-%i_d-%i.png'%(nSnap,rank,d))
    plt.show()







#

print('Done')