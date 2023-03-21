import numpy as np
import matplotlib.pyplot as plt
from spectral import *
import spectralAdv.atmosphericConversions as atmpy
import os
import pickle
import time

lib = envi.open('clean_ENVI_lib.hdr')
remove_bad_bands = True
remove_noisy_spectra = True
remove_water_bands = True
plot_sample_atm = True
add_to_current = False
# atmospheric correction terms to use
include_gain_offset_squared = False # 0th order, 1st order, 2nd order
include_gain_squared = False # 1st order, 2nd order
include_gain = True # 1st order
N = 40 # spectra per batch
num_sample_spectra_plots = 0 # the number individual spectra to veiw
num_sample_XY_data = 0 # the number of batches of Xdata,Ydata to view
num_XY_data_to_generate = 100000

if remove_bad_bands:
    print("Removing Bad Bands...")
    # Define regions to remove
    # regions = [[0,0.45],[1.3,1.5],[1.75,2], [2.4,3]] # remove the low-signal extremes and water bands
    # regions = [[0,0.4],[1.3,1.5],[1.75,2], [2.479,3]] # remove the low-signal extremes and water bands
    regions = [[0, 0.4], [2.4, 3]]  # remove the low-signal extremes
    for r in regions:
        # get the range of wavelengths for bands to remove
        idx_start = np.argmin(np.abs(np.asarray(lib.bands.centers) - r[0]))
        idx_end = np.argmin(np.abs(np.asarray(lib.bands.centers) - r[1]))
        # remove the bands
        for i in sorted(range(idx_start, idx_end+1), reverse=True):
            del lib.bands.centers[i]
        # remove the data from the spectra
        lib.spectra = np.delete(lib.spectra, range(idx_start, idx_end+1), axis=1)

# Compute derivative with fringe bands removed but including water bands
D = np.abs(lib.spectra[:, 1:-1] - lib.spectra[:, 0:-2])
if remove_noisy_spectra:
    print("Removing Noisy Spectra...")
    # remove bands with significant noise spikes, defined by a pair of bands that change by
    # greater than Dthresh
    Dmax = np.max(D, axis=1)
    Dthresh = 0.005
    lib.spectra = np.delete(lib.spectra, np.where(Dmax > Dthresh), axis=0)
    D = np.delete(D, np.where(Dmax > Dthresh), axis=0)
    lib.names = np.delete(lib.names, np.where(Dmax > Dthresh), axis=0)

    Mmax = np.max(lib.spectra, axis=1)
    Mthresh = 0.15
    lib.spectra = np.delete(lib.spectra, np.where(Mmax < Mthresh), axis=0)
    D = np.delete(D, np.where(Mmax < Mthresh), axis=0)
    lib.names = np.delete(lib.names, np.where(Mmax < Mthresh), axis=0)

# get the metadata for the cleaned library
nSpec, nBands = lib.spectra.shape

# test - choose a single random spectrum
for repeat in range(num_sample_spectra_plots):
    idx = np.random.randint(0, nSpec, 1)
    plt.figure()
    plt.subplot(211)
    plt.plot(lib.bands.centers, lib.spectra[idx, :].flatten())
    plt.title('Spectrum '+str(idx))
    plt.ylim([0, np.min([np.max(lib.spectra[idx, :].flatten()),1])])
    plt.subplot(212)
    plt.plot(lib.bands.centers[1:-1], D[idx, :].flatten())
    plt.title('Absolute Value of Derivative')
    plt.show()

if remove_water_bands:
    print("Removing Water Bands...")
    # Define regions to remove
    # regions = [[0,0.45],[1.3,1.5],[1.75,2], [2.4,3]] # remove the low-signal extremes and water bands
    regions = [[1.3,1.5],[1.7,2.1]] # remove the low-signal water bands
    for r in regions:
        # get the range of wavelengths for bands to remove
        idx_start = np.argmin(np.abs(np.asarray(lib.bands.centers) - r[0]))
        idx_end = np.argmin(np.abs(np.asarray(lib.bands.centers) - r[1]))
        # remove the bands
        for i in sorted(range(idx_start, idx_end+1), reverse=True):
            del lib.bands.centers[i]
        # remove the data from the spectra
        lib.spectra = np.delete(lib.spectra, range(idx_start, idx_end+1), axis=1)

# compute the difference slopes for the cleaned livrary
D = np.abs(lib.spectra[:, 1:-1] - lib.spectra[:, 0:-2])
print("Number of clean spectra: "+str(nSpec))

# read the MODTRAN atmospheric coefficients
print("Reading atmospheric coefficients...")
ok, atm_coeff = atmpy.read_atmospheric_coefficients()

# make sure we have an even number of wl
#if np.mod(len(atm_coeff['wl']),2) == 1:
#    # remove the first band (378nm, not likely to get through atm)
#    atm_coeff['wl'] = atm_coeff['wl'][1:-1]


# read the atmospheric gases library
print("Reading gas library...")
fname = 'spectralAdv\\atm_gas_dict.pkl'
os.chdir(os.path.dirname(__file__))
pkl_file = open(fname, 'rb')
atm_dict = pickle.load(pkl_file)
pkl_file.close()


# test - read and plot atmospheric coefficients
if plot_sample_atm:
    conversion_type = 'ref_to_rad'
    solar_zenith_angle = 10
    atmospheric_index = 5
    aerosol_index = 3
    ok, atm_poly_coeff = atmpy.get_atm_poly_coeff(atm_coeff, conversion_type, solar_zenith_angle, atmospheric_index,
                                                  aerosol_index)
    plt.plot(atm_poly_coeff[:,0], color='b', label='offset')
    plt.plot(atm_poly_coeff[:,1], color='r', label='absorption')
    plt.plot(atm_poly_coeff[:,2], color='g', label='quadratic')
    plt.legend()
    plt.show()

# resample the library to the atmospheric wavelengths
print('Resampling to atmospheric coefficients bands')
resample = BandResampler(lib.bands.centers, atm_coeff['wl'])
spec = np.zeros([nSpec, len(atm_coeff['wl'])])
# resample the coefficients
for i in range(nSpec):
    spec[i, :] = resample(lib.spectra[i])

# get the metadata for the cleaned resampled library
nSpec, nBands = spec.shape
wl = atm_coeff['wl']

# create X and Y data for example viewing
conversion_type = 'ref_to_rad'
for i in range(num_sample_XY_data):
    # create random Y data (reflectance)
    idx = np.random.randint(0, nSpec, N)
    Yspec = spec[idx,:]
    Yspec[(N-1),:] = np.mean(Yspec[0:(N-2),:],axis=0)

    # get a randomly selected atmopsheric model
    solar_zenith_angle = 5*np.random.randint(1, 17, 1)
    atmospheric_index = np.random.randint(0, 6, 1)
    aerosol_index = np.random.randint(0, 12, 1)
    ok, atm_poly_coeff = atmpy.get_atm_poly_coeff(atm_coeff, conversion_type, solar_zenith_angle[0], atmospheric_index,
                                                  aerosol_index)
    Xspec = np.zeros([N,nBands])

    if include_offset:
        for i in range (N-1):
            Xspec[i] = atm_poly_coeff[:,0]*Yspec[i]**2 + atm_poly_coeff[:,1]*Yspec[i] + atm_poly_coeff[:,2]
    else:
        for i in range (N-1):
            Xspec[i] = atm_poly_coeff[:,0]*Yspec[i]**2 + atm_poly_coeff[:,1]*Yspec[i]
    Xspec[(N-1),:] = np.mean(Xspec[0:(N-2),:],axis=0)

    # example plot
    plt.figure()
    plt.subplot(211)
    plt.plot(wl,Xspec.T, alpha=0.4)
    plt.plot(wl,Xspec[(N-1),:].flatten(), color='k', linewidth=2)
    plt.subplot(212)
    plt.plot(wl,Yspec.T, alpha=0.4)
    plt.plot(wl,Yspec[(N-1),:].flatten(), color='k', linewidth=2)
    plt.show()



# create X and Y data
Xdata = np.zeros([num_XY_data_to_generate, N, nBands])
Ydata = np.zeros([num_XY_data_to_generate, N, nBands])
start = time.time()
for i in range(num_XY_data_to_generate):
    if (i % 500 == 0):
        print('Generating data batch: '+str(i)+' of '+str(num_XY_data_to_generate))
    # create a random subset for Y data (reflectance)
    idx = np.random.randint(0, nSpec, N)
    Yspec = spec[idx,:]
    YspecMean = np.mean(Yspec, axis=0)

    # get a randomly selected atmopsheric model
    solar_zenith_angle = 5*np.random.randint(1, 10, 1)
    atmospheric_index = np.random.randint(0, 6, 1)
    aerosol_index = np.random.randint(0, 12, 1)
    ok, atm_poly_coeff = atmpy.get_atm_poly_coeff(atm_coeff, conversion_type, solar_zenith_angle[0], atmospheric_index,
                                                  aerosol_index)
    Xspec = np.zeros([N,nBands])
    if include_gain_offset_squared:
        suffix = 'OGS'
        for spec_idx in range (N-1):
            # include gain, offset, quadratic
            Xspec[spec_idx] = atm_poly_coeff[:,0]*Yspec[spec_idx]**2 + atm_poly_coeff[:,1]*Yspec[spec_idx] + atm_poly_coeff[:,2]
    if include_gain_squared:
        suffix = 'GS'
        for spec_idx in range (N-1):
            # include gain and quadratice (no offset)
            Xspec[spec_idx] = atm_poly_coeff[:,0]*Yspec[spec_idx]**2 + atm_poly_coeff[:,1]*Yspec[spec_idx]
    if include_gain:
        suffix = 'G'
        for spec_idx in range (N-1):
            # include gain and quadratice (no offset)
            Xspec[spec_idx] = atm_poly_coeff[:,1]*Yspec[spec_idx]
    XspecMean = np.mean(Xspec,axis=0)

    Xspec[0, :] = XspecMean
    Xdata[i, :, :] = Xspec
    Yspec[0,:] = YspecMean
    Ydata[i, :, :] = Yspec

"""
print('Ensuring that the number of bands is even.')
if np.mod(Xspec.shape[1],2)==1:
    Xdata = Xdata[:,1:nBands]
    Ydata = Ydata[:,1:nBands]
    wl = wl[1:nBands]
"""

np.save('wl.npy', wl)
if add_to_current:
    if os.path.exists('Xdata_Means'+suffix+'.npy'):
        print('Reading prior data.')
        Xdata_from_file = np.load('Xdata_Means'+suffix+'.npy')
        Ydata_from_file = np.load('Ydata_Means'+suffix+'.npy')
        print('Merging data.')
        Xdata = np.vstack((Xdata_from_file, Xdata))
        Ydata = np.vstack((Ydata_from_file, Ydata))

    print('Saving data.')
    np.save('Xdata_Means' + suffix + '.npy', Xdata)
    np.save('Ydata_Means' + suffix + '.npy', Ydata)

else:
    print('Saving data.')
    np.save('Xdata_Means' + suffix + '.npy', Xdata)
    np.save('Ydata_Means' + suffix + '.npy', Ydata)

end = time.time()
print("Elapsed time to generate data: "+str(end - start)+" seconds.")
print("Number of observations generated: "+str(num_XY_data_to_generate))
print("Total number of observations: "+str(Xdata.shape[0]))
print('pause')