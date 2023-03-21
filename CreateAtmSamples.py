import numpy as np
import matplotlib.pyplot as plt
from spectral import *
import spectralAdv.atmosphericConversions as atmpy
import os
import pickle
import time

num_XY_data_to_generate = 10000


# get the metadata for the cleaned library
nSpec, nBands = lib.spectra.shape

# read the MODTRAN atmospheric coefficients
print("Reading atmospheric coefficients...")
ok, atm_coeff = atmpy.read_atmospheric_coefficients()


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
    Yspec[(N-1),:] = np.nanmean(Yspec[0:(N-2),:],axis=0)

    # get a randomly selected atmopsheric model
    solar_zenith_angle = 5*np.random.randint(1, 10, 1)
    atmospheric_index = np.random.randint(0, 6, 1)
    aerosol_index = np.random.randint(0, 12, 1)
    ok, atm_poly_coeff = atmpy.get_atm_poly_coeff(atm_coeff, conversion_type, solar_zenith_angle[0], atmospheric_index,
                                                  aerosol_index)
    Xspec = np.zeros([N,nBands])
    if include_offset:
        for spec_idx in range (N-1):
            # include gain, offset, quadratic
            Xspec[spec_idx] = atm_poly_coeff[:,0]*Yspec[spec_idx]**2 + atm_poly_coeff[:,1]*Yspec[spec_idx] + atm_poly_coeff[:,2]
    else:
        for spec_idx in range (N-1):
            # include gain and quadratice (no offset)
            Xspec[spec_idx] = atm_poly_coeff[:,0]*Yspec[spec_idx]**2 + atm_poly_coeff[:,1]*Yspec[spec_idx]
    Xspec[(N-1),:] = np.nanmean(Xspec[0:(N-2),:],axis=0)

    Xdata[i,:,:] = Xspec
    Ydata[i,:,:] = Yspec

print('Ensuring that the number of bands is even.')
if np.mod(Xspec.shape[1],2)==1:
    Xdata = Xdata[:,:,1:nBands]
    Ydata = Ydata[:,:,1:nBands]
    wl = wl[1:nBands]

if add_to_current:
    if os.path.exists('Xdata'+suffix+'.npy'):
        print('Reading prior data.')
        Xdata_from_file = np.load('Xdata'+suffix+'.npy')
        Ydata_from_file = np.load('Ydata'+suffix+'.npy')
        print('Merging data.')
        Xdata = np.vstack((Xdata_from_file, Xdata))
        Ydata = np.vstack((Ydata_from_file, Ydata))

print('Saving data.')
np.save('Xdata'+suffix+'_small.npy', Xdata)
np.save('Ydata'+suffix+'_small.npy', Ydata)


end = time.time()
print("Elapsed time to generate data: "+str(end - start)+" seconds.")
print("Number of observations generated: "+str(num_XY_data_to_generate))
print("Total number of observations: "+str(Xdata.shape[0]))
print('pause')