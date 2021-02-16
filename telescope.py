##created by Hannah Gulick 02/09/2021

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

class Telescope:
    h = 6.63e-27*u.cm**2*u.g*u.s**-1 #cm2 g/s
    h_erg = 6.63e-27*u.erg*u.s #erg s
    c = 2.99e10*u.cm/u.s #cm/s
    #tau_atm = 1. #transmissivity of atmosphere
    
    #filter coverage in angstrom (Bessell et al. 1998,  http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
    dellu = u.angstrom
    deltaLamb = {'U': 600*dellu, 'B': 900*dellu, 'V': 850*dellu, 'R': 1500*dellu, 'I': 1500*dellu, 
                 'J': 2600*dellu, 'H': 2900*dellu, 'K': 4100*dellu}
    #sky background in erg/s/cm**2/angstrom/arcsec**2 https://hst-docs.stsci.edu/wfc3ihb/chapter-9-wfc3-exposure-time-calculation/9-7-sky-background
    bskyu = u.erg/u.s/(u.cm)**2/u.angstrom/(u.arcsec)**2
    Bsky_erg = {'U': 3.55E-18*bskyu, 'B': 7.57E-18*bskyu, 'V': 7.72E-18*bskyu, 'R': 7.56E-18*bskyu, 
                'I': 5.38E-18*bskyu, 'J': 2.61E-18*bskyu, 'H': 1.43E-18*bskyu, 'K': 4100}
    #effective wavelength for filters (cm) http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    lambu = u.cm
    lamb_eff = {'U': 3.60e-5*lambu, 'B': 4.38e-5*lambu, 'V': 5.45e-5*lambu, 'R': 6.41e-5*lambu, 
                'I': 7.98e-5*lambu, 'J': 0.000122*lambu, 'H': 0.000163*lambu, 'K': 0.000219*lambu}
    
   
    #Vega zero-point flux: erg cm-2 s-1 A-1 (Bessell et al. 1998,  http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
    F0u = u.erg*(u.cm**(-2))/u.s/u.angstrom
    F0_lamb = {'U': 417e-11*F0u, 'B': 632e-11*F0u, 'V': 363.1e-11*F0u, 'R': 217.7e-11*F0u, 'I': 112.6e-11*F0u, 
               'J': 31.47e-11*F0u, 'H': 11.38e-11*F0u, 'K': 3.961e-11*F0u}
    
    ##Zero-point fluxes: Jy (from http://astroweb.case.edu/ssm/ASTR620/mags.html)
    F0lamb_AB = {'U': 1810*F0u, 'B': 4260*F0u, 'V': 3640*F0u, 'R': 3080*F0u, 'I': 2550*F0u, 
              'J': 1600*F0u, 'H': 1080*F0u, 'K': 670*F0u}
    
    #converting Vega magnitude to AB magnitude: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    deltaMag = {'U': 0.79, 'B': -0.09, 'V': 0.02, 'R': 0.21, 'I': 0.45, 'J': 0.91, 'H': 1.39, 'K': 1.85}
    
    #put units
    def __init__(self, filt, phot_r, D, tau_ins, tau_atm, eta, L_ins, Dark, Ron, n_pixel):
        '''This comment is a work in progress:
        
        Inputs: 
        filt: string
            a single letter for the filter name; either U, B, V, R, I, J, H, or K
        phot_r: integer or float with astropy units
            an integer value (recommended units: ") which gives the photometric radius
        D: integer or float with astropy units
            the diamter of the telescope (recommeded units: cm)
        tau_ins: integer or float
            the instrumental efficiency
        tau_atm: integer or float
            the transmissivity of the atmosphere (take to be 1 for space)
        eta: integer or float with astropy units
            the quantum efficiency of the detector (recommended units e-)
        L_ins: integer or float with astropy units
            the instrumental irradiance (recommended units: erg/cm**2/sr/s)
        Dark: integer or float with astropy units
            the dark current (recommended units: electron/pix/s)
        Ron: integer or float with astropy units
            the RMS readout noise (recommended units: electron**0.5/s**0.5)
        n_pixel: integer or float with astropy units
            the number of pixels covered by a source (recommended units: pix)'''
        
        
        self.filt = filt
        self.phot_r = phot_r
        self.D = D
        self.tau_ins = tau_ins
        self.tau_atm = tau_atm
        self.eta = eta
        self.L_ins = L_ins
        self.Dark = Dark
        self.Ron = Ron
        self.n_pixel = n_pixel
        
    def photon_E(self, lamb_eff=lamb_eff):
        h_erg = 6.63e-27*u.erg*u.s #erg s
        c = 2.99e10*u.cm/u.s #cm/s
        E_photon = h_erg*c/lamb_eff[self.filt] #erg
        return E_photon
        
    def bsky(self, lamb_eff = lamb_eff, deltaLamb = deltaLamb, Bsky_erg = Bsky_erg, h_erg = h_erg, c =c):
        Bsky = Bsky_erg[self.filt]*self.tau_ins*self.eta*np.pi*(self.D/2.)**2*self.phot_r**2*(lamb_eff[self.filt]*deltaLamb[self.filt]/(h_erg*c))
        return Bsky

    def mag_to_flux(self, m1, deltaLamb = deltaLamb, F0_lamb = F0_lamb):
        '''Inputs: Vega magnitude of source and filter;
        filter options include: 'U', 'B', 'V', 'R', 'I', 'J', 'H', or 'K'
        Output: Vega magnitude source flux in erg cm-2 s-1 A-1''' 
    
        F0_lamb = F0_lamb[self.filt]*deltaLamb[self.filt] #erg cm-2 s-1
        m2 = 0.0 #vega mag zero point
        F1 = F0_lamb*10**(-(m1-m2)/2.5) #erg cm-2 s-1 A-1
        return F1
    
    def mag_to_flux_AB(self, mag, F0lamb=F0lamb_AB, deltaMag=deltaMag, deltaLamb=deltaLamb, lamb_eff=lamb_eff):
        '''Inputs: Vega magnitude of source and filter;
            filter options include: 'U', 'B', 'V', 'R', 'I', 'J', 'H', or 'K'
            Output: AB magnitude source flux in erg cm-2 s-1 A-1''' 
    
        magAB = mag+deltaMag[self.filt]
    
        F1 = F0lamb[self.filt]*10**(-0.4*magAB) *deltaLamb[self.filt]#jy
        #conversion Jy --> erg cm-2 s-1 A-1
        Flamb = 3e-5*F1/(lamb_eff[self.filt])**2 #erg cm-2 s-1 A-1 ( conversion from https://www.stsci.edu/~strolger/docs/UNITS.txt)
        return Flamb
    
    def sigma( self, m, t, h =h, c=c, lamb_eff=lamb_eff):
        
        #times = [t[i] for i in range(len(t))]
        F_sig_vega = self.mag_to_flux(m)
        F_sig_AB = self.mag_to_flux_AB(m)
    
        Sig = self.tau_atm*self.tau_ins*self.eta*F_sig_vega*np.pi*(self.D/2.)**2*lamb_eff[self.filt]/(h*c)#signal electrons per second
        B_ins = self.eta*self.L_ins*np.pi*(self.D/2.)** 2*self.phot_r**2*(lamb_eff[self.filt]/(h*c))#electrons per second from the total instrumental background
        sigma = [np.sqrt(Sig*t[i]+2.*(self.bsky()*t[i]+B_ins*t[i]+self.n_pixel*(self.Dark*t[i]+self.Ron**2)))for i in range(len(t))]
        
        return sigma, self.bsky(), Sig, B_ins

    
    def s_n(self, m, t, n_f):
        
        beta = self.bsky()/self.sigma(m,t)[2]
        alpha1 = self.sigma(m,t)[3]/self.bsky()
        alpha2 = [self.n_pixel*(self.Dark*t[i]+self.Ron**2)/(self.bsky()*t[i]) for i in range(len(t))]
        
        S_N = [np.sqrt(n_f*t[i])*np.sqrt(self.bsky()/beta)*np.sqrt(1./(1.+2.*(beta+beta*alpha1+beta*alpha2[i])))for i in range(len(t))]

        return S_N, beta
    
    
    def sn_v_time_plot(self, t_low, t_high, m, n_f, lamb_eff=lamb_eff):
    
        times = np.linspace(t_low, t_high)
        times = [i*u.s for i in times]
        
        s_n_vega, beta_vega = self.s_n( m, times, n_f)
        snvega = [i/(1.*(u.electron**0.5)) for i in s_n_vega]
        times = [i/(1.*(u.s)) for i in times]
        plt.plot(times, snvega, linestyle = 'dashed', linewidth = 5, label = 'Vega Mag')
        plt.xlabel('Time (s)', fontsize = 40)
        plt.ylabel('S/N', fontsize = 40)
        plt.xticks(size = 30)
        plt.yticks(size = 30)
        plt.title('Time versus S/N for '+str(m)+' mag star', fontsize = 30)
        plt.legend(fontsize = 20)
        plt.savefig('time_sn_'+str(lamb_eff[self.filt])+'_'+str(m)+'_'+str(self.D)+'mag.png')
        plt.show()
    
        return
    
    def sn_v_mag_plot(self, mag_low, mag_high, t, n_f, lamb_eff=lamb_eff):
        times = np.array([t,0.1])
        times = [i*u.s for i in times]
        mags = np.linspace(mag_low, mag_high)

        s_n_vega, beta_vega = self.s_n( mags, times, n_f)
        snvega = [i/(1.*(u.electron**0.5)) for i in s_n_vega[0]]
        diff_func = lambda l: abs(l-10)
    
        SN10_vega = min(snvega, key=diff_func)
        SN10i_vega = list(snvega).index(SN10_vega)
        mag10_vega = mags[SN10i_vega]

        plt.plot(mags, snvega, linestyle = 'dashed', linewidth = 5, label = 'Vega Mag')
        plt.xlabel('Apparent Magnitude', fontsize = 40)
        plt.ylabel('S/N', fontsize = 40)
        plt.xticks(size = 30)
        plt.yticks(size = 30)
        if max(snvega) >= 10.:
            plt.axhline(10, alpha = 0.5, label = '10 sigma', color = 'magenta', linestyle = 'dashed', linewidth = 3)
            plt.axvline(mag10_vega, alpha = 0.5, label = 'Vega Mag for SN10: ' + str(round(mag10_vega, 2)), color = 'black', linestyle = 'dashed', linewidth = 3)
            plt.legend(fontsize = 30)
        plt.title('Magnitude versus S/N for ' + str(t) + ' sec Exposure', fontsize = 30)
        plt.legend()
        plt.savefig('time_sn_'+str(lamb_eff[self.filt])+'_'+str(t)+'_'+str(self.D)+'mag.png')
        plt.show()
    
        return
    

