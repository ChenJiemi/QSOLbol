import numpy as np
import logging
from sklearn.impute import KNNImputer
import sompy
import astropy.constants as ac
import astropy.units as au
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
import pickle
from pkg_resources import resource_filename
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize cosmology model
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class QSOLbol:
    def __init__(self, config_path=None):
        """
        Initialize QSOLbol
        
        Parameters:
            config_path: Path to the config file. If None, use the default config within the package.
        """
        try:
            if config_path is None:
                config_path = self._get_default_config_path()

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            self.config = self._load_config(config_path)
            self._load_models()
            
            logger.info("QSOLbol initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QSOLbol: {e}")
            raise

    def _get_default_config_path(self):
        """Get the default config file path"""
        try:
            return resource_filename(__name__, os.path.join("config", "default.yaml"))
        except Exception as e:
            logger.error(f"Failed to locate default config: {e}")
            raise

    def _load_config(self, config_path):
        """Load configuration file"""
        try:
            from QSOLbol.utils.io import load_config
            config = load_config(config_path)
            
            # Convert relative paths in config to absolute paths
            for path_key in ["som_model", "lbol_data", "lbol_err_sim"]:
                if path_key in config.get("paths", {}):
                    rel_path = config["paths"][path_key]
                    abs_path = os.path.join(os.path.dirname(config_path), rel_path)
                    config["paths"][path_key] = os.path.abspath(abs_path)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _load_models(self):
        """Load models and data"""
        try:

            from QSOLbol.utils.io import load_pickle, load_npz
            self.som = load_pickle(self.config["paths"]["som_model"])
            self.lbol_data = load_npz(self.config["paths"]["lbol_data"])
            self.lbol_err = load_npz(self.config["paths"]["lbol_err_sim"])
            self.imputer = KNNImputer(n_neighbors=self.config.get("knn_neighbors", 100))
            
            logger.info("Models loaded successfully")
            
        except KeyError as e:
            logger.error(f"Missing required path in config: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def calculate(self, wave: np.ndarray, mags: np.ndarray, mags_err: np.ndarray, z: np.ndarray, f_isotropy: bool = False,scale:bool=False,max_iter=3, wave_range: list = [4.0e13, 2.4e18]):
        """
        Calculate luminosity (Lbol) for multiple sources.

        Parameters
        ----------
        wave : np.ndarray
            wavelengh array (Angstrom), shape: (n_sources, n_frequencies).
        mags : np.ndarray
            SED array (mag), shape: (n_sources, n_frequencies).
        mags_err : np.ndarray
            SED error array (mag), shape: (n_sources, n_frequencies).
        z : np.ndarray
            Redshift array, shape: (n_sources, 1).
        f_isotropy : bool, optional
            Whether to apply isotropy correction (default: False).
        scale : bool, optional
            Whether to rescale the flux (default: False).
        wave_range : list, optional
            Integration frequency range (default: [4.0e13, 2.4e18]).
        max_iter: int, optional
            Maximum number of iterations for rescaling (default: 3).
        Returns
        -------
            All sources' log10(Lbol) and uncertainty; returns None if SNR <= 3.
        """
        nu = 2.99e18/wave #(ac.c / (wave * au.Angstrom)).to(au.Hz).value  
        flux = 10**(-0.4 * mags) * 3631
        flux_err = np.abs(np.log(10)*flux*mags_err*0.4)

        flux_orig = flux.copy()
        flux_err_orig = flux_err.copy()

        results = self.calculate_bol(nu, flux, flux_err, z, f_isotropy, wave_range)
        logLbol = results[0]
        warn = results[2]
        iteration = 0
        logger.info("Initial Lbol prediction completed.")

        while scale and not np.all(warn == 1) and iteration < max_iter:
            bad_idx = warn != 1

            # Recalculate the scaling factor from logLbol
            f_all = np.ones_like(logLbol)
            f_all[bad_idx] = 10 ** (46 - logLbol[bad_idx])
            f = f_all[bad_idx].reshape(-1, 1)

            results_r = self.calculate_bol(
                nu[bad_idx],
                flux_orig[bad_idx] * f,
                flux_err_orig[bad_idx] * f,
                z[bad_idx],
                f_isotropy,
                wave_range
            )
            logger.info("Iteration %d Lbol prediction completed.", iteration + 1)

            # Update logLbol and correct to original reference system (subtract log10(f))
            results[0][bad_idx] = results_r[0] - np.log10(f).reshape(-1,)
            results[1][bad_idx] = results_r[1]
            results[2][bad_idx] = results_r[2]

            logLbol = results[0]
            warn = results[2]
            iteration += 1

        if np.all(warn == 1.):
            logger.warning("All sources are within the safe range after %d iterations.", iteration)
        else:
            logger.warning("Some sources are still outside the safe range after %d iterations. Please check manually.", iteration)

        return results[0], results[1]#,results[2]
    
    def calculate_bol(self, nu: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, z: np.ndarray, f_isotropy: bool = False, wave_range: list = [4.0e13, 2.4e18]):

        SN = (flux / flux_err > 3)
        if not np.any(SN):
            logger.warning("No sources with SNR > 3. Returning None.")
            return None, None, None

        # Compute luminosity distance
        dL = cosmo.luminosity_distance(z).to(au.cm)
        Lnu = 4 * np.pi * dL.value**2 * flux / (1 + z) * 1e-23  # erg/s/Hz
        Lnu_err = 4 * np.pi * dL.value**2 * flux_err / (1 + z) * 1e-23  # erg/s/Hz

        # Compute logv and logvLv
        v = (nu * (1 + z))
        logv = np.log10(v)
        logvLv = np.log10(Lnu) + logv  # erg/s
        logvLv_err = np.abs(Lnu_err / Lnu / np.log(10))  # erg/s
        len1= logv.shape[0]

        # Sort logv in descending order and get indices
        sort_indices = np.argsort(logv, axis=1)[:, ::-1]

        # Apply sorting
        logv, logvLv, logvLv_err = [np.take_along_axis(arr, sort_indices, axis=1) for arr in [logv, logvLv, logvLv_err]]

        # Interpolate for L2500
        xnew2 = np.arange(np.log10((ac.c / (4 * au.um)).to(au.Hz).value), np.log10((10 * au.keV / ac.h).to(au.Hz).value), 0.02)
        xnew2_11 = np.append(xnew2[5:87:9], np.log10((2 * au.keV / ac.h).to(au.Hz).value))
        fre_2500 =np.log10((ac.c / (2500 * au.Angstrom)).to(au.Hz).value)
        fre_2kev=np.log10((2 * au.keV / ac.h).to(au.Hz).value)
        L2500 =[interpolate.interp1d(logv[i,SN[i]], logvLv[i,SN[i]], fill_value=np.nan, bounds_error=False)(fre_2500)  for i in range(len1)]
        L2500=np.array(L2500)

        # Estimate luminosity at 0.2keV, 2keV and 10keV
        need_to_interp=np.array([max(logv[i,SN[i]])!= fre_2kev for i in range(len1)])

        l2kev = (L2500 - fre_2500) * 0.721 + 4.531
        Lxpre=l2kev + fre_2kev
        c=[~np.isnan(L2500) & need_to_interp][0]

        fre_10kev=np.log10((10 * au.keV / ac.h).to(au.Hz).value)
        fre_02kev=np.log10((0.2 * au.keV / ac.h).to(au.Hz).value)
        fre_2kev=np.log10((2 * au.keV / ac.h).to(au.Hz).value)

        fre_2kev_logv = np.full((len1, 1), fre_2kev)
        fre_10kev_logv = np.full((len1, 1), fre_10kev)
        fre_02kev_logv = np.full((len1, 1), fre_02kev)

        logvLv_add = np.full((len1, 1), np.nan) 
        logvLv_err_add = np.full((len1, 1),np.nan)
        SN_add = np.full((len1, 1), False) 

        SN_add[c] = True
        logvLv_err_add[c]=0.01
        logvLv_add[c]=Lxpre[c].reshape(-1,1)

        logv0 = np.hstack((fre_10kev_logv, fre_2kev_logv, fre_02kev_logv, logv))
        logvLv0 = np.hstack((logvLv_add, logvLv_add, logvLv_add, logvLv))
        logvLv_err0 = np.hstack((logvLv_err_add, logvLv_err_add, logvLv_err_add, logvLv_err))
        SN0 = np.hstack((SN_add, SN_add, SN_add, SN))

        # interpolate into 11 features
        YPRE0 =[interpolate.interp1d(logv0[i,SN0[i]], logvLv0[i,SN0[i]], fill_value=np.nan, bounds_error=False)(xnew2_11) for i in range(len1)]
        YPRE0=np.array(YPRE0)

        # recover SED
        imputer = KNNImputer(n_neighbors=100)
        data=np.hstack((self.lbol_data["data_imputed"][:,5:87:9],self.lbol_data["data_imputed"][:,-1].reshape(-1,1)))
        YPRE0_nan = np.vstack((YPRE0, data))
        data_imputed_ = imputer.fit_transform(YPRE0_nan)[:len1]
        mask_feature = ~np.isnan(YPRE0)

        # Recalculate Lbol map if wave range changes
        if wave_range != [4.0e13, 2.4e18]:

            data_train_p = self.lbol_data["data_imputed"]
            p = self.lbol_data["p"]
            p_u = np.unique(p)
            SIZE = len(p_u)
            xnew2 = np.arange(np.log10(4.0e13), np.log10(2.4e18), 0.02)
            lenp = data_train_p.shape[0]
            xnew0 = np.arange(np.log10(wave_range[0]), np.log10(wave_range[1]), 0.02)
            data_train_p = [np.interp(xnew0, xnew2, data_train_p) for i in range(lenp)]
            logLbol_new = np.log10([np.trapz(np.log(10) * 10**data_train_p[i], xnew0) for i in range(lenp)])

            median_luminosities = np.array([np.median(logLbol_new[np.where(p == p_u[i])]) for i in range(SIZE)])
            median_luminosities_err = np.array([(np.percentile(logLbol_new[np.where(p == p_u[i])], 84) - np.percentile(logLbol_new[np.where(p == p_u[i])], 16)) / 2 for i in range(SIZE)])
        else:
            median_luminosities = self.lbol_data['median_luminosities']
            median_luminosities_err = self.lbol_data['median_luminosities_err']

        # SOM prediction
        p = self.som.find_k_nodes(data_imputed_, k=1)[1].reshape(-1,)
        logLbol = median_luminosities[p]

        # Compute Lbol uncertainty
        logLbol_err = np.full(len1, np.nan)  

        logLbol_err[mask_feature.sum(axis=1) == 11] = median_luminosities_err[p][0]
        logLbol_err[mask_feature.sum(axis=1) == 0] = np.nan

        partial_mask = (mask_feature.sum(axis=1) != 11) & (mask_feature.sum(axis=1) != 0)
        if np.any(partial_mask):
          logLbol_err_all = self.lbol_err['error']
          mask_feature_all = self.lbol_err['mask_feature_all']
          for i in np.where(partial_mask)[0]:
            idd = np.where((mask_feature_all == mask_feature[i]).all(axis=1))[0]
            if len(idd) > 0:
              logLbol_err[i] = logLbol_err_all[idd][0]        


        warning = np.full((len1,), 1.) 
        warning[(logLbol < 45.13) | (logLbol > 46.74)] = 10**(46 - logLbol[(logLbol < 45.13) | (logLbol > 46.74)])
 
        if f_isotropy:
            logLbol += np.log10(0.75)
            
        return logLbol, logLbol_err, warning
