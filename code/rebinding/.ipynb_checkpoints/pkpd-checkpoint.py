import numpy as np
from scipy.integrate import solve_ivp


def PK(t, D0=1.0, alpha=1 / 60 / 8, texp=3 * 60):
    return D0 * (t < texp) + D0 * np.exp(-alpha * (t - texp)) * (t >= texp)


class PD(object):
    """
    Pharmacodynamics model
    """

    def __init__(self, bone_marrow=False, kwargs={}):
        """
        :param kwargs: tuned parameters
        """
        self.KD = 5e-3  # uM
        default_params = {
            "k0": 0,  # 1 / 60 / 24 * 20,
            "k1": 3 * 60,
            "k2": 1 * 60,
            "kon": 6.1e-2 * 60,
            "koff": 6.1e-2 * 60 * self.KD,
            "gamma": 1 / 60,
            "lamda": 1 / 60 / 24,
            "arrest_hill_coef": 2,
            "arrest_hill_const": 0.4,
            "tub_expr_hill_const": 1,
            "km": 1 / 60 / 24,
            "kd": 1 / 60,
            "ka": 1 / 60,
            "mu": 1 / 60 / 24,
            "tub_expr": lambda x: 1,
            "drug_stab_factor": lambda x: 1,
            "neutrophil_feedback": lambda x: 1,
            "nu": 1 / 40 / 60,
        }
        default_params |= kwargs
        for key, value in default_params.items():
            setattr(self, key, value)

    def arrest_rate(self, Sd) -> float:
        m = self.arrest_hill_coef
        return Sd**m / (Sd**m + self.arrest_hill_const**m)

    def time_step(
        self, t, y, PK, n_transit_death=3, bone_marrow=False, n_transit_granul=3
    ) -> list:
        """
        ODE for drug-target kinetics
        :param t: time
        :param y: state vector
        :param PK: plasma drug concentration
        :param n_transit_death: number of transit compartment before death
        """
        # Independent variables
        D, T, Tm, Td, I, M = y[:6]
        Sd = Td / (Tm + Td)

        # Rates
        r_syn = self.k0 * self.tub_expr(T)
        r_pol = self.k1 * T
        r_depol = self.k2 * Tm * self.drug_stab_factor(Sd)
        r_bind = self.kon * D * Tm
        r_unbind = self.koff * Td
        r_im = self.km * I
        r_div = self.kd * M * (1 - self.arrest_rate(Sd))
        r_arrest = self.ka * self.arrest_rate(Sd) * M

        # ODEs
        ## Drug-target kinetics
        dD = self.gamma * (PK(t) - D)
        dT = r_syn - r_pol + r_depol - self.lamda * T
        dTm = r_pol - r_depol - r_bind + r_unbind - self.lamda * Tm
        dTd = r_bind - r_unbind - self.lamda * Td

        ## Cell cycle
        dI = -r_im + 2 * r_div
        dM = r_im - r_div - r_arrest

        ## Transit compartment
        dA1 = r_arrest - self.mu * y[6]
        dA_other = self.mu * (
            np.array(y[6 : 5 + n_transit_death]) - np.array(y[7 : 6 + n_transit_death])
        )

        ## Granulopoiesis
        if not bone_marrow:
            return [dD, dT, dTm, dTd, dI, dM, dA1] + dA_other.tolist()

        k_granul = self.nu * self.neutrophil_feedback(y[-1])
        dI -= k_granul * I
        dN = k_granul * (
            np.array([I] + list(y[6 + n_transit_death : -1]))
            - np.array(y[6 + n_transit_death :])
        )
        return [dD, dT, dTm, dTd, dI, dM, dA1] + dA_other.tolist() + dN.tolist()
