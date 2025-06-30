import numpy as np


@dataclass
class PDConfig:
    """
    Pharmacodynamics model configuration
    """

    # Population dynamics
    k_growth: float = 1 / 60 / 24
    m_arrest: float = 2
    K_arrest: float = 0.3
    k_mitosis: float = 1 / 60 / 24
    k_division: float = 1 / 60
    k_arrest: float = 1 / 60
    k_transit_death: float = 1 / 60 / 24
    bone_marrow: bool = False
    k_transit_granul: float = 1 / 60 / 24
    f_neutrophil_feedback: lambda x: 1

    def __post_init__(self):
        self.koff = self.kon * self.KD
        m, K = self.m_arrest, self.K_arrest
        self.f_arrest = lambda x: x**m / (x**m + K**m)


class PD(object):
    """
    Pharmacodynamics model
    """

    def __init__(self, config: PDConfig, iv: np.array, pk: callable):
        """
        :param config: PDConfig class for parameters
        """
        # Parameters
        self.config = config
        # Initial values, which should be the concentration of:
        # (0) Interphase cells,
        # (1) M-phase cells,
        # (2 - 1+n) apoptosis transit compartments
        # (2+n - 1+n+m) granulopoiesis transit compartments, where the last one is the neutrophil.
        assert (
            len(iv) == 3 + config.n_transit_death + config.n_transit_granul
        ), "Wrong number of initial values"
        self.state = iv

        # PK function
        self.pk = pk

    def time_step(self, cp) -> list:
        """
        Update the model by one time step
        :param cp: plasma concentration of the drug
        """
        config = self.config
        I, M = self.state[:2]
        death_compartments = self.state[2 : 1 + config.n_transit_death]

        # Rates
        r_im = config.k_mitosis * I
        r_div = config.k_division * M
        r_arrest = config.k_arrest * config.f_arrest(Sd) * M

        ## Cell cycle
        dI = -r_im + 2 * r_div
        dM = r_im - r_div - r_arrest

        ## Transit compartments
        dA1 = r_arrest - config.k_transit_death * death_compartments[0]
        dA_other = config.k_transit_death * (
            death_compartments[:-1] - death_compartments[1:]
        )

        if not config.bone_marrow:
            return [dI, dM, dA1] + dA_other.tolist()

        ## Granulopoiesis
        granul_compartments = self.state[6 + config.n_transit_death :]
        k_granul = config.k_transit_granul * config.f_neutrophil_feedback(
            granul_compartments[-1]
        )
        dI -= config.k_transit_granul * I
        from_compartments = np.array([I] + list(granul_compartments[:-1]))
        dN = config.k_transit_granul * (from_compartments - granul_compartments)
        return [dD, dT, dTm, dTd, dI, dM, dA1] + dA_other.tolist() + dN.tolist()
