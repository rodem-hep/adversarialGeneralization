from matplotlib import pyplot as plt
import numpy as np

from scipy import integrate

import concurrent.futures
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate

# Constants
m_W = 80.4 # GeV
m_Z = 91.2 # GeV
m_b = 4.18 # GeV
m_t = 173 # GeV
v = 247 # GeV
sin2theta_W = 0.231
cos2theta_W = np.power(m_W/m_Z,2)
N_c = 3
alpha_EM = 1/137
alpha_s = 1
mass_H = 125 # GeV

# Define the decay rates
def Gamma_Hff(m_H, m_f):
    return N_c * alpha_EM/(8*sin2theta_W) * (m_H*np.power(m_f,2)/np.power(m_W,2)) * np.power((1-4*np.power(m_f,2)/np.power(m_W,2)),1.5)

def Gamma_Hww(m_H):
    return alpha_EM/(16*sin2theta_W)*(np.power(m_H,3)/np.power(m_W,2))*(1-4*np.power(m_W,2)/np.power(m_H,2)+12*np.power(m_W,4)/np.power(m_H,4))*np.sqrt(1-4*np.power(m_W,2)/np.power(m_H,2))

def Gamma_Hzz(m_H):
    return 2*alpha_EM/(64*cos2theta_W*sin2theta_W)*(np.power(m_H,3)/np.power(m_Z,2))*(1-4*np.power(m_Z,2)/np.power(m_H,2)+12*np.power(m_Z,4)/np.power(m_H,4))*np.sqrt(1-4*np.power(m_Z,2)/np.power(m_H,2))

def Gamma_Hgg(m_H, m_q):
    # Compute double integral from x=0 to 1 and z=0 to 1-x of the function If(x,z, m_H, m_q)
    I, err = integrate.dblquad(If, 0, 1, lambda x: 0, lambda x: 1-x, args=(m_H, m_q))

    return 1/(4*np.power(np.pi,2))*alpha_EM*np.power(alpha_s,2)*np.power(m_H,3)*I/sin2theta_W/np.power(m_W,2)

def If(x,z, m_H, m_q):
    return (1-4*x*z)/(1-x*z*(np.power(m_H,2)/np.power(m_q,2)))

def calculate_Gamma_g(m_H, m_q, Gamma_Hgg=Gamma_Hgg):
    return Gamma_Hgg(m_H, m_q)*1000 # GeV -> MeV

def sf(energy, theta):
    return 4*np.power(energy,2)

def tf(energy, theta):
    p = np.sqrt(np.power(energy,2)-np.power(m_W,2))
    return -2*np.power(energy,2) + 2*m_W**2 + 2*(p**2)*np.cos(theta)

def uf(energy, theta):
    p = np.sqrt(np.power(energy,2)-np.power(m_W,2))
    return -2*np.power(energy,2) + 2*m_W**2 - 2*(p**2)*np.cos(theta)


def A_4(energy, theta):
    s = sf(energy, theta)
    t = tf(energy, theta)
    u = uf(energy, theta)

    return 1/(v**2) * (np.power(s,2) + 4*s*t + np.power(t,2) - 4*np.power(m_W,2)*(s+t) - (8*np.power(m_W,2)/s)*u*t)

def A_s(energy, theta):
    s = sf(energy, theta)
    t = tf(energy, theta)
    u = uf(energy, theta)

    return -1/(v**2) * (s*(t-u)-3*np.power(m_W,2)*(t-u))

def A_t(energy, theta):
    s = sf(energy, theta)
    t = tf(energy, theta)
    u = uf(energy, theta)

    return -1/(v**2) * ((s-u)*t - 3*np.power(m_W,2)*(s-u) + (8*np.power(m_W,2)/s)*np.power(u,2))

def A_higgs(energy, theta, m_H):
    s = sf(energy, theta)
    t = tf(energy, theta)
    u = uf(energy, theta)

    return -1/(v**2) * ((np.power(s-2*np.power(m_W,2),2))/(s-np.power(m_H,2)) + (np.power((t-2*np.power(m_W,2)),2))/(t-np.power(m_H,2)))

def A_gauge(energy, theta):
    return -1/(v**2) * uf(energy, theta)
def A_tot(energy, theta, m_H):
    # return A_gauge(energy, theta) + A_higgs(energy, theta, m_H)
    return A_4(energy, theta) + A_s(energy, theta) + A_t(energy, theta) + A_higgs(energy, theta, m_H)

def main():

    # Plot the decay rates for bottom quarks for different Higgs masses
    m_Hs = np.linspace(50, 250, 250)
    Gamma = Gamma_Hff(m_Hs, m_b)*1000 # GeV -> MeV

    # Use a style suitable for scientific reports
    plt.style.use('seaborn-whitegrid')

    # Increase the figure size
    plt.figure(figsize=(10, 6))

    plt.plot(m_Hs, Gamma, linewidth=2)
    plt.xlabel("Higgs boson mass $M_H$ [GeV]", fontsize=14)
    plt.ylabel(r"Decay Width $\Gamma_{fb\bar{b}}$ [MeV]", fontsize=14) 
    # plt.title("Decay Width vs Higgs Boson Mass", fontsize=16)

    # Add a grid
    plt.grid(True)

    # Add a vertical line at the actual mass of the Higgs boson
    plt.axvline(x=mass_H, color='r', linestyle='--')

    # Tight layout
    plt.tight_layout()
    # Save the plot
    plt.savefig("C:/Users/franc/Documents/UNIGE - Physique/Master - Première Année - Printemps 2024/LaboTheo/Report/graphics/GammaHbb.png")
    plt.close()


    # Plot the decay rates for W bosons and Z bosons for different Higgs masses (same plot)

    m_Hs = np.linspace(150, 345, 500)

    # Calculate the decay rates
    Gamma_W = Gamma_Hww(m_Hs)*1000 # GeV -> MeV
    Gamma_Z = Gamma_Hzz(m_Hs)*1000 # GeV -> MeV

    # Increase the figure size
    plt.figure(figsize=(10, 6))

    plt.plot(m_Hs, Gamma_W, linewidth=2, label=r"$\Gamma_{W^+W^-}$")
    plt.plot(m_Hs, Gamma_Z, linewidth=2, label=r"$\Gamma_{ZZ}$")
    plt.xlabel("Higgs boson mass $M_H$ [GeV]", fontsize=14)
    plt.ylabel(r"Decay Width $\Gamma$ [MeV]", fontsize=14)
    # plt.title("Decay Width vs Higgs Boson Mass", fontsize=16)
    plt.legend()

    # Add a grid
    plt.grid(True)

    # Add a vertical line at the actual mass of the Higgs boson
    plt.axvline(x=mass_H, color='r', linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("C:/Users/franc/Documents/UNIGE - Physique/Master - Première Année - Printemps 2024/LaboTheo/Report/graphics/GammaHVV.png")
    plt.close()

    # Plot the decay rates for gluons for different Higgs masses
    m_Hs = np.linspace(50, 350, 500)

    # Calculate the decay rates
    Gamma_g = np.zeros(len(m_Hs))


    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, Gamma in enumerate(executor.map(calculate_Gamma_g, m_Hs, [m_t]*len(m_Hs))):
            Gamma_g[i] = Gamma

    # Increase the figure size
    plt.figure(figsize=(10, 6))

    plt.plot(m_Hs, Gamma_g, linewidth=2)
    plt.xlabel("Higgs boson mass $M_H$ [GeV]", fontsize=14)
    plt.ylabel(r"Decay Width $\Gamma_{gg}$ [MeV]", fontsize=14)
    # plt.title("Decay Width vs Higgs Boson Mass", fontsize=16)

    # Add a grid
    plt.grid(True)

    # Add a vertical line at the actual mass of the Higgs boson
    plt.axvline(x=mass_H, color='r', linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig("C:/Users/franc/Documents/UNIGE - Physique/Master - Première Année - Printemps 2024/LaboTheo/Report/graphics/GammaHgg.png")
    plt.close()

    # Plot the scattering amplitude in function of the energy for different higgs masses
    energies = np.linspace(50, 500, 500)
    theta = np.pi/2


    def integrand(theta, energy, m_H):
        return A_tot(energy, theta, m_H)

    # Increase the figure size
    plt.figure(figsize=(10, 6))

    for m_H in [50, 125, 300]:
        results = []
        for energy in energies:
            result, error = integrate.quad(integrand, 0, 2*np.pi, args=(energy, m_H))
            results.append(result)
        plt.plot(energies, results, label=f"$M_H = {m_H}$ GeV")
    # for m_H in [50, 125, 300]:
    #     plt.plot(energies, A_tot(energies, theta, m_H), label=f"$M_H = {m_H}$ GeV")
    
    # plt.plot(energies, -1/(v**2)*uf(energies, theta), label="u", linestyle='--')
    plt.xlabel("Energy [GeV]", fontsize=14)
    plt.ylabel(r"Scattering Amplitude $A_{tot}$", fontsize=14)
    # plt.title("Scattering Amplitude vs Energy", fontsize=16)
    plt.legend()

    # Add a grid
    plt.grid(True)

    # Save the plot
    plt.savefig("C:/Users/franc/Documents/UNIGE - Physique/Master - Première Année - Printemps 2024/LaboTheo/Report/graphics/Atot.png")
    plt.close()




    return

if __name__ == "__main__":
    main()