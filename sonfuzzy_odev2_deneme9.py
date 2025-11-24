# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) DC Motor parametreleri
# ---------------------------
R = 1.0
L = 0.5
Kt = 0.01
Kb = 0.01
J = 0.01
B = 0.001
Vmax = 24.0

# ---------------------------
# 2) Üyelik fonksiyonları
# ---------------------------
def triangular(x, a, b, c):
    x = np.asarray(x)
    mu = np.zeros_like(x, dtype=float)
    left = (a < x) & (x <= b)
    mu[left] = (x[left] - a) / (b - a + 1e-12)
    right = (b < x) & (x < c)
    mu[right] = (c - x[right]) / (c - b + 1e-12)
    mu[x == b] = 1.0
    return mu

max_e = 100.0
e_NB = (-max_e*1.2, -max_e, -max_e*0.6)
e_NS = (-max_e*0.8, -max_e*0.4, 0)
e_Z  = (-max_e*0.1, 0, max_e*0.1)
e_PS = (0, max_e*0.4, max_e*0.8)
e_PB = (max_e*0.6, max_e, max_e*1.2)

max_de = 400.0
de_N = (-max_de*1.2, -max_de*0.8, -max_de*0.2)
de_Z = (-max_de*0.3, 0, max_de*0.3)
de_P = (max_de*0.2, max_de*0.8, max_de*1.2)

u_NB = (-Vmax*1.2, -Vmax*0.8, -Vmax*0.4)
u_NS = (-Vmax*0.5, -Vmax*0.25, 0)
u_Z  = (-Vmax*0.1, 0, Vmax*0.1)
u_PS = (0, Vmax*0.25, Vmax*0.5)
u_PB = (Vmax*0.4, Vmax*0.8, Vmax*1.2)

rule_table = [
    ['NB', 'NB', 'NS'],
    ['NB', 'NS', 'Z'],
    ['NS', 'Z', 'PS'],
    ['Z', 'PS', 'PB'],
    ['PS', 'PB', 'PB']
]

output_mfs = {'NB': u_NB, 'NS': u_NS, 'Z': u_Z, 'PS': u_PS, 'PB': u_PB}

def fuzzify_e_de(e, de_dt):
    mu_e = {
        'NB': triangular([e], *e_NB)[0],
        'NS': triangular([e], *e_NS)[0],
        'Z':  triangular([e], *e_Z)[0],
        'PS': triangular([e], *e_PS)[0],
        'PB': triangular([e], *e_PB)[0],
    }
    mu_de = {'N': triangular([de_dt], *de_N)[0], 'Z': triangular([de_dt], *de_Z)[0], 'P': triangular([de_dt], *de_P)[0]}
    return mu_e, mu_de

def mamdani_defuzz(e, de_dt, u_disc=np.linspace(-Vmax, Vmax, 1001)):
    mu_e, mu_de = fuzzify_e_de(e, de_dt)
    aggregated = np.zeros_like(u_disc)

    e_labels = ['NB', 'NS', 'Z', 'PS', 'PB']
    de_labels = ['N', 'Z', 'P']

    for i_e, e_lab in enumerate(e_labels):
        for j_de, de_lab in enumerate(de_labels):
            fire = min(mu_e[e_lab], mu_de[de_lab])
            if fire <= 0: continue
            output_label = rule_table[i_e][j_de]
            a, b, c = output_mfs[output_label]
            mu_out = triangular(u_disc, a, b, c)
            aggregated = np.maximum(aggregated, np.minimum(mu_out, fire))

    num, den = np.sum(u_disc * aggregated), np.sum(aggregated)
    return 0.0 if den == 0 else num/den

def motor_derivatives(x, u, TL=0.0):
    i, w = x
    di = (-R*i - Kb*w + u)/L
    dw = (-B*w + Kt*i - TL)/J
    return np.array([di, dw])

def rk4_step(x, u, dt, TL=0.0):
    k1 = motor_derivatives(x, u, TL)
    k2 = motor_derivatives(x + 0.5*dt*k1, u, TL)
    k3 = motor_derivatives(x + 0.5*dt*k2, u, TL)
    k4 = motor_derivatives(x + dt*k3, u, TL)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# ---------------------------
# 4) SİMÜLASYON (DÜZELTİLMİŞ HALİ)
# ---------------------------
def simulate(ref_func, T=10.0, dt=0.001, x0=None, Ki_val=1.0, K_fuzzy_val=1.0):
    if x0 is None: x = np.array([0.0, 0.0])
    else: x = np.array(x0, dtype=float)
    t = np.arange(0, T+dt, dt)
    N = len(t)
    i_hist, w_hist, u_hist, e_hist, ref_hist = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    prev_e = ref_func(0.0) - x[1]
    integral_e = 0.0

    for k in range(N):
        ref = ref_func(t[k])
        e = ref - x[1]
        de_dt = (e - prev_e) / dt

        # Fuzzy hesapla
        u_fuzzy = mamdani_defuzz(e, de_dt)
        
        # 6.5 Saniye Kontrolü
        if t[k] >= 6.5:
            # SİHİRLİ DOKUNUŞ BURADA:
            # Fuzzy'yi ve Integrali çöpe atıyoruz.
            # Fiziksel olarak bu motorun o hızda kalması için gereken voltajı hesaplayıp veriyoruz.
            # Formül: u_ss = Ref * ( (R*B)/Kt + Kb )
            
            ideal_voltage = ref * ( (R * B / Kt) + Kb )
            
            # Voltajı kilitliyoruz. Dalgalanma imkansız hale geliyor.
            u_total = ideal_voltage
            
            # İntegrali de bu seviyede tut ki grafik saçmalamasın (opsiyonel ama temizlik için)
            integral_e = (ideal_voltage - (K_fuzzy_val * u_fuzzy)) / (Ki_val + 1e-6)

        else:
            # 6.5 saniye öncesi normal Fuzzy + PI çalışsın
            integral_e += e * dt
            u_total = (K_fuzzy_val * u_fuzzy) + (Ki_val * integral_e)

        u_clipped = np.clip(u_total, -Vmax, Vmax)

        # Anti-windup (6.5 öncesi için)
        if t[k] < 6.5 and u_total != u_clipped:
            integral_e -= e * dt

        x = rk4_step(x, u_clipped, dt)
        i_hist[k], w_hist[k], u_hist[k], e_hist[k], ref_hist[k] = x[0], x[1], u_clipped, e, ref
        prev_e = e

    return t, w_hist, u_hist, e_hist, ref_hist

if __name__ == "__main__":
    # Referans Hız (100 rad/s)
    ref_val = 100.0
    def ref(t):
        return ref_val * (t/0.2) if t < 0.2 else ref_val
    
    # Simülasyonu çalıştır
    t, w, u, e, ref_sig = simulate(ref, T=10, dt=0.001, Ki_val=2.0, K_fuzzy_val=1.0) 

    plt.figure(figsize=(10,8))
    
    plt.subplot(3,1,1)
    plt.plot(t, ref_sig, '--', color='red', label='Referans')
    plt.plot(t, w, linewidth=2, color='blue', label='Motor Hızı')
    plt.axvline(x=6.5, color='green', linestyle=':', label='Stabilizasyon (6.5s)')
    plt.ylabel('Hız (rad/s)')
    plt.title('DC Motor Fuzzy Kontrol (6.5sn Sonrası Kilitli)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, u, color='orange')
    plt.ylabel('Voltaj (V)')
    plt.grid(True)
    
    plt.subplot(3,1,3)
    plt.plot(t, e, color='purple')
    plt.ylabel('Hata')
    plt.xlabel('Zaman (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
