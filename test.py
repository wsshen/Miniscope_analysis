import numpy as np
import matplotlib.pyplot as plt

# time constants

tau = 0.05
tau_delta = 0.02
alpha_v = 0.5
alpha_g = 0.05
alpha_h = 0.02
alpha_sigma_g = 0.05
alpha_sigma_h = 0.1

l = 0.9

n_trials = 100
dt = 0.001
t_duration = 2.4

interval = 0.2
n_timesteps = int(t_duration/dt)
t = np.linspace(0,t_duration,n_timesteps)
sigma_a = 1

# initial conditions
h0 = 0
q0 = 0.1
v0 = 0.1
r0 = 1
sigma_g0 = 1
sigma_h0 =100

# initialize matries
a = np.zeros((n_trials,len(t)))
delta_g = np.zeros((n_trials,len(t)))
delta_h = np.zeros((n_trials,len(t)))
q = np.zeros((n_trials,len(t)))
h = np.zeros((n_trials,len(t)))
sigma_g = np.zeros((n_trials,len(t)))
sigma_h = np.zeros((n_trials,len(t)))
v = np.zeros((n_trials,len(t)))

delta_v = np.zeros((n_trials,len(t)))
r = np.zeros((n_trials,len(t)))
r0 = np.zeros((n_trials,len(t)))
r0[:,2001:-1] = 2

num_intervals = 5

w = np.zeros((n_trials,len(t),num_intervals))
e = np.zeros((n_trials,len(t),num_intervals))
# e_intervals = np.zeros((n_trials,num_intervals,len(t)))
s = np.zeros((n_trials,len(t),num_intervals))

v_trace = np.zeros((n_trials,len(t)))

for i in range(num_intervals):
    s[:,1001+200*i:1001+200*(i+1),i] = 1
    # s[:,1001:2001,:] = 1
sigma_g[0,0] = sigma_g0
sigma_h[0,0] = sigma_h0
q[0,0] = q0
h[0,0] = h0

for i_trials in range(0, n_trials - 1, 1):
    current_interval = -1
    for i_time in range(0, n_timesteps - 1, 1):

        if i_time > 1000 and i_time < 2000 and i_time % 200 == 1:
            current_interval += 1
            # print(i_time,current_interval)

        # v_intervals[i_trials,current_interval+1] = np.dot(w[i_trials,current_interval+1],s[i_trials,current_interval+1])
        v_trace[i_trials, i_time] = np.dot(w[i_trials, i_time, :], s[i_trials, i_time, :])
        v_trace[i_trials, i_time + 1] = np.dot(w[i_trials, i_time + 1, :], s[i_trials, i_time + 1, :])
        if i_trials == 20 and i_time > 1800:
            print(i_trials, i_time, v_trace[i_trials, i_time], v_trace[i_trials, i_time + 1])

        v[i_trials, i_time + 1] = v[i_trials, i_time] + dt * 1 / tau * (
                    v_trace[i_trials, i_time + 1] - v[i_trials, i_time])
        # if i_trials == 20 and i_time>1800:
        #     print(i_time,v[i_trials,i_time+1],v_trace[i_trials,i_time+1])

        # v[i_trials,i_time+1] = v[i_trials,i_time] + dt * 1/tau * (w[i_trials,i_time] * s[i_trials,i_time] - v[i_trials,i_time])
        r[i_trials, i_time + 1] = r[i_trials, i_time] + dt * 1 / tau * (r0[i_trials, i_time] - r[i_trials, i_time])

        delta_v[i_trials, i_time + 1] = delta_v[i_trials, i_time] + dt * 1 / tau_delta * (
                    r[i_trials, i_time + 1] + v[i_trials, i_time + 1] -
                    v[i_trials, i_time - 200] - delta_v[i_trials, i_time])

        for i_interval in range(num_intervals):
            if i_time - 2 >= 0:
                e[i_trials, i_time + 1, i_interval] = e[i_trials, i_time, i_interval] + dt * 1 / tau * (
                            l * e[i_trials, i_time - int((interval + 3 * tau) / dt), i_interval] +
                            s[i_trials, i_time, i_interval] - e[i_trials, i_time, i_interval])

            w[i_trials, i_time + 1, i_interval] = w[i_trials, i_time, i_interval] + dt * alpha_v * delta_v[
                i_trials, i_time + 1] * e[
                                                      i_trials, i_time, i_interval]  # e_intervals[i_trials,current_interval,i_time+1]

            if w[i_trials, i_time + 1, i_interval] < 0:
                w[i_trials, i_time + 1, i_interval] = 0
        # if i_time == 2000:
        #     print(w[i_trials,current_interval],current_interval)

        # update goal and habit systems
        a[i_trials, i_time + 1] = a[i_trials, i_time] = dt * 1 / tau * (
                    delta_g[i_trials, i_time] * q[i_trials, i_time] * s[i_trials, i_time, current_interval] +
                    (h[i_trials, i_time] * s[i_trials, i_time, current_interval] - a[i_trials, i_time]) / sigma_h[
                        i_trials, i_time])
        delta_g[i_trials, i_time + 1] = delta_g[i_trials, i_time] + dt * 1 / tau_delta * (
                    r[i_trials, i_time + 1] + v[i_trials, i_time + 1] -
                    a[i_trials, i_time + 1] * q[i_trials, i_time] * s[i_trials, i_time, current_interval] -
                    sigma_g[i_trials, i_time] * delta_g[i_trials, i_time])
        delta_h[i_trials, i_time + 1] = delta_h[i_trials, i_time] + dt * 1 / tau_delta * (
                    a[i_trials, i_time + 1] - h[i_trials, i_time] * s[i_trials, i_time, current_interval] -
                    delta_h[i_trials, i_time])
        q[i_trials, i_time + 1] = q[i_trials, i_time] + dt * alpha_g * delta_g[i_trials, i_time + 1] * a[
            i_trials, i_time + 1] * s[i_trials, i_time, current_interval]
        h[i_trials, i_time + 1] = h[i_trials, i_time] + dt * alpha_h * delta_h[i_trials, i_time + 1] * s[
            i_trials, i_time, current_interval]
        sigma_g[i_trials, i_time + 1] = sigma_g[i_trials, i_time] + dt * alpha_sigma_g * (
                    delta_g[i_trials, i_time + 1] * delta_g[i_trials, i_time + 1] * sigma_g[i_trials, i_time] * sigma_g[
                i_trials, i_time] -
                    sigma_g[i_trials, i_time])
        sigma_h[i_trials, i_time + 1] = sigma_h[i_trials, i_time] + dt * alpha_sigma_h * (
                    delta_h[i_trials, i_time + 1] * delta_h[i_trials, i_time + 1] -
                    sigma_h[i_trials, i_time])

    w[i_trials + 1, 0, :] = w[i_trials, -1, :]
    q[i_trials + 1, 0] = q[i_trials, -1]
    h[i_trials + 1, 0] = h[i_trials, -1]
    sigma_g[i_trials + 1, 0] = sigma_g[i_trials, -1]
    sigma_h[i_trials + 1, 0] = sigma_h[i_trials, -1]