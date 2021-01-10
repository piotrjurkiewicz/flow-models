#!/usr/bin/python3
"""
Creates General Mixture Models (GMM) fitted to flow records (requires `scipy`).
"""

import argparse
import collections
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .lib.mix import to_json
from .lib.data import load_data, LINE_NBINS, detect_x_value
from .lib.plot import plot_pdf, plot_cdf, plot_avg
from .lib.util import logmsg, measure_memory

X_VALUES = ['length', 'size']
Y_VALUES = ['flows', 'packets', 'octets']
EPS = np.finfo(np.float64).eps

def fit_mix(x, mix, ws=None, max_iter=100, max_pareto_w=None, cb=None):
    weights = np.zeros((x.shape[0], len(mix)))
    if ws is not None:
        avg_ws = np.average(ws)
    else:
        avg_ws = 1.0

    for i in range(max_iter):
        if cb:
            cb(mix, i)
        for k, comp in enumerate(mix):
            dist = getattr(scipy.stats, comp[1])
            pdf = dist.pdf(x, *comp[2]) * comp[0]
            z = x[:1]
            previous_z = z - 1.0
            previous_z[z == z.min()] = 0.0
            pdf[:1] = (dist.cdf(z, *comp[2]) - dist.cdf(previous_z, *comp[2])) * comp[0]
            weights[:, k] = pdf

        w_sum = np.sum(weights, axis=1)[:, None]
        w_sum[w_sum == 0.0] = 1.0
        weights = weights / w_sum

        if ws is not None:
            weights = weights * ws[:, None] / avg_ws

        max_upper = 0.0
        for comp in mix:
            if comp[1] == 'uniform':
                max_upper = max(max_upper, comp[2][1])
            if comp[1] == 'genpareto':
                max_upper = max(max_upper, comp[2][1] + 1.0)

        if not cb:
            logmsg(f"Iteration: {i}")

        for k, comp in enumerate(mix):
            frozen_loc = False
            if comp[1] == 'genpareto':
                frozen_loc = True
            old_params = comp[2]
            comp[2] = fit_comp(x, comp, weights, k, frozen_loc=frozen_loc)
            if comp[1] == 'lognorm' and comp[2][2] < max_upper:
                comp[2] = old_params

        for k, weight in enumerate(np.mean(weights, axis=0, keepdims=True).flat):
            mix[k][0] = weight
            if max_pareto_w and mix[k][1] == 'genpareto':
                mix[k][0] = min(weight, max_pareto_w)

    if cb:
        cb(mix, max_iter)
    return mix

def fit_comp(x, comp, weights, k, frozen=False, frozen_loc=False):
    if frozen:
        return comp[2]

    w = weights[:, k]

    if comp[1] == 'uniform':
        return comp[2]

    elif comp[1] in ('norm', 'lognorm'):
        if comp[1] == 'lognorm':
            x = np.log(x)
            mu = np.log(comp[2][2])
        else:
            mu = comp[0]
        sumw = np.sum(w)
        if not sumw > 0.0:
            sumw = EPS
        if not frozen_loc:
            mu = np.sum(w * x) / sumw
        sigma = np.sqrt(np.sum(w * ((x - mu) ** 2)) / sumw)
        if not sigma > 0.0:
            sigma = EPS
        if comp[1] == 'norm':
            return mu, sigma
        else:
            return sigma, 0, np.exp(mu)

    elif comp[1] == 'genpareto':
        beta = comp[2][1]
        if not frozen_loc:
            max_idx = np.argmax(weights, axis=1)
            max_x = x[max_idx == k]
            if len(max_x):
                beta = np.min(max_x)
        alpha = 1 / ((np.log(x) @ w) / np.sum(w) - np.log(beta))
        return 1 / alpha, beta, beta / alpha

    elif comp[1] == 'gamma':
        bhat = np.sum(w * x) / np.sum(w) / comp[2][0]

        def func(ah):
            return np.sum(w * (-np.log(bhat) - scipy.special.psi(ah) + np.log(x)))

        ahat = scipy.optimize.fsolve(func, np.array([1.0]))
        return ahat, 0, bhat

    elif comp[1] == 'weibull_min':
        lhat = (np.sum(w * (x ** comp[2][0])) / np.sum(w)) ** (1 / comp[2][0])

        def f1(kh):
            return 1 / kh * np.sum(w) + np.sum(w * np.log(x)) - np.log(lhat) * np.sum(w) - lhat ** (-kh) * np.sum(w * x ** kh * np.log(x / lhat))

        def df1(kh):
            return -1 / (kh ** 2) * np.sum(w) - np.sum(w * (x / lhat) ** 2 * (np.log(x / lhat)) ** 2)

        khat = scipy.optimize.zeros.newton(f1, 1.0, fprime=df1, maxiter=5, disp=False)

        return khat, 0, lhat

    else:
        raise ValueError

def initial_mix(mode, x):
    mix = []
    min_x = x.min()
    uniform_number = mode.get('U', 0)
    pareto_number = mode.get('P', 0)
    lognorm_number = mode.get('L', 0)
    geom = np.geomspace(x.min() + max(uniform_number, 1) - 1, x.max(), pareto_number + lognorm_number)
    ng = 0
    for n in range(uniform_number):
        mix.append([0.1, 'uniform', [0, min_x + n]])
    for n in range(pareto_number):
        if n == 0:
            mix.append([0.1, 'genpareto', [1.0, min_x + uniform_number - 1, 10.0]])
        else:
            mix.append([0.1, 'genpareto', [1.0, geom[ng], 10.0]])
            ng += 1
    for n in range(lognorm_number):
        mix.append([0.1, 'lognorm', [2.0, 0, geom[ng]]])
        ng += 1

    return mix

def fit(path, y_value, max_iter=100, initial=None, max_pareto_w=None, cb=None):
    path = pathlib.Path(path)
    logmsg(f'Processing: {path}')

    data = pd.read_csv(path, index_col=0, sep=',', low_memory=False, usecols=lambda n: not n.endswith('_ssq'))
    x = data.index.values
    weights = data[f'{y_value}_sum'].values

    if isinstance(initial, dict):
        mix = initial_mix(initial, x)
    else:
        mix = json.load(open(str(initial)))

    result_mix = fit_mix(x, mix, weights, max_iter=max_iter, max_pareto_w=max_pareto_w, cb=cb)
    return {'mix': result_mix, 'sum': np.sum(weights)}

def gui(**kwargs):
    import tkinter
    import tkinter.filedialog
    import tkinter.scrolledtext
    import tkinter.ttk
    from tkinter import N, E, S, W
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.backend_bases import key_press_handler

    mixtures = collections.defaultdict(dict)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.flatten()

    rv = [{}]

    root = tkinter.Tk()
    root.wm_title("flow-models-fit")

    style = tkinter.ttk.Style()
    style.theme_use('default')
    style.configure("black.Horizontal.TProgressbar", background='black')

    animate = tkinter.IntVar()
    y_var = tkinter.StringVar()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky=W+E+N+S, rowspan=10)

    toolbar = tkinter.Frame(master=root)
    toolbar.grid(row=10, column=0, sticky=W)
    toolbar = NavigationToolbar2Tk(canvas, toolbar)
    toolbar.update()

    dd = load_data([kwargs['path']])
    df = list(dd.values())[0]
    idx = np.unique(np.rint(np.geomspace(df.index.min(), df.index.max(), LINE_NBINS)).astype(int))

    first_plot = False

    def plot_data(*_):
        if not first_plot or y_var.get() and y_var.get() != kwargs['y_value']:
            if y_var.get():
                kwargs['y_value'] = y_var.get()
            for ff in pathlib.Path().glob('*.json'):
                mixtures[ff.stem] = json.load(open(str(ff)))
            x_val = detect_x_value(df.index)
            ax[0].clear()
            fig.sca(ax[0])
            plot_pdf(df, idx, x_val, kwargs['y_value'], mode={'line', 'mixture', 'comp', 'points'})
            ax[0].set_title('Probability Density Function (PDF)')
            ax[1].clear()
            fig.sca(ax[1])
            for what in Y_VALUES:
                plot_cdf(df, idx, x_val, what, mode={'line'})
            ax[1].set_title('Cumulative Distribution Function (CDF)')
            ax[2].clear()
            fig.sca(ax[2])
            plot_avg(df, idx, x_val, 'packets', mode={'line', 'mixture'})
            ax[2].set_title('Average flow length')
            ax[3].clear()
            fig.sca(ax[3])
            plot_avg(df, idx, x_val, 'octets', mode={'line', 'mixture'})
            ax[3].set_title('Average flow size')
            canvas.draw()

    y_var.trace('w', plot_data)
    plot_data()
    first_plot = True
    plots = []

    def plot_mixture(mix, iteration):
        bar['value'] = 100 * iteration / (kwargs['max_iter'] - 1)
        label_bar.configure(text=f"Progress: ({iteration}/{kwargs['max_iter']})")
        txt_mix.delete('1.0', tkinter.END)
        txt_mix.insert(tkinter.INSERT, to_json(mix))
        root.update_idletasks()
        if animate.get() and iteration % 8 == 0 or iteration == kwargs['max_iter']:
            mixtures[kwargs['y_value']]['mix'] = mix
            for line in plots:
                line.remove()
            del plots[:]
            fig.sca(ax[0])
            x_val = detect_x_value(df.index)
            plots.extend(plot_pdf(mix, idx, x_val, kwargs['y_value'], mode={'mixture', 'comp'}))
            fig.sca(ax[1])
            plots.extend(plot_cdf(mix, idx, x_val, kwargs['y_value'], mode={'mixture'}))
            fig.sca(ax[2])
            plots.extend(plot_avg(mixtures, idx, x_val, 'packets', mode={'mixture'}) or [])
            fig.sca(ax[3])
            plots.extend(plot_avg(mixtures, idx, x_val, 'octets', mode={'mixture'}) or [])
            canvas.draw()

    def on_key_press(event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect("key_press_event", on_key_press)

    def _quit():
        root.quit()  # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def _save():
        save_dir = tkinter.filedialog.askdirectory()
        save(pathlib.Path(save_dir) / kwargs['y_value'], rv[0], kwargs)

    def _fit():
        kwargs['max_iter'] = int(entry_iterations.get())
        kwargs['initial']['U'] = int(spin_uniform.get())
        kwargs['initial']['L'] = int(spin_lognorm.get())
        kwargs['initial']['P'] = int(spin_pareto.get())
        kwargs['y_value'] = combo.get()
        kwargs['y_value'] = combo.get()
        rv[0] = fit(**kwargs, cb=plot_mixture)

    tkinter.Label(master=root, text="Parameters:").grid(column=1, row=0, sticky=W)

    tkinter.Label(master=root, text="Iterations:").grid(column=1, row=1, sticky=W)
    entry_iterations = tkinter.Entry(master=root, width=6)
    entry_iterations.grid(column=2, row=1, sticky=E)

    tkinter.Label(master=root, text="Number of uniforms:").grid(column=1, row=2, sticky=W)
    spin_uniform = tkinter.Spinbox(master=root, from_=0, to=100, width=6)
    spin_uniform.grid(column=2, row=2, sticky=E)

    tkinter.Label(master=root, text="Number of lognorms:").grid(column=1, row=3, sticky=W)
    spin_lognorm = tkinter.Spinbox(master=root, from_=0, to=100, width=6)
    spin_lognorm.grid(column=2, row=3, sticky=E)

    tkinter.Label(master=root, text="Number of Pareto:").grid(column=1, row=4, sticky=W)
    spin_pareto = tkinter.Spinbox(master=root, from_=0, to=100, width=6)
    spin_pareto.grid(column=2, row=4, sticky=E)

    tkinter.Label(master=root, text="Y value:").grid(column=1, row=5, sticky=W)
    combo = tkinter.ttk.Combobox(master=root, textvar=y_var, values=Y_VALUES)
    combo.grid(column=2, row=5, sticky=E)

    check_animate = tkinter.Checkbutton(master=root, text="Animate", variable=animate)
    check_animate.grid(column=2, row=6, sticky=E)

    label_bar = tkinter.Label(master=root, text="Progress:")
    label_bar.grid(column=1, row=7, sticky=W)
    bar = tkinter.ttk.Progressbar(master=root, length=200, style='black.Horizontal.TProgressbar')
    bar.grid(column=2, row=7, sticky=E)

    tkinter.Label(master=root, text="Fitted mixture:").grid(column=1, row=8, sticky=W)
    txt_mix = tkinter.scrolledtext.ScrolledText(master=root, width=96, height=24)
    txt_mix.grid(column=1, row=9, sticky=E+W, columnspan=2)

    button_frame = tkinter.Frame(master=root)
    button_frame.grid(column=1, row=10, sticky=E, columnspan=2)

    button_quit = tkinter.Button(master=button_frame, text="Quit", command=_quit)
    button_quit.grid(row=0, column=2)
    button_save = tkinter.Button(master=button_frame, text="Save", command=_save)
    button_save.grid(row=0, column=1)
    button_fit = tkinter.Button(master=button_frame, text="Fit", command=_fit)
    button_fit.grid(row=0, column=0)

    entry_iterations.insert(0, kwargs['max_iter'])
    spin_uniform.delete(0)
    spin_uniform.insert(0, kwargs['initial'].get('U', 0))
    spin_lognorm.delete(0)
    spin_lognorm.insert(0, kwargs['initial'].get('L', 0))
    spin_pareto.delete(0)
    spin_pareto.insert(0, kwargs['initial'].get('P', 0))
    combo.current(combo['values'].index(kwargs['y_value']))

    tkinter.mainloop()
    return rv

def save(path, mix, args):
    logmsg(f'Saving: {path}')
    pathlib.Path(str(path) + '.json').write_text(to_json(**mix) + '\n')
    pathlib.Path(str(path) + '.mode').write_text(json.dumps({k: v for k, v in args.items() if k != 'path'}) + '\n')

def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument('file', help='input histogram file')
    p.add_argument('-y', default='flows', choices=Y_VALUES, help='y axis value')
    p.add_argument('-i', default=100, type=int, help='number of iterations')
    p.add_argument('-U', type=int, help='number of uniform distributions')
    p.add_argument('-P', type=int, help='number of Pareto distributions')
    p.add_argument('-L', type=int, help='number of lognorm distributions')
    p.add_argument('--mpw', type=float, help='maximum pareto weight')
    p.add_argument('--initial', help='initial mixture', default={})
    p.add_argument('--interactive', action='store_true', help='interactive')
    p.add_argument('--test', action='store_true', help='test fitting')
    p.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    return p

def main():
    app_args = parser().parse_args()

    mode = {}
    for key, val in vars(app_args).items():
        if key.isupper() and val:
            mode[key] = val

    if mode:
        initial = mode
    else:
        initial = app_args.initial

    with measure_memory(app_args.measure_memory):
        args = dict(path=app_args.file, y_value=app_args.y, max_iter=app_args.i, initial=initial, max_pareto_w=app_args.mpw)
        if app_args.interactive:
            gui(**args)
        else:
            new_mix = fit(**args)
            save(pathlib.Path() / args['y_value'], new_mix, args)


if __name__ == '__main__':
    main()
