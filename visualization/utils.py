import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def decode_fnc(x, y, model):
    with torch.no_grad():
        z = torch.Tensor([[x, y]])
        x_decoded = model.decode(z).detach().numpy()
    fnc = np.zeros((53, 53))
    fnc[np.triu_indices(53, 1)] = x_decoded[0]
    fnc[np.tril_indices(53, -1)] = fnc.T[np.tril_indices(53, -1)]
    fnc[np.diag_indices(53)] = 1
    return fnc

def plot_fnc(fnc, ax, title=None, title_color='k', title_fontsize=44, show_xticks=False, xticks_fontsize=24, show_colorbar=False):
    network_ind = [4.5, 6.5, 15.5, 24.5, 41.5, 48.5]
    img = ax.imshow(fnc, vmin=-1, vmax=1, cmap="seismic")
    if show_colorbar:
        bar = plt.colorbar(img)
    if title:
        ax.set_title(title, fontsize=title_fontsize, color=title_color)
    for j in network_ind:
        ax.axhline(y=j, linewidth=1, color='k')
        ax.axvline(x=j, linewidth=1, color='k')
    if show_xticks:
        network_ind_ = [-0.5, 4.5, 6.5, 15.5, 24.5, 41.5, 48.5, 52.5]
        network_label_ind = [(network_ind_[i+1]+network_ind_[i])/2 for i in range(len(network_ind_)-1)]
        network_list = ['SC', 'AU', 'SM', 'VI', 'CC', 'DM', 'CB']
        ax.set_xticks([])
        ax.set_yticks(network_label_ind, network_list, fontsize=xticks_fontsize)
    else:
        ax.tick_params(left = False,
                    bottom = False,
                    labelleft = False,
                    labelbottom = False)

def calculate_mse(fnc_orig, fnc_vae):
    n = 15
    space = 10
    matrix_size = 53
    orig = np.zeros((n, n, 1378))
    vae = np.zeros((n, n, 1378))
    mse = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub_fnc_orig = fnc_orig[i * (matrix_size+space)+space//2 : i * (matrix_size+space)+space//2+matrix_size, j * (matrix_size+space)+space//2 : j * (matrix_size+space)+space//2+matrix_size]
            sub_fnc_vae = fnc_vae[i * (matrix_size+space)+space//2 : i * (matrix_size+space)+space//2+matrix_size, j * (matrix_size+space)+space//2 : j * (matrix_size+space)+space//2+matrix_size]
            if np.sum(sub_fnc_orig) == 0:
                mse[i,j] = np.nan
            else:
                orig[i,j,:] = sub_fnc_orig[np.triu_indices(matrix_size, 1)]
                vae[i,j,:] = sub_fnc_vae[np.triu_indices(matrix_size, 1)]
                mse[i,j] = np.mean((orig[i,j,:]-vae[i,j,:])**2)
    return mse

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def compute_x_position(pvalue):
    offset = -0.025
    if pvalue <= 0.0001:
        return 0.8 + offset
    elif pvalue <= 0.001:
        return 0.85 + offset
    elif pvalue <= 0.01:
        return 0.9 - 0.01
    elif pvalue <= 0.05:
        return 1 + offset
    return 0.9 + offset

def plot_dwell_time(ax, mean_pt, ste_pt, mean_hc, ste_hc, pval, color_pt, label_pt, title_label, n_state=5, title_fontsize=16, label_fontsize=14, tick_fontsize=12):
    ax.plot(np.arange(1,n_state+1), mean_pt, linewidth=2, linestyle='-.', marker='o', markersize=7, color=color_pt)
    ax.fill_between(np.arange(1,n_state+1), mean_pt-ste_pt, mean_pt+ste_pt, alpha=0.4, color=color_pt, label=label_pt)
    ax.plot(np.arange(1,n_state+1), mean_hc, linewidth=2, linestyle='-.', marker='o', markersize=7, color='cornflowerblue')
    ax.fill_between(np.arange(1,n_state+1), mean_hc-ste_hc, mean_hc+ste_hc, alpha=0.4, color='cornflowerblue', label='CTR')

    for idx, p in enumerate(pval):
        x_position = idx+compute_x_position(p)
        y_position = np.max([mean_pt[idx]+ste_pt[idx], mean_hc[idx]+ste_hc[idx]]) + 4.5
        asterisk = convert_pvalue_to_asterisks(p*n_state) # correction for multiple comparison
        ax.text(x=x_position, y=y_position, s=asterisk)

    ax.set_title(title_label, fontsize=title_fontsize)
    ax.set_xticks(list(range(1,n_state+1)), [str(i) for i in range(1,n_state+1)])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xlabel("State", fontsize=label_fontsize)
    ax.set_ylabel("Dwell time (windows)", fontsize=label_fontsize)
    ax.set_xlim([0.5,n_state+0.5])
    ax.set_ylim([-5,135])
    ax.legend(loc='best', fontsize=12)

def plot_transition_matrix(ax, transition_matrix, p_value, title_label, n_state=5, title_fontsize=16, label_fontsize=14, tick_fontsize=12):
    mask = p_value < 0.05
    np.fill_diagonal(mask,1)
    label = np.arange(1,n_state+1)
    pmask = 1 - (p_value < 0.05)
    sns.heatmap(transition_matrix,mask=mask,cmap="magma",annot=True,fmt=".2f",vmin=0,vmax=1,xticklabels=label,yticklabels=label,ax=ax,cbar=False,annot_kws={"size": tick_fontsize})
    sns.heatmap(transition_matrix,mask=pmask,cmap="magma",annot=True,fmt=".2f",vmin=0,vmax=1,xticklabels=label,yticklabels=label,ax=ax,annot_kws={"size": tick_fontsize, "style": "italic", "weight": "bold"})
    ax.set_title(title_label, fontsize=title_fontsize)
    ax.set_xlabel("To state", fontsize=label_fontsize)
    ax.set_ylabel("From state", fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

def compute_arrow(centroid, anchor, alpha, beta):
    if anchor[0] == 0 and anchor[1] == 0:
        return centroid[0]*alpha, centroid[1]*alpha, centroid[0]*(1-alpha)*beta, centroid[1]*(1-alpha)*beta
    elif anchor[0] == 0 and anchor[1] == 1:
        return centroid[0]*alpha, (1-centroid[1])*(1-alpha)+centroid[1], centroid[0]*(1-alpha)*beta, -(1-centroid[1])*(1-alpha)*beta
    elif anchor[0] == 1 and anchor[1] == 0:
        return 1-alpha+alpha*centroid[0], centroid[1]*alpha, (1-alpha)*(centroid[0]-1)*beta, centroid[1]*(1-alpha)*beta
    elif anchor[0] == 1 and anchor[1] == 1:
        return 1-alpha+alpha*centroid[0], 1-alpha+alpha*centroid[1], -(1-alpha+alpha*centroid[0]-centroid[0])*beta, -(1-alpha+alpha*centroid[1]-centroid[1])*beta

def mix_colors(color1, color2, ratio):
    rgb1 = np.array(color1)
    rgb2 = np.array(color2)
    return (1 - ratio) * rgb1 + ratio * rgb2