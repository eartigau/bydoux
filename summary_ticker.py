import numpy as np
import matplotlib.pyplot as plt
import bydoux_tools as bt

ticker = 'EBNK.TO'
fontsize = 6  # Variable for controlling fontsize throughout

tbl = bt.read_quotes(ticker)
tbl['mjd'] = np.array(tbl['mjd'], dtype = int)
spx= bt.read_quotes('^SPX')
info = bt.get_info(ticker)

# Setup a 8.5x11" page with 2x4 grid of subplots
fig, axes = plt.subplots(4, 2, figsize=(8.5, 11))
fig.suptitle(f'Analysis for {ticker}', fontsize=fontsize+3)

# Flatten axes array for easier indexing
axes = axes.flatten()

axes[0].plot(tbl['plot_date'],10000*np.exp(tbl['log_close']-tbl['log_close'][0]))
axes[0].set_title('Croissance de 10 000$', fontsize = fontsize)
axes[0].set_ylabel('Valeur ($)', fontsize=fontsize)

# axes[1] as text box - fill entire area with beige background
axes[1].set_facecolor('#F5F5DC')  # Set background to beige
axes[1].axis('off')  # Turn off axes for text box

info_text = f"""Ticker: {ticker}
Nom complet: {info['longName']}
Prix initial: {tbl['Close'][0]:.2f}$
Prix final: {tbl['Close'][-1]:.2f}$
Rendement total: {((tbl['Close'][-1]/tbl['Close'][0])-1)*100:.1f}%
Rendement annualisé: {(np.exp(tbl['log_close'][-1]-tbl['log_close'][0])**(365.25/(tbl['mjd'][-1]-tbl['mjd'][0]))-1)*100:.1f}%"""

# Text fills the entire axes area without bbox
axes[1].text(0.05, 0.95, info_text, transform=axes[1].transAxes, fontsize=fontsize,
             verticalalignment='top', horizontalalignment='left')

# Apply smaller fontsize to all subplots
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-1)
    if ax != axes[1]:  # Don't add grid to text box
        ax.grid()

plt.tight_layout()
plt.show()