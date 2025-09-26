# generic imports 

from bydoux_tools import read_google_sheet_csv, get_info
import warnings
import numpy as np
from scipy.optimize import minimize
import bydoux_tools as bt
from astropy.time import Time,TimeDelta
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF

# Parameters for the code that may change
force = True
verbose = True

t0 = Time('2023-03-01', format='iso').mjd
t1 = Time.now().mjd

# Fetch the disnat summary table of my accounts
tbl = bt.get_disnat_summary()
tbl = tbl[np.argsort(tbl['mjd'])]

# Construct a time span for the analysis (daily steps)
time_span = np.arange(t0, t1, 1)
tbl0 = Table()
tbl0['mjd'] = time_span

# Read S&P 500 and CAD/USD FX rate data
sp500 = bt.read_quotes('^SPX', force=force)
cadusd = bt.read_quotes('CADUSD=X', force=force)

# Interpolate CAD/USD and SP500 values onto the analysis time grid
tbl0['CADUSD'] = np.interp(tbl0['mjd'], cadusd['mjd'], cadusd['Close'])
tbl0['SP500'] = np.interp(tbl0['mjd'], sp500['mjd'], sp500['Close'])
tbl0['N_SP500'] = 0.0
tbl0['CONTRIBUTIONS'] = 0.0

# Also interpolate SP500 onto the transaction table for reference
tbl['SP500'] = np.interp(tbl['mjd'], sp500['mjd'], sp500['Close'])

# Compute running sum of contributions (COTISATION and TRANSFERT REÇU in CAD)
for i in tqdm(range(len(tbl)),leave = False, desc='Computing contributions'):
    if tbl['Devise du prix'][i] != 'CAN':
        continue
    if tbl['Type de transaction'][i] in ['COTISATION', 'TRANSFERT REÇU']:
        g = tbl0['mjd'] > tbl['mjd'][i]
        tbl0['CONTRIBUTIONS'][g] += tbl["Montant de l'opération"][i]

# Get the symbol of all actions bought or sold: 'Type de transaction' is 'ACHAT' or 'VENTE'
flag = [
    ('ACHAT' in x or 'VENTE' in x)
    for x in tbl['Type de transaction']
]
symbols = np.unique(tbl[flag]['Symbole'])

quotes_dict = {}
problems = []
symbols_query = np.zeros_like(symbols, dtype='U50')
for isymbol, sym in tqdm(enumerate(symbols), total=len(symbols), desc='Fetching quotes'):
    sym2 = sym.split('-')[0]
    sym2 = sym2.split('.')[0]
    quotes = None
    try:
        quotes = bt.read_quotes(sym2+'.TO', force=force)
        symbols_query[isymbol] = sym2+'.TO'
        info_ticker = bt.get_info(sym2+'.TO')
    except Exception as e:
        if verbose:
            bt.printc(f"Error reading {sym2}.TO: {e}")
    if quotes is None:
        quotes = bt.read_quotes(sym2, force=force)
        symbols_query[isymbol] = sym2
    if quotes is None:
        problems.append(sym)
        continue
    quotes_dict[sym] = quotes
    quotes['mjd'] = np.round(quotes['mjd'])
    val = np.interp(tbl0['mjd'], quotes['mjd'], quotes['Close'])
    val_close_dividend = np.interp(tbl0['mjd'], quotes['mjd'], quotes['Close_dividends'])
    val_div_yield = np.interp(tbl0['mjd'], quotes['mjd'], quotes['Dividends'])
    tbl0[sym] = val
    tbl0[sym+'_dividend'] = val_close_dividend
    tbl0[sym+'_div_yield'] = val_div_yield
    tbl0[sym+'_quantity'] = 0
    tbl0[sym+'_val'] = 0.0

tbl0['Month'] = '0000-00'
for i in range(len(tbl0)):
    mjd = tbl0['mjd'][i]
    t = Time(mjd, format='mjd')
    tbl0['Month'][i] = t.iso[:7]

# For each transaction, update the running quantity and value for each symbol
for i in range(len(tbl)):
    if ('ACHAT' in tbl['Type de transaction'][i]) or ('VENTE' in tbl['Type de transaction'][i]):
        sym = tbl['Symbole'][i]
        g = tbl0['mjd'] > tbl['mjd'][i]
        tbl0[sym+'_quantity'][g] += tbl['Quantité'][i]
        if '-U' in sym:
            tbl0[sym+'_val'][g] += tbl['Quantité'][i] * tbl0[sym][g] / tbl0['CADUSD'][g]
        else:
            tbl0[sym+'_val'][g] += tbl['Quantité'][i] * tbl0[sym][g]

tbl0['val_total'] = 0.0
for sym in symbols:
    tbl0['val_total'] += tbl0[sym+'_val']

percent_gains = [-10, -5, 5, 10, 15, 20,25]
for i in range(1, len(tbl0)):
    tbl0['N_SP500'][i] = tbl0['N_SP500'][i-1]
    if np.round(tbl0['CONTRIBUTIONS'][i]) != np.round(tbl0['CONTRIBUTIONS'][i-1]):
        cont = (tbl0['CONTRIBUTIONS'][i] - tbl0['CONTRIBUTIONS'][i-1])  
        nsp500 = cont / tbl0['SP500'][i]
        tbl0['N_SP500'][i] += nsp500

tbl0['val_SP500'] = tbl0['N_SP500'] * tbl0['SP500']

map_gain = np.zeros([len(tbl0), len(percent_gains)], dtype=float)
for ipg, pg in enumerate(percent_gains):
    tbl0[f'growth_{pg}_percent'] = 1.0
    for i in range(1, len(tbl0)):
        tbl0[f'growth_{pg}_percent'][i] = (
            tbl0[f'growth_{pg}_percent'][i-1] * ((1 + pg/100.0) ** (1.0/365.0))
        )
        if np.round(tbl0['CONTRIBUTIONS'][i]) != np.round(tbl0['CONTRIBUTIONS'][i-1]):
            tbl0[f'growth_{pg}_percent'][i] += (tbl0['CONTRIBUTIONS'][i] - tbl0['CONTRIBUTIONS'][i-1])
    map_gain[:, ipg] = tbl0[f'growth_{pg}_percent']

tbl0['effective_growth'] = 0.0
tbl0['effective_growth_sp500'] = 0.0
for i in range(len(tbl0)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        fit = np.polyfit(map_gain[i], percent_gains, 2)
    tbl0['effective_growth'][i] = np.polyval(fit, tbl0['val_total'][i])
    tbl0['effective_growth_sp500'][i] = np.polyval(fit, tbl0['val_SP500'][i])
bad = (tbl0['effective_growth'] < -30) | (tbl0['effective_growth'] > 100)
tbl0['effective_growth'][bad] = np.nan
bad_sp500 = (tbl0['effective_growth_sp500'] < -30) | (tbl0['effective_growth_sp500'] > 100)
tbl0['effective_growth_sp500'][bad_sp500] = np.nan

# --- PDF output section ---
pdf_pages = []
text_chunks = []

def add_fig_to_pdf(fig):
    pdf_pages.append(fig)

def add_text_to_pdf(text):
    text_chunks.append(text)

# Plot 1: Portfolio value and synthetic growth scenarios
future_projection = 31*2
fig1, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 16), sharex=True)
tbl0['time_plot'] = Time(tbl0['mjd'], format='mjd').datetime
for i in range(2):
    ref = 0 if i == 0 else tbl0['CONTRIBUTIONS']
    ax[i].plot(tbl0['time_plot'], tbl0['val_total'] - ref, label='Portefeuille', color='blue')
    norm = mcolors.Normalize(vmin=min(percent_gains), vmax=max(percent_gains))
    cmap = plt.colormaps['plasma']
    if i ==1:
        for pg in percent_gains:
            color = cmap(norm(pg))
            ax[i].plot(tbl0['time_plot'], tbl0[f'growth_{pg}_percent'] - ref, label=f'{pg}%', color=color)
            ref0=0
            if i != 0:
                ref0 = ref[-1]
            for dt in range(0,int(future_projection),2):
                gg = (1+pg/100.0)**(dt/365.0)
                ax[i].plot(tbl0['time_plot'][-1]+(dt*1.0) * TimeDelta(1, format='jd').datetime, tbl0['val_total'][-1]*gg - ref0,'.', color=color)
    else:
        ax[0].plot(tbl0['time_plot'], tbl0['CONTRIBUTIONS'], label='Portefeuille', color='orange')
    ax[i].plot(tbl0['time_plot'], tbl0['val_SP500'] - ref, label='SP500', color='black', linestyle='--')
maxi = np.max(tbl0[f'growth_{pg}_percent'] - ref)
ax[0].set_ylim([00000, 200000])
ax[1].set_ylim([0,  (maxi * 1.2)])
ax[0].legend()
ax[1].legend(loc = 'lower left')
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel('Valeur du portefeuille (CAD)')
ax[1].set_ylabel('Croissance du portefeuille (CAD)')
ax[0].set_xlim(Time('2024-04-01', format='iso').datetime, Time.now().datetime+TimeDelta(future_projection,format='jd').datetime)
add_fig_to_pdf(fig1)

# Plot 2: Log growth
latest_portfolio = []
vv = np.zeros(len(tbl0), dtype=float)
total_last = 0.0
all_vv = []
all_valid_sym = []
for isym, sym in enumerate(symbols):
    last_quantity = tbl0[sym+'_quantity'][-1]
    if last_quantity == 0:
        continue
    last_val = tbl0[sym+'_val'][-1]
    total_last += last_val
    if '-U' in sym:
        last_val /= tbl0['CADUSD'][-1]
    last_val = np.round(last_val, 2)
    sym = sym.replace('-U', '').replace('.TO', '')
    latest_portfolio.append((sym, last_quantity, last_val))
    tmp = np.array(tbl0[sym+'_dividend'])
    tmp/=tmp[0]
    tmp*= last_val
    all_vv.append(tmp*1.0)
    all_valid_sym.append(sym)
    vv += tmp
all_vv = np.array(all_vv).T
all_valid_sym = np.array(all_valid_sym)

lvv = np.log(vv)
vv_sp500 = tbl0['SP500'] 
lvv_sp500 = np.log(vv_sp500)
fit_growth = np.polyfit(tbl0['mjd']/364.24,lvv, 1)
std_growth = np.std(lvv - (fit_growth[0] * tbl0['mjd']/364.24 + fit_growth[1]))
std_sp500 = np.std(lvv_sp500 - (np.polyfit(tbl0['mjd']/364.24,lvv_sp500, 1)[0] * tbl0['mjd']/364.24 + np.polyfit(tbl0['mjd']/364.24,lvv_sp500, 1)[1]))

fig2, ax = plt.subplots(figsize=(15, 12), nrows = 2,ncols=1)
ax[0].plot(tbl0['time_plot'], np.log(vv)-np.mean(np.log(vv)), label='Portefeuille', color='blue')
ax[0].plot(tbl0['time_plot'], np.log(vv_sp500)-np.mean(np.log(vv_sp500)), label='SP500', color='black', linestyle='--')
ax[0].grid()
ax[1].plot(tbl0['time_plot'], np.log(vv) - np.polyval(np.polyfit(tbl0['mjd']/364.24, np.log(vv), 1), tbl0['mjd']/364.24), label='Portefeuille', color='blue')
ax[1].plot(tbl0['time_plot'], np.log(vv_sp500) - np.polyval(np.polyfit(tbl0['mjd']/364.24, np.log(vv_sp500), 1), tbl0['mjd']/364.24), label='SP500', color='black', linestyle='--')
ax[1].grid()
add_fig_to_pdf(fig2)

# Plot 3: Scatter diff
vmin = -1.0
vmax = 1.0
nbin = 30
nroll = 91
diff_portefeuille = lvv - np.roll(lvv, nroll)
diff_sp500 = lvv_sp500 - np.roll(lvv_sp500, nroll)
diff_portefeuille = diff_portefeuille[nroll:]
diff_sp500 = diff_sp500[nroll:]
moy_portefeuille, med_portefeuille, sig_portefeuille = np.nanmean(diff_portefeuille)*(365/nroll), np.nanmedian(diff_portefeuille)*(365/nroll), np.nanstd(diff_portefeuille)
moy_sp500, med_sp500, sig_sp500 = np.nanmean(diff_sp500)*(365/nroll), np.nanmedian(diff_sp500)*(365/nroll), np.nanstd(diff_sp500)
tt = tbl0['time_plot'][nroll:]
tt_numeric = [t.timestamp() for t in tt]
norm = mcolors.Normalize(vmin=min(tt_numeric), vmax=max(tt_numeric))
cmap = cm.get_cmap('viridis')
colors = [cmap(norm(t)) for t in tt_numeric]
fig3, ax = plt.subplots(figsize=(15, 8))
ax.scatter(diff_portefeuille*(365/nroll), diff_sp500*(365/nroll), c=colors, s=60, edgecolor='k', label='Portefeuille vs SP500')
ax.plot([-1, 1], [-1, 1], 'r--', label='1:1')
mini = np.minimum(np.min(diff_portefeuille*(365/nroll)), np.min(diff_sp500*(365/nroll)))
maxi = np.maximum(np.max(diff_portefeuille*(365/nroll)), np.max(diff_sp500*(365/nroll)))
ax.set_xlim([1.01*mini, 1.01*maxi])
ax.set_ylim([1.01*mini, 1.01*maxi])
ax.set_ylabel('SP500')
ax.set_xlabel('Portefeuille')
ax.set_title('Différences de croissance effective (coloré par date)')
ax.grid()
ax.legend()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(tt_numeric)
cbar = plt.colorbar(sm, ax=ax)
date_ticks = np.linspace(min(tt_numeric), max(tt_numeric), num=6)
cbar.set_ticks(date_ticks)
from datetime import datetime
cbar.set_ticklabels([datetime.fromtimestamp(dt).strftime('%Y-%m-%d') for dt in date_ticks])
cbar.set_label('Date')
add_fig_to_pdf(fig3)

# Plot 4: Growth rate vs std for each symbol
current_symbols = []
for sym in symbols:
    v = tbl0[sym+'_val'][-1]
    if v ==0 or not np.isfinite(v) or np.abs(v)<1e-2:
        continue
    current_symbols.append(sym)
durations = []
for key in current_symbols:
    quotes = quotes_dict[key]
    duration = (quotes['mjd'][-1] - quotes['mjd'][0])/365.24
    durations.append(duration)
durations = np.array(durations)
norm = plt.Normalize(vmin=np.min(durations), vmax=np.max(durations))
cmap = cm.get_cmap('viridis')
fig4, ax = plt.subplots(figsize=(18,8), nrows = 1, ncols=2)
for idx, key in enumerate(current_symbols):
    log_div = np.log(tbl0[key+'_dividend'])
    fit = np.polyfit(tbl0['mjd']/364.24, log_div, 1)
    std = np.std(log_div - (fit[0] * tbl0['mjd']/364.24 + fit[1]))
    ax[0].plot(100*fit[0], 100*std, 'k.',alpha=0.5, markersize=12)
    slope0 = fit[0]
    sig0 = std
    quotes = quotes_dict[key]
    fit = np.polyfit(quotes['mjd']/365.24, quotes['log_close'], 1)
    std = np.std(quotes['log_close'] - np.polyval(fit, quotes['mjd']/365.24))
    duration = (quotes['mjd'][-1] - quotes['mjd'][0])/365.24
    color = cmap(norm(duration))
    ax[0].plot(100*fit[0], 100*std, 'o', color=color, markersize=12, markeredgecolor='k',alpha=0.5)
    ax[0].text(100*fit[0], 100*std, '   '+key, fontsize=12)
    ax[0].plot([100*slope0, 100*fit[0]], [100*sig0, 100*std], 'k-', alpha=0.1)
    ax[1].errorbar(100*slope0, 100*fit[0], xerr = 100*sig0, yerr = 100*std, fmt='o', color=color, markersize=12, markeredgecolor='k', alpha=0.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Quote duration (years)')
ax[0].set_xlabel('Growth rate (%/year)')
ax[0].set_ylabel('Standard deviation of growth rate (%/year)')
ax[0].set_title('Growth rate vs Standard deviation for each symbol')
ax[1].set_ylabel('Long-term rate (%/year)')
ax[1].set_xlabel('Portfolio rate (%/year)')
ax[1].set_xlim([-10, 50])
ax[1].set_ylim([-10, 50])
ax[1].plot([-100, 100], [-100, 100], 'k--', alpha=0.5)
ax[0].grid()
ax[1].grid()
add_fig_to_pdf(fig4)

# Text page: Dividends
umonth = np.unique(tbl0['Month'])
div_text = ""
for month in umonth[-5:]:
    div_text += f'\nDividends for month: {month}\n'
    tbl1 = tbl0[tbl0['Month'] == month]
    tot = 0
    for sym in symbols:
        val_div_yield = tbl1[sym+'_div_yield']
        n_sym = tbl1[sym+'_quantity']
        total_div_yield = np.sum(val_div_yield * n_sym)
        if total_div_yield == 0:
            continue
        div_text += f'{sym} - {month}: {total_div_yield:.2f}$\n'
        tot += total_div_yield
    if tot == 0:
        continue
    div_text += f'Total dividends for : {tot:.2f}$\n'
add_text_to_pdf(div_text)

# Text page: Summary
current_value = tbl0['val_total'][-1]
keys = 'COTISATION','TRANSFERT REÇU'
keep_injection = [tbl['Type de transaction'][i] in keys for i in range(len(tbl))]
tbl = tbl[keep_injection]
tbl = tbl[tbl['Devise du compte'] != 'USD']
tbl["Montant"] = tbl["Montant de l'opération"].astype(float)
tbl['Time_ago'] = Time.now().mjd-Time(tbl['Date de règlement']).mjd
growth_rate = -0.1
def get_current_account(growth_rate):
    return np.sum((1+growth_rate)**(tbl['Time_ago']/365.24)*tbl["Montant"])
def get_val(growth_rate):
    return np.abs(get_current_account(growth_rate)-current_value)
growth_rate = minimize(get_val,0.1).x[0]
total_cost = np.sum(tbl["Montant"])
value_sp500 =  np.sum(tbl["Montant de l'opération"]/tbl['SP500'])*sp500['Close'][-1]
current_computed = get_current_account(growth_rate)
summary_text = (
    f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
    f'Portefeuille: Moyenne={moy_portefeuille*100:.2f}%/yr, Médiane={med_portefeuille*100:.2f} %/yr, Écart-type={sig_portefeuille*100:.2f} %/yr\n'
    f'SP500: Moyenne={moy_sp500*100:.2f}%/yr, Médiane={med_sp500*100:.2f} %/yr, Écart-type={sig_sp500*100:.2f} %/yr\n\n'
    f'Effective mean growth: {fit_growth[0]*100.0:.2f} %/an (std: {std_growth*100.0:.2f} %/an)\n'
    f'Effective mean growth SP500: {np.polyfit(tbl0["mjd"]/364.24,np.log(vv_sp500), 1)[0]*100.0:.2f} %/an (std: {std_sp500*100.0:.2f} %/an)\n\n'
    f'Current : {bt.pdollar(current_value)}\n'
    f'Total cost : {bt.pdollar(total_cost)}\n'
    f'Total gain : {bt.pdollar(current_value - total_cost)}\n\n'
    f'Current growth rate : {100*growth_rate:.2f}%\n'
    f'Yearly growth rate : {bt.pdollar(current_computed*growth_rate)}/yr\n'
    f'Per 2 weeks : {bt.pdollar(current_computed*growth_rate/26.09)}/pay\n'
    f'Daily growth rate  (250 days/yr) : {bt.pdollar(current_computed*growth_rate/250)}/day\n'
    f'Daily growth rate  (365 days/yr) : {bt.pdollar(current_computed*growth_rate/365)}/day\n\n'
    f'Latest value of SP500 : {bt.pdollar(sp500["Close"][-1])}\n'
    f'Value if investing everything in SP500 : {bt.pdollar(value_sp500)}\n'
)


from fpdf import FPDF
import tempfile
import os


add_text_to_pdf(summary_text)
# --- Save all figures as images ---
fig_image_files = []
for i, fig in enumerate(pdf_pages):
    tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmpfile.name, bbox_inches='tight')
    fig_image_files.append(tmpfile.name)
    plt.close(fig)

# --- Create a single PDF with figures and text ---
class TextBoxPDF(FPDF):
    def text_box(self, text, x, y, w, h, font_size=8, fill_color=(245, 222, 179), border_radius=5):
        # Set font and colors
        self.set_xy(x, y)
        self.set_font("Arial", size=font_size)
        self.set_fill_color(*fill_color)
        # Draw rounded rectangle (simulate with four arcs and lines)
        self.rounded_rect(x, y, w, h, border_radius, style='F')
        # Write text inside the box
        self.set_xy(x + 3, y + 3)
        self.multi_cell(w - 6, font_size + 2, text, align='L')

    def rounded_rect(self, x, y, w, h, r, style=''):
        # Draw a rounded rectangle (approximation)
        self.set_draw_color(200, 180, 120)
        self.set_line_width(0.5)
        self.rect(x + r, y, w - 2 * r, h, style)
        self.rect(x, y + r, w, h - 2 * r, style)
        # Four corners (simulate arcs with small circles)
        self.ellipse(x, y, 2 * r, 2 * r)
        self.ellipse(x + w - 2 * r, y, 2 * r, 2 * r)
        self.ellipse(x, y + h - 2 * r, 2 * r, 2 * r)
        self.ellipse(x + w - 2 * r, y + h - 2 * r, 2 * r, 2 * r)

pdf = TextBoxPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=8)

# Add each figure as a page
for img_path in fig_image_files:
    pdf.add_page()
    pdf.image(img_path, x=10, y=10, w=pdf.w - 20)

# Add each text chunk as a page with a wheat rounded textbox
for chunk in text_chunks:
    pdf.add_page()
    box_width = pdf.w - 30
    box_height = pdf.h - 40
    pdf.text_box(chunk, x=15, y=20, w=box_width, h=box_height, font_size=8, fill_color=(245, 222, 179), border_radius=8)

pdf.output("effective_growth_report.pdf")

# --- Clean up temporary image files ---
for img_path in fig_image_files:
    os.remove(img_path)
# End of script