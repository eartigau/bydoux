import bydoux_tools as bt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table

# List of tickers to analyze
etf_table = 'summary_ETF.csv'

# List of BMO ETF tickers (with some non-ETF entries, but filtered later)

tbl = Table.read(etf_table,format='ascii.tab')

bad = np.zeros(len(tbl), dtype=bool)
# check if these is a 'FAIL' in the comment column
for i, comment in enumerate(tbl['comment']):
    if 'FAIL' in comment.upper():
        bad[i] = True
tbl = tbl[~bad]    


tickers = tbl['Ticker'].data

# sort tickers alphabetically
tickers.sort()

# Add more tickers as needed
# Create a multipage PDF
with PdfPages('/Users/eartigau/bydoux/summary_etf_report.pdf') as pdf:
    for ticker in tickers:
        # Read the historical quotes table for the ticker
        tbl = bt.read_quotes(ticker)
        info = bt.get_info(ticker)

        # Print ticker and name info
        short_name = info['shortName'] if 'shortName' in info else ''
        print(f"Processing {ticker} {short_name}")

        # Calculate time since each data point (in days)
        dt = tbl['mjd'].max() - tbl['mjd']

        # Collect summary lines for the text box
        output_lines = []

        # Compute and collect returns for various time windows
        if np.max(dt) > 31:
            tbl_mo = tbl[dt < 31]
            lg = np.log(tbl_mo['Close'][-1] / tbl_mo['Close'][0])
            output_lines.append(f"Last month: {lg:.2%} total")
            lg2 = tbl_mo['log_close'][-1] - tbl_mo['log_close'][0]
            output_lines.append(f"Last month: {lg2:.2%} total")
        if np.max(dt) > 93:
            tbl_qt = tbl[dt < 93]
            lg = np.log(tbl_qt['Close'][-1] / tbl_qt['Close'][0])
            output_lines.append(f"Last quarter: {lg:.2%} total")
            lg2 = tbl_qt['log_close'][-1] - tbl_qt['log_close'][0]
            output_lines.append(f"Last quarter: {lg2:.2%} total")
        if np.max(dt) > 182:
            tbl_6mo = tbl[dt < 182]
            lg = np.log(tbl_6mo['Close'][-1] / tbl_6mo['Close'][0])
            output_lines.append(f"Last 6 months: {lg:.2%} total")
            lg2 = tbl_6mo['log_close'][-1] - tbl_6mo['log_close'][0]
            output_lines.append(f"Last 6 months: {lg2:.2%} total")
        if np.max(dt) > 365:
            tbl_yr = tbl[dt < 365]
            lg = np.log(tbl_yr['Close'][-1] / tbl_yr['Close'][0])
            output_lines.append(f"Last year: {lg:.2%} /yr")
            lg2 = tbl_yr['log_close'][-1] - tbl_yr['log_close'][0]
            output_lines.append(f"Last year: {lg2:.2%} /yr")
        if np.max(dt) > 1095:
            tbl_3yr = tbl[dt < 1095]
            lg = np.log(tbl_3yr['Close'][-1] / tbl_3yr['Close'][0]) / 3
            output_lines.append(f"Last 3 years: {lg:.2%} /yr")
            lg2 = (tbl_3yr['log_close'][-1] - tbl_3yr['log_close'][0]) / 3
            output_lines.append(f"Last 3 years: {lg2:.2%} /yr")
        if np.max(dt) > 1825:
            tbl_5yr = tbl[dt < 1825]
            lg = np.log(tbl_5yr['Close'][-1] / tbl_5yr['Close'][0]) / 5
            output_lines.append(f"Last 5 years: {lg:.2%} /yr")
            lg2 = (tbl_5yr['log_close'][-1] - tbl_5yr['log_close'][0]) / 5
            output_lines.append(f"Last 5 years: {lg2:.2%} /yr")
        if np.max(dt) > 3650:
            tbl_10yr = tbl[dt < 3650]
            lg = np.log(tbl_10yr['Close'][-1] / tbl_10yr['Close'][0]) / 10
            output_lines.append(f"Last 10 years: {lg:.2%} /yr")
            lg2 = (tbl_10yr['log_close'][-1] - tbl_10yr['log_close'][0]) / 10
            output_lines.append(f"Last 10 years: {lg2:.2%} /yr")

        # Compute annual log returns for each year (capital only and with dividends)
        yrs = np.unique(tbl['year'])
        lgs = []
        lgs2 = []
        for yr in yrs:
            g = tbl[tbl['year'] == yr]
            lg = np.log(g['Close'][-1] / g['Close'][0])
            lg2 = g['log_close'][-1] - g['log_close'][0]
            lgs.append(lg)
            lgs2.append(lg2)
        lgs = np.array(lgs)
        lgs2 = np.array(lgs2)

        # Plot step plot of annual log returns
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(yrs, lgs, where='mid', label='Capital only')
        ax.step(yrs, lgs2, where='mid', label='With dividends', color='tab:orange')

        # Annotate percent values above (lgs) and below (lgs2) each year
        for i, yr in enumerate(yrs):
            ax.text(yr, lgs[i] + 0.01, f"\n{lgs[i]*100:.1f}%", ha='center', va='top', fontsize=9, color='tab:blue')
            ax.text(yr, lgs2[i] - 0.01, f"{lgs2[i]*100:.1f}%\n", ha='center', va='bottom', fontsize=9, color='tab:orange')

        # Draw a horizontal line at y=0
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual log return')
        ax.set_title(f'Annual log returns for {ticker}\n{short_name}')
        ax.legend()

        # Add the summary printout as a text box on the right of the plot
        textstr = "\n".join(output_lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=props, family='monospace')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


    # loop through tickes and overplot the annual log returns
    fig, ax = plt.subplots(figsize=(10, 5))
    for ticker in tickers:
        tbl = bt.read_quotes(ticker)
        yrs = np.unique(tbl['year'])
        lgs = []
        for yr in yrs:
            g = tbl[tbl['year'] == yr]
            lg = np.log(g['Close'][-1] / g['Close'][0])
            lgs.append(lg)
        lgs = np.array(lgs)

        ax.step(yrs, lgs,  label=ticker, alpha = 0.5, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual log return')
    ax.set_title('Annual log returns for all ETFs')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
