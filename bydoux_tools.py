from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from astropy.time import Time
import os

from astropy.table import Table, vstack
import numpy as np
import os
import wget

import datetime
import pickle
from tqdm   import tqdm

from currency_converter import CurrencyConverter
from datetime import date

from scipy.interpolate import UnivariateSpline as ius

import pandas as pd

import matplotlib.pyplot as plt
from astropy.table import Table
import glob



def compare_spx(ticker, tbl = None):
    if tbl is None:
        tbl = read_quotes(ticker)
    tbl_spx = read_quotes('^SPX')

    mjd_spx = tbl_spx['mjd']
    log_close_spx = tbl_spx['log_close']

    mjd = tbl['mjd']
    log_close = tbl['log_close']

    jd1 = mjd[0]
    jd2 = mjd[-1]

    g = (mjd_spx>=jd1) & (mjd_spx<=jd2)

    slope_spx = np.polyfit(mjd_spx[g]/365.24, log_close_spx[g], 1)[0]
    sig_spx = np.std(log_close_spx[g] - (slope_spx * mjd_spx[g]/365.24))

    slope = np.polyfit(mjd/365.24, log_close , 1)[0]
    sig = np.std(log_close - (slope * mjd/365.24))

    dict_spx = {
        'diff_slope': slope - slope_spx,
        'diff_ratio': sig / sig_spx,
        'ticker_slope': slope,
        'ticker_stdv': sig,
        'spx_slope': slope_spx,
        'spx_sigma': sig_spx,
        'overlap_years': (jd2-jd1)/365.24
    }

    return dict_spx

def top_holdings(ticker):
    t = yf.Ticker(ticker)
    fd = t.funds_data
    tbl = Table.from_pandas(fd.top_holdings, index=True)

    return tbl

def gini(v):
    # Compute the Gini coefficient of a vector v
    # Gini is a measure of inequality, 0 = perfect equality, 1 = perfect inequality
    v = np.array(v)
    if np.sum(v) == 0:
        return np.nan
    v = v[v > 0]
    n = len(v)
    if n == 0:
        return np.nan
    v = np.sort(v)
    index = np.arange(1, n + 1)
    gini_coeff = 1-((2 * np.sum(index * v)) / (n * np.sum(v)) - (n + 1) / n)
    return gini_coeff

def sectors(ticker):
    sectors = "realestate","consumer_cyclical","basic_materials","consumer_defensive","technology","communication_services","financial_services","utilities","industrials","energy","healthcare"
    tbl0 = dict()
    tbl0['sector'] = sectors
    tbl0['frac'] = np.zeros(len(sectors))

    tbl0['gini'] = -1

    try:
        wsectors = yf.Ticker(ticker).funds_data.sector_weightings
    except:
        return tbl0

    wsectors = dict(wsectors.items())

    frac = []
    vkey = []
    for key in wsectors:
        frac.append(wsectors[key])
        vkey.append(key)
    frac  = np.array(frac)
    vkey  = np.array(vkey)

    tbl = dict(sector = vkey, frac = frac)

    if len(tbl) == 0:
        return tbl0

    tbl['gini'] = gini(frac)

    return tbl


def bmo_etf():
    # List of BMO ETF tickers (with some non-ETF entries, but filtered later)
    tickers = [
        'ZGI', 'ZCN', 'STPL', 'ZIU', 'COMM', 'ZDM', 'DISC', 'ZEA', 'ZEO', 'ZEM',
        'ZGD', 'ZUE', 'ZJG', 'ZSP', 'ZEAT', 'ZSP.U', 'ZMT', 'ZDJ', 'ZQQ', 'ESGA',
        'ZNQ', 'ESGY', 'ZNQ.U', 'ESGE', 'ZMID.F', 'ESGG', 'ZMID', 'ZGRN', 'ZMID.U', 'ZCLN',
        'ZSML.F', 'ESGB', 'ZSML', 'ESGF', 'ZSML.U', 'ZJPN', 'ZJPN.F', 'ZCS', 'ZID', 'ZCS.L',
        'ZCH', 'ZCM', 'ZLC', 'ZLB', 'ZPS', 'ZLH', 'ZPS.L', 'ZLU', 'ZFS', 'ZLU.U',
        'ZFS.L', 'ZLD', 'ZMP', 'ZLI', 'ZPL', 'ZLE', 'ZFM', 'ZUQ.F', 'ZFL', 'ZUQ',
        'ZUQ.U', 'ZAG', 'ZGQ', 'ZSB', 'ZEQ', 'ZCB', 'ZDV', 'ZDB', 'ZUD', 'ZSDB',
        'ZDY', 'ZCDB', 'ZDY.U', 'ZMMK', 'ZDH', 'ZST', 'ZDI', 'ZST.L', 'ZVC', 'ZMBS',
        'ZVU', 'ZGB', 'ZRR', 'ZOCT', 'ZBI', 'ZJUL', 'ZBBB', 'ZAPR', 'ZQB', 'ZJAN',
        'ZUAG', 'ZWC', 'ZUAG.F', 'ZWS', 'ZUAG.U', 'ZWH', 'ZSU', 'ZWH.U', 'ZMU', 'ZWA',
        'ZIC', 'ZWE', 'ZIC.U', 'ZWP', 'ZUCM', 'ZWG', 'ZUCM.U', 'ZWB', 'ZUS.U', 'ZWB.U',
        'ZUS.V', 'ZWK', 'ZTS', 'ZWU', 'ZTS.U', 'ZWHC', 'ZTM', 'ZWEN', 'ZTM.U', 'ZWT',
        'ZTL.F', 'ZWQT', 'ZTL', 'ZPAY.F', 'ZTL.U', 'ZPAY', 'ZTIP', 'ZPAY.U', 'ZTIP.U', 'ZPH',
        'ZTIP.F', 'ZPW', 'ZJK', 'ZPW.U', 'ZHY', 'ZJK.U', 'ZEB', 'ZFH', 'ZUT', 'ZEF',
        'ZRE', 'ZIN', 'ZPR', 'ZUB', 'ZPR.U', 'ZBK', 'ZHP', 'ZUH', 'ZUP', 'ZHU',
        'ZUP.U', 'PAGE', 'Ticker', 'Date', 'ASSET', 'ZCON', 'ZBAL', 'ZGRO', 'ZEQT', 'ZESG',
        'ZBAL.T', 'ZGRO.T', 'ZMI', 'ACTIVE', 'ZLSC', 'ZLSU', 'ZZZD', 'ZACE', 'ARKK', 'ARKG',
        'ARKW', 'BGEQ', 'BGHC', 'BGIF', 'BGIN', 'BGDV', 'BGRT', 'GRNI', 'TOWR', 'WOMN',
        'ZGSB', 'ZMSB', 'ZCPB', 'ZFC', 'ZFN', 'ZXLE', 'ZXLU', 'ZXLK', 'ZXLB', 'ZXLP',
        'ZXLY', 'ZXLI', 'ZXLC', 'ZXLV', 'ZXLF', 'ZXLR', 'ZXLE.F', 'ZXLU.F', 'ZXLK.F', 'ZXLB.F',
        'ZXLP.F', 'ZXLY.F', 'ZXLI.F', 'ZXLC.F', 'ZXLV.F', 'ZXLF.F', 'ZXLR.F'
    ]
    # Add .TO suffix for Yahoo Finance Canada tickers
    tickers = np.array([ticker + '.TO' for ticker in tickers])

    return tickers

def desjardins_etf():
    tickers = ['DCU.TO', 'DCS.TO', 'DCG.TO', 'DCC.TO', 'DCP.TO', 'DCBC.TO',
               'DMEC.TO', 'DMEU.TO', 'DMEI.TO', 'DMEE.TO','DRCU.TO', 'DRMC.TO', 'DRMU.TO',
                'DRMD.TO', 'DRME.TO', 'DRFC.TO', 'DRFU.TO', 'DRFD.TO', 'DRFE.TO',
                'DSAE.TO', 'DCBC.TO', 'DCU.TO', 'DCS.TO', 'DCG.TO', 'DCC.TO', 'DCP.TO',
                'DANC.TO', 'DAMG.TO']

    tickers = np.unique(np.array(tickers))
    return tickers

def pdollar(x0):
    """
    Convert a number to a string with a dollar sign and two decimal places.
    :param x: the number to convert
    :return: the string with the dollar sign and two decimal places
    Also add spaces between thousands and two decimal places.
    1.0 -> '1,00$' and 1000.0 -> '1 000,00$'
    Put the $ at the end with the French format.
    """
    if np.isnan(x0):
        return '---'
    else:
        # Convert to float and format with two decimal places, using space as thousands separator and comma as decimal
        x = float(x0)
        s = f"{abs(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
        if x < 0:
            return f"-{s}$"
        else:
            return f"{s}$"

def get_bydoux_path():
    # This function returns the path to the bydoux data directory
    # It is set to ~/bydoux_data/ by default
    # You can change this to your desired path
    # by changing the return statement

    bydoux_path = os.path.expanduser("~")+'/bydoux_data/'
    if not os.path.exists(bydoux_path):
        print('Creating the directory', bydoux_path)
        os.makedirs(bydoux_path)

    # if sub-directory 'dividends' does not exist, create it
    dividends_path = os.path.join(bydoux_path, 'dividends')
    if not os.path.exists(dividends_path):
        print('Creating the directory', dividends_path)
        os.makedirs(dividends_path)

    # if sub-directory 'quotes' does not exist, create it
    quotes_path = os.path.join(bydoux_path, 'quotes')
    if not os.path.exists(quotes_path):
        print('Creating the directory', quotes_path)
        os.makedirs(quotes_path)
    
    # if sub-directory 'disnat' does not exist, create it
    disnat_path = os.path.join(bydoux_path, 'disnat')
    if not os.path.exists(disnat_path):
        print('Creating the directory', disnat_path)
        os.makedirs(disnat_path)

    return bydoux_path


def get_disnat_summary(verbose = False):
    # all files are in the disnat directory and are in the xlsx format
    # just copy-paste from the website and any duplicate will be handled. It's
    # ok if multiple sheets have the same transaction.

    xlsx_files =  glob.glob(get_bydoux_path()+'disnat/*xlsx')

    for ifile, xlsx_file in enumerate(xlsx_files):
        df = pd.read_excel(xlsx_file)  # Change sheet_name if needed

        # Convert the pandas DataFrame to an Astropy Table
        table = Table.from_pandas(df)

        # check if the table is empty
        if len(table) == 0:
            if verbose:
                printc('Empty table')
            continue

        prix = np.array(table['Prix'],dtype = str)
        prix_float = np.zeros(len(prix))
        for i in range(len(prix)):
            try:
                prix_float[i] = float(prix[i].replace(',','.'))
            except:
                prix_float[i] = np.nan
        table['Prix'] = prix_float

        # set 'Prix' as 'object'
        table['IFILE'] = ifile
        #if first file, create a new table
        if 'tbl' not in locals():
            tbl = table
        else:
            if len(table) !=0:
                # otherwise, append to the existing table
                tbl = vstack([tbl, table])

    tbl['mjd'] = 0.
    for i in range(len(tbl)):
        # if transaction date is missing, use settlement date
        if '20' not in tbl['Date de transaction'][i]:
            tbl['Date de transaction'][i] = tbl['Date de règlement'][i]
        tbl['mjd'][i] = Time(tbl['Date de transaction'][i]).mjd


    duplicates = np.zeros(len(tbl), dtype=bool)

    dates = tbl['Date de transaction'].data
    montants = tbl['Montant de l\'opération'].data
    ifile = tbl['IFILE'].data
    for i in tqdm(range(len(tbl)), desc='Removing duplicates', leave=verbose):
        for j in range(i+1, len(tbl)):
            cond1 = dates[i] == dates[j]
            cond2 = montants[i] == montants[j]
            cond3 = ifile[i] != ifile[j]
            if cond1 and cond2 and cond3:
                duplicates[j] = True
    tbl = tbl[duplicates == False]



    return tbl

def mjd2workday(mjd):
    # Convert Modified Julian Date (MJD) to a custom "workday" index.
    # 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
    # This function maps MJD to a workday number, where each week has 5 workdays.
    # The calculation shifts the week so that Monday is 0, then calculates the week number.
    day_of_week = (mjd+3) % 7  # Shift so Monday is 0
    mjd_week = (mjd+2)//7      # Integer division to get week number
    return day_of_week + mjd_week*5  # Workday index (Monday=0, Tuesday=1, ...)

def summary(ticker):
    """
    This function reads the quotes for a given ticker and plots the dividends.
    :param ticker: the ticker to read (string)
    :return: None
    """
    # Reference MJD for plotting (1970-01-01)
    t0 = Time('1970-01-01').mjd

    # Read the quotes table for the given ticker
    tbl = read_quotes(ticker)

    # Find indices where dividends are paid (Dividends > 0)
    idiv = np.where([tbl['Dividends'] > 0])[1]
    # Calculate annualized dividend fraction for each dividend event
    div_frac = tbl['Dividends'][idiv]/tbl['Close'][idiv]*365.25/np.gradient(tbl['mjd'][idiv])

    # Add an annualized dividend column to the table, initialized to zero
    tbl['div_frac'] = 0.
    # For each dividend event, fill the corresponding range in the table with the calculated dividend fraction
    for i in range(len(idiv)):
        if i == 0:
            i1=0
        else:
            i1 = idiv[i-1]
        if i == len(idiv)-1:
            i2 = len(tbl)
        else:
            i2 = idiv[i+1]
        tbl['div_frac'][i1:i2] = div_frac[i]

    # Create a figure with two subplots
    fig, ax  = plt.subplots(2,1, figsize=(8,12))

    # Plot the log of the close price with dividends on the first subplot
    ax[0].plot_date(tbl['mjd']-t0, tbl['log_close'], 'b-', label='Close')
    # Plot the log of the close price without dividends for comparison
    ax[0].plot_date(tbl['mjd']-t0, np.log(tbl['Close']), 'r-', label='Close without dividends')
    ax[0].set_title(ticker)
    ax[0].legend()

    # Plot the difference between log_close and log(Close) to show dividend impact
    ax[1].plot_date(tbl['mjd']-t0, (tbl['log_close'] - np.log(tbl['Close'])), 'b-', label='Volume')
    ax[1].set_title('Dividends')
    ax[1].legend()

    # Display the plots
    plt.show()

def today():
    # Returns today's date as a string in YYYYMMDD format (no dashes)
    return datetime.date.today().isoformat().replace('-','')

def get_recommentations():
    # Print Yahoo Finance recommendation for each S&P500 ticker
    tickers = get_snp500()
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        if 'recommendationKey' in info:
            print(ticker,info['recommendationKey'])

def xchange_range(mjd):
    # Interpolate the CAD to USD exchange rate for a range of MJDs
    c = CurrencyConverter()
    mjd = np.array(mjd)
    # Convert MJDs to ISO date strings
    dates = Time(mjd, format='mjd').iso

    # Initialize rates array with NaNs
    rates = np.zeros(len(mjd))+np.nan
    for i in range(len(mjd)):
        try:
            # Parse year, month, day from ISO date
            yr,mo,day = dates[i].split(' ')[0].split('-')
            # Get exchange rate for that date
            rate = c.convert(100, 'CAD', 'USD', date=date(int(yr), int(mo), int(day)))
            rates[i] = rate
        except:
            # If conversion fails, leave as NaN
            pass
    # Interpolate missing rates using a linear spline
    valid = np.isfinite(rates)
    rates = ius(mjd[valid], rates[valid], k=1,ext=3)(mjd)

    return rates

def write_pickle(tbl, filename):
    # Write a Python object to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(tbl, f)

def read_pickle(filename):
    # Read a Python object from a pickle file
    with open(filename,'rb') as f:
        tbl = pickle.load(f)
    return tbl

def get_snp500():
    """
    This function returns the list of the S&P500 tickers.
    It reads the list from the Wikipedia page using pandas.
    :return: the list of tickers as a numpy array
    """
    # Wikipedia link for S&P500 companies
    link = (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    )
    # Add headers so Wikipedia doesn't block
    headers = {"User-Agent": "Mozilla/5.0"}

    df = pd.read_html(link, header=0, storage_options=headers)[0]
    # Extract the 'Symbol' column as a numpy array
    ticker = np.array([ticker for ticker in df['Symbol']])

    return ticker

def read_google_sheet_csv(sheet_id: str, gid: str) -> Table:
    """
    Reads a Google Sheet as a CSV and returns it as an astropy Table.

    Args:
        sheet_id (str): The Google Sheet ID.
        gid (str): The GID (tab identifier) for the sheet.

    Returns:
        Table: The astropy Table containing the sheet data.
    """
    # Remove any existing temporary CSV file to avoid conflicts
    if os.path.exists('.tmp.csv'):
        os.remove('.tmp.csv')

    # Construct the Google Sheets CSV export URL
    GOOGLE_URL = 'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    csv_url = GOOGLE_URL.format(sheet_id=sheet_id, gid=gid)

    # Download the CSV file from Google Sheets
    _ = wget.download(csv_url, out='.tmp.csv', bar=None)
    # Read the CSV file into an astropy Table
    tbl = Table.read('.tmp.csv', format='ascii.csv')

    # Remove the temporary CSV file after reading
    os.remove('.tmp.csv')

    # Clean up masked/empty values and convert to fixed-length strings
    for key in tbl.keys():
        try:
            masked = tbl[key].mask
            tbl[key][masked] = '0'
            # Convert to string of 20 characters
            tbl[key] = np.array(tbl[key], dtype='S20')
        except:
            pass

    # If 'yf' column exists, fill missing values with TICKER+'.TO'
    if 'yf' in tbl.keys():
        tbl['yf'] = tbl['yf'].astype(str)
        for i in range(len(tbl)):
            if tbl['yf'][i] == '0':
                tbl['yf'][i] = tbl['TICKER'][i] + '.TO'

    # If transaction date exists, compute MJD and time since transaction
    if 'Date de transaction' in tbl.keys():
        tbl['MJD'] = 0.
        tbl['Time_ago'] = 0.
        now = Time.now().mjd
        for i in range(len(tbl)):
            # If transaction date is missing, use settlement date
            if '-' not in tbl['Date de transaction'][i]:
                tbl['Date de transaction'][i] = tbl['Date de règlement'][i]
            tbl['MJD'][i] = Time(tbl['Date de transaction'][i]).mjd
            tbl['Time_ago'][i] = now - tbl['MJD'][i]

    # Convert columns with float values (replace comma with dot if needed)
    columns_float = 'Commission payée',"Montant de l'opération"
    for key in columns_float:
        if key in tbl.keys():
            tmp = tbl[key]
            for i in range(len(tmp)):
                try:
                    tmp[i] = float(tmp[i].replace(',','.'))
                except:
                    pass
            tbl[key] = tbl[key].astype(float)

    return tbl

def file_age(file):
    """
    Returns the age of a file in days.

    Args:
        file (str): The path to the file.

    Returns:
        float: The age of the file in days.
    """
    if os.path.exists(file):
        # Get the last modification time of the file (in seconds since epoch)
        last =  Time(os.path.getmtime(file),format='unix').mjd
        # Calculate the age in days
        delta_time = Time.now().mjd - last
    else:
        delta_time = np.inf

    return delta_time

def get_info(key, refresh_delay = 30):
    """
    Retrieves and caches Yahoo Finance info for a given ticker.

    Args:
        key (str): The ticker symbol.

        refresh_delay (int): Days before refreshing cached info.

    Returns:
        dict: The info dictionary from Yahoo Finance.
    """

    
    outname = get_bydoux_path()+'/quotes/' + key + '_info.pkl'

    # If info already cached, read from pickle, else fetch and cache
    if os.path.exists(outname):
        age = file_age(outname)
        if age < refresh_delay:
            info = read_pickle(outname)
            return info
        else:
            # remove old file
            os.remove(outname)
           
    info = yf.Ticker(key).info
    # make a dictionnary from the info
    info = dict(info.items())

    info['sectors'] = sectors(key)
    tbl = read_quotes(key)
    info['compare_sp500'] = compare_spx(key, tbl = tbl)

    info['events'] = compute_events(key,tbl = tbl)
    info['mjd0'] = tbl['mjd'][0]
    info['mjd1'] = tbl['mjd'][-1]
    info['duration_years'] = np.round((info['mjd1'] - info['mjd0'])/365.24,2)

    write_pickle(info, outname)

    # Optionally add 'currency' key if missing but 'financialCurrency' exists
    if 'currency' not in info.keys() and 'financialCurrency' in info.keys():
        info['currency'] = info['financialCurrency']

    return info

def printc(info):
    """
    Prints a message with a timestamp.

    Args:
        info (str): The message to print.
    """
    # Get current time (HH:MM:SS) and print with info
    tt =  Time.now().iso.split(' ')[1]+ ' | '
    print(tt, info)

def get_sp500_history():
    """
    Retrieves the historical data for the S&P 500 index, using cache if available.

    Returns:
        Table: Astropy Table with S&P 500 historical data.
    """
    today = datetime.date.today().isoformat().replace('-','')
    path = get_bydoux_path()
    all_sp500_name = f'{path}/quotes/snp500_all_{today}_index.pkl'

    # Use cached data if available, else download and cache
    if os.path.exists(all_sp500_name):
        printc('Accessing the S&P500 history in pickles')
        printc('\t'+all_sp500_name)
        all_sp500 = read_pickle(all_sp500_name)
    else:
        printc('Downloading the S&P500 history')
        data = yf.download('^SPX', period='max')
        tbl = Table()
        # Store date and MJD
        tbl['date'] =  Time(data.index).iso
        tbl['mjd'] = Time(data.index).mjd
        # Store all columns from Yahoo Finance DataFrame
        dd = dict(data)
        for col in dd.keys():
            tbl[col] = np.array(data[col])
        # Store close price for dividends
        tbl['Close_dividends'] = np.array(data['Close'])
        all_sp500 = write_pickle(tbl, all_sp500_name)

    return all_sp500

def read_quotes(ticker, force = False, verbose = False, try_failed = False):
    """
    Reads and updates (if needed) the quotes for a given ticker, including dividend-adjusted close.

    Args:
        ticker (str): The ticker symbol.
        force (bool): If True, forces re-download of data.
        verbose (bool): If True, prints progress info.

    Returns:
        Table: Astropy Table with quote data.
    """
    updated_cols = False  # Flag to track if columns were updated

    if verbose:
        printc(f'Reading quotes for {ticker}')

    # Construct the output filename for the FITS file containing the quotes
    outname = get_bydoux_path()+'quotes/' + ticker  + '.fits'

    # Path for a token file indicating a failed download
    flag_failed_file = get_bydoux_path()+'quotes/' + ticker  + '_failed.token'
    # If a failed token exists and we're not retrying, skip this ticker
    if not try_failed and os.path.isfile(flag_failed_file):
        if verbose:
            printc(f'Failed to read {ticker} from Yahoo')
            printc('We will not try again')
        return None
    
    # If retrying, remove the failed token so we can try again
    if try_failed and os.path.isfile(flag_failed_file) and verbose:
        os.remove(flag_failed_file)
        if verbose:
            printc(f'We will try again to read {ticker} from Yahoo')

    # If force is set and file exists, remove it to force re-download
    if os.path.isfile(outname) and force:
        if verbose:
            printc(f'Forcing re-download of {ticker}')
        os.remove(outname)

    # If the FITS file already exists, read it
    if os.path.isfile(outname):
        if verbose:
            printc(f'We have the file {outname}, reading it')
        tbl = Table.read(outname)

        # Find the last modification date of the FITS file (in MJD)
        last =  Time(os.path.getctime(outname),format='unix').mjd
        delta_time = Time.now().mjd - last

    else:
        # Otherwise, download the data from Yahoo Finance
        if verbose:
            printc(f'We do not have the file {outname}, downloading it from Yahoo')
        updated_cols = True
        try:
            data = yf.Ticker(ticker).history(period='max')
        except:
            data = pd.DataFrame()

        # If download failed and not retrying, create a failed token and return None
        if data.empty and not try_failed:
            if verbose:
                printc(f'Failed to read {ticker} from Yahoo')
            with open(flag_failed_file, 'w') as f:
                f.write('Failed to read from Yahoo')
            return None

        # Create an Astropy Table from the downloaded data
        tbl = Table()
        try:
            tbl['date'] =  Time(data.index).iso
            tbl['mjd'] = Time(data.index).mjd
            dd = dict(data)
            for col in dd.keys():
                tbl[col] = np.array(data[col])
            tbl['Close_dividends'] = np.array(data['Close'])
            delta_time = 0.0
        except:
            if verbose:
                printc(f'Failed to read {ticker} from Yahoo')
            with open(flag_failed_file, 'w') as f:
                f.write('Failed to read from Yahoo')
            return None



    # If the data is outdated, fetch only the missing period
    if delta_time > 1:
        # Choose the period to fetch based on how old the data is
        if delta_time < 5:
            period = '5d'
        elif delta_time < 30:
            period = '1mo'
        elif delta_time < 90:
            period = '3mo'
        elif delta_time < 180:
            period = '6mo'
        elif delta_time < 365:
            period = '1y'
        elif delta_time < 2*365:
            period = '2y'
        elif delta_time < 5*365:
            period = '5y'
        else:
            period = 'max'
        
        # Download new data for the missing period
        if verbose:
            printc(f'Downloading new data for {ticker} from Yahoo over {period}')
        data2 = yf.Ticker(ticker).history(period=period)
        tbl2 = Table()

        try:
            tbl2['date'] =  Time(data2.index).iso
            tbl2['mjd'] = Time(data2.index).mjd.astype(int)
        except:
            if verbose:
                printc(f'Failed to read {ticker} from Yahoo')
            with open(flag_failed_file, 'w') as f:
                f.write('Failed to read from Yahoo')
            return None


        dd = dict(data2)
        for col in dd.keys():
            tbl2[col] = np.array(data2[col])
        tbl2['Close_dividends'] = np.array(data2['Close'])
        # Only keep new rows (dates after the last in tbl)
        keep = tbl2['mjd'] > np.max(tbl['mjd'])
        if np.sum(keep) != 0:
            tbl2 = tbl2[keep]
            tbl = vstack([tbl, tbl2])
        updated_cols = True

    # If we updated columns (new data or fresh download), process the table
    if updated_cols:
        tbl['Close_dividends'] = np.array(tbl['Close'])

        # Remove rows where close price is zero (bad data)
        bad = tbl['Close_dividends'] == 0
        tbl = tbl[bad == False]

        # Apply dividend adjustment: for each dividend, increase all subsequent close prices
        for i in range(len(tbl)):
            if tbl['Dividends'][i] > 0:
                gain_frac = 1+tbl['Dividends'][i]/tbl['Close_dividends'][i]
                tbl['Close_dividends'][i:] *= gain_frac

        # Compute log of adjusted close price (for log-return analysis)
        tbl['log_close'] = np.log(tbl['Close_dividends'])

        # Optionally print and write the table to disk
        if verbose:
            printc(f'Writing {outname}')

        # Compute yearly running dividend yield
        mjd = np.array(tbl['mjd'])
        div = np.array(tbl['Dividends'])
        idiv = np.where(div > 0)[0]
        if len(idiv) > 2:
            # Calculate annualized dividend yield for each dividend event
            div_frac = div[idiv]/tbl['Close_dividends'][idiv]*365.25/np.gradient(mjd[idiv])
            tbl['Dividend_yearly'] = 0.
            for i in range(len(idiv)):
                if i == 0:
                    i1=0
                else:
                    i1 = idiv[i-1]
                if i == len(idiv)-1:
                    i2 = len(tbl)
                    tbl['Dividend_yearly'][i1:i2] = np.nan
                else:
                    i2 = idiv[i+1]
                    tbl['Dividend_yearly'][i1:i2] = div_frac[i]
        else:
            tbl['Dividend_yearly'] = 0.0

        # Ensure Dividend_yearly is a numpy array (not a masked array)
        tbl['Dividend_yearly'] = np.array(tbl['Dividend_yearly'].data)

        # Add columns for day of week, week number, and custom workday index
        tbl['day of week'] = (np.array(tbl['mjd'],dtype = int)+3) % 7  # Monday=0
        tbl['mjd week'] = np.array((tbl['mjd']+2)//7,dtype = int)      # Week number
        tbl['work day'] =  tbl['day of week']+tbl['mjd week']*5        # Custom workday index

        # Optionally print the table for debugging
        if verbose:
            print(tbl)

        # Add year column for convenience (as string)
        tbl['year'] = 0
        for i in range(len(tbl)):
            tbl['year'][i] = Time(tbl['date'][i]).iso.split('-')[0]

        if verbose:
            # Write updated table to disk (overwrite existing file)
            print(f'Writing {outname}')
        tbl.write(outname, overwrite=True)

    # Add plot_date column as numpy datetime64 for easy plotting
    tbl['plot_date'] = np.array(tbl['date'], dtype='datetime64')

    # Remove data before year 2000 for consistency (optional, but keeps tables manageable)
    tbl = tbl[tbl['mjd'] > Time('2000-01-01').mjd]


    info = yf.Ticker(ticker).info
    # if longName not in info, we have a problem and return an empty table
    if 'longName' not in info:
        printc(f'Problem with {ticker}, no longName in info')
        return Table()

    typeDisp = info['typeDisp'] if 'typeDisp' in info else 'Other'

    type_table_compile = os.path.expanduser("~")+f'/bydoux/summary_{typeDisp}.csv'
    if os.path.exists(type_table_compile):
        tbl_compile = Table.read(type_table_compile, format='ascii.tab')
        if ticker not in tbl_compile['Ticker']:
            new_row = {'Ticker': ticker, 'longName': info['longName'],'comment': '--None--'}
            tbl_compile.add_row(new_row)
            tbl_compile.write(type_table_compile, format='ascii.tab', overwrite=True)
            printc(f'Added {ticker} to {type_table_compile}')
    else:
        tbl_compile = Table()
        tbl_compile['Ticker'] = [ticker]
        tbl_compile['longName'] = [info['longName']]
        tbl_compile['comment'] = ['--None--']
        tbl_compile.write(type_table_compile, format='ascii.tab', overwrite=True)
        printc(f'Created {type_table_compile} with {ticker}')


    return tbl

def get_tsx(FNB = False):
    """
    Returns TSX tickers from a Google Sheet, optionally filtering for ETFs.

    Args:
        FNB (bool): If True, only return ETF tickers.

    Returns:
        np.ndarray: Array of TSX ticker symbols (with '.TO' suffix).
    """
    # Read TSX tickers from Google Sheet
    tbl = read_google_sheet_csv('1bx3oBEFAmksB6no7_DV7AP_qM9zQ5iHUepc9wcPjXO8', '460184511')

    # If FNB is True, filter for ETFs
    if FNB:
        fnb = np.zeros(len(tbl), dtype=bool)
        for i in range(len(tbl)):
            if 'ETF' in tbl['Stock Company Name'][i]:
                fnb[i] = True
        tbl = tbl[fnb]
    
    # Get unique tickers and add '.TO' suffix
    tickers = np.array(tbl['TICKER'])
    tickers = np.unique(tickers)
    for i in range(len(tickers)):
        tickers[i] = tickers[i] + '.TO'
    return tickers

def batch_quotes(sample, full = False, force = False):
    """
    Downloads and caches quotes for a batch of tickers (S&P500, FNB, or TSX).

    Args:
        sample (str): Which sample to use ('S&P500', 'FNB', or 'TSX').
        full (bool): Unused, for future extension.
        force (bool): If True, force re-download of all data.

    Returns:
        dict: Dictionary of ticker:table pairs.
    """
    # Select tickers and output file prefix based on sample type
    if sample == 'S&P500':
        tickers = get_snp500()
        name_prefix = 'snp500_'
    else:
        if sample == 'FNB':
            name_prefix = 'fnb_'
            GID = '0'
            ID = '1bx3oBEFAmksB6no7_DV7AP_qM9zQ5iHUepc9wcPjXO8'
            tbl = read_google_sheet_csv(ID, GID)
            tickers = get_tsx(FNB = True)
        if sample == 'TSX':
            name_prefix = 'tsx_'
            GID = '460184511'
            ID = '1bx3oBEFAmksB6no7_DV7AP_qM9zQ5iHUepc9wcPjXO8'
            tbl = read_google_sheet_csv(ID, GID)
            tickers = tbl['TICKER']

    path = get_bydoux_path()
    all_sp500_name = f'{path}/quotes/{name_prefix}all_{today()}.pkl'

    # If force is set, remove any existing cache file
    if force and os.path.exists(all_sp500_name):
        os.remove(all_sp500_name)

    # Use cached data if available, else download and cache
    if os.path.exists(all_sp500_name):
        all_sp500 = read_pickle(all_sp500_name)
    else:
        doplot = False
        all_sp500 = {}
        for ticker in tickers:
            printc(f' input of {ticker} in batch_quotes')
            outname = get_bydoux_path()+'/quotes/' + ticker + '_' + today() + '.fits'
            try:
                tbl1 = read_quotes(ticker)
                # Convert astropy Table to dictionary for pickling
                tbl1 = {k: tbl1[k].data for k in tbl1.keys()}
                all_sp500[ticker] = tbl1
            except:
                printc(f'failed for {ticker}')
        # Remove any previous cache files for today
        delname = all_sp500_name.replace(today(),'*')
        os.system('rm '+delname)
        write_pickle(all_sp500, all_sp500_name)

    return all_sp500


def compute_events(ticker, tbl = None):
    if tbl is None:
        tbl = read_quotes(ticker)
    path = get_bydoux_path()
    event_table = Table.read(f'{path}/events.csv', format='csv', delimiter='&')


    dict_event = {}
    for j in range(len(event_table)):
            
        mjd1 = Time(event_table['event_start'][j]).mjd
        mjd2 = Time(event_table['event_end'][j]).mjd
        mjd3 = Time(event_table['baseline_start'][j]).mjd
        mjd4 = Time(event_table['baseline_end'][j]).mjd

        # make a bolean mask between mjd1 and mjd2
        mask12 = (tbl['mjd'] >= mjd1) & (tbl['mjd'] <= mjd2)
        mask34 = (tbl['mjd'] >= mjd3) & (tbl['mjd'] <= mjd4)

        log_close12 = np.mean(tbl['log_close'][mask12])
        log_close34 = np.mean(tbl['log_close'][mask34])
        dict_event[event_table['Event'][j]] = log_close12 - log_close34


    return dict_event