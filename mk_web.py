import bydoux_tools as bt
from astropy.table import Table
import numpy as np
from jinja2 import Environment, FileSystemLoader
import bydoux_tools as bt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.table import Table
import time
from datetime import datetime
import os

def translator(col):
    translations = {
        'longName': 'Nom complet',
        'energy': 'énergie',
        'materials': 'matériaux',
        'realestate': 'immobilier',
        'utilities': 'services_publics',
        'consumer_defensive': 'conso_de_base',
        'communication_services': 'communication',
        'consumer_cyclical': 'conso_cyclique',
        'basic_materials': 'matériaux',
        'slope': 'Croissance',
        'sigma': 'Volatilité',
        'slope_sp500': 'CroissanceSP500',
        'sigma_sp500': 'VolatilitéSP500',
        'time_span': 'Durée',
        'healthcare': 'santé',
        'financial_services': 'finance',
        'industrials': 'industriel',
        'information_technology': 'information',
    }
    return translations.get(col, col)  # Return the translated name or the original if not found

def get_bulk():
    tbl_current = Table.read('current_symbols.csv')
    current = tbl_current['symbols'].data
    current_symbols = [s.replace('-C', '.TO') for s in current]
    current_symbols = [s.replace('.U.TO', '-U.TO') for s in current_symbols]

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
    # Check if BMP in longName
    keep = np.zeros(len(tickers), dtype=bool)
    for i, ticker in enumerate(tickers):
        if 'BMO' in tbl['longName'][i]:
            keep[i] = True

    tickers = np.concatenate([tickers, current_symbols])

    # Arrays to store results for each ticker
    out = Table()
    out['Ticker'] = np.zeros(len(tickers), dtype='U10')
    out['longName'] = np.zeros(len(tickers), dtype='U300')
    out['slope'] = np.zeros(len(tickers), dtype=float)
    out['sigma'] = np.zeros(len(tickers), dtype=float)
    out['slope_sp500'] = np.zeros(len(tickers), dtype=float)
    out['sigma_sp500'] = np.zeros(len(tickers), dtype=float)
    out['q'] = np.zeros(len(tickers), dtype=float)
    out['gini'] = np.zeros(len(tickers), dtype=float)
    out['time_span'] = np.zeros(len(tickers), dtype=float)

    info0 = bt.get_info('ZCN.TO')
    for sector in info0['sectors']['sector']:
        out[sector] = np.zeros(len(tickers), dtype=float)


    all_data = dict()

    print(f"Processing {len(tickers)} tickers...")
    verbose = True  # Set to True for detailed output
    for iticker, ticker in tqdm(enumerate(tickers),leave = False):
        try:
            info = bt.get_info(ticker)
        except:
            if verbose:
                print(f"Failed to get info for {ticker}, skipping.")
                tbl = Table.read(etf_table,format='ascii.tab')
                if ticker in tbl['Ticker'].data:
                    i = np.where(tbl['Ticker'].data == ticker)[0][0]
                    tbl['comment'][i] = 'FAIL: no data'
                    tbl.write(etf_table,format='ascii.tab',overwrite=True)
            continue

        out['Ticker'][iticker] = ticker
        out['longName'][iticker] = info['longName']
        out['slope'][iticker] = np.round(info['compare_sp500']['ticker_slope']*100, 2)
        out['sigma'][iticker] = np.round(info['compare_sp500']['ticker_stdv']*100, 2)
        out['slope_sp500'][iticker] = np.round(info['compare_sp500']['diff_slope']*100, 2)
        out['sigma_sp500'][iticker] = np.round(info['compare_sp500']['diff_ratio']*100, 2)
        out['time_span'][iticker] = info['duration_years']
        out['gini'][iticker] = np.round(info['sectors']['gini'], 2)

        for sector in info0['sectors']['sector']:
            if sector in info['sectors']['sector']:
                idx = np.where(np.array(info['sectors']['sector']) == sector)[0][0]
                out[sector][iticker] = np.round(info['sectors']['frac'][idx]*100,2)
            else:
                out[sector][iticker] = 0.0



    out['slope'][np.isnan(out['slope'])] = 0
    out['sigma'][np.isnan(out['sigma'])] = 0
    out['q'] = np.round(out['slope'] / out['sigma'], 3)
    out['q'][~np.isfinite(out['q'])] = 0

    out['time_span'][np.isnan(out['time_span'])] = 0
    out = out[np.isfinite(out['gini'])]
    out = out[out['gini'] >= 0]

    # loop on all keys and float columns have nans set to zero
    for key in out.colnames:
        if out[key].dtype == float:
            out[key][np.isnan(out[key])] = 0

    return out


tbl_bulk = get_bulk()
# find unique tbl_bulk['Ticker']
_, idx = np.unique(tbl_bulk['Ticker'].data, return_index=True)
tbl_bulk = tbl_bulk[np.sort(idx)]

for iticker in tqdm(range(len(tbl_bulk)),leave = False):
    ticker = tbl_bulk['Ticker'].data[iticker]
    outname = f"/Users/eartigau/bydoux/website/{ticker.replace('.','_')}.html"
    if os.path.exists(outname):
        tmod = os.path.getmtime(outname)
        if time.time() - tmod < 7*24*3600:
            continue

    # we put the ticker of interest last
    # index of all tickers except the one of interest
    idx = np.where(tbl_bulk['Ticker'].data != ticker)[0]
    # index of the ticker of interest
    idx2 = np.where(tbl_bulk['Ticker'].data == ticker)[0]
    # new order: all except the ticker of interest, then the ticker of interest
    new_order = np.concatenate([idx, idx2])
    tbl_bulk = tbl_bulk[new_order]

    tickers = '"'+'","'.join(tbl_bulk['Ticker'].data)+'"'
    slopes = ','.join((np.round(tbl_bulk['slope'],3)).astype(str))
    sigmas = ','.join((np.round(tbl_bulk['sigma'],3)).astype(str))



    # Préparer l'environnement
    env = Environment(loader=FileSystemLoader("/Users/eartigau/bydoux/templates"))
    template = env.get_template("template_ticker.html")


    info = bt.get_info(ticker)

    tbl = bt.read_quotes(ticker)

    # check if log_close has a mask attribute
    if hasattr(tbl['log_close'], 'mask'):
        tbl = tbl[~tbl['log_close'].mask]


    tbl['mjd'] = tbl['mjd'].astype(int)
    sp500 = bt.read_quotes('^SPX')
    sp500['mjd'] = sp500['mjd'].astype(int)

    tbl['log_sp500'] = np.zeros(len(tbl), dtype=float)
    for i in range(len(tbl)):
        g = np.argmin(np.abs(tbl['mjd'][i] - sp500['mjd']))
        tbl['log_sp500'][i] = sp500['log_close'][g]
    tbl['log_sp500'] -= np.mean(tbl['log_sp500'])
    tbl['log_close'] -= np.mean(tbl['log_close'])

    tbl2 = Table()
    tbl2['date'] = [t.split(' ')[0] for t in tbl['date']]
    tbl2['div'] = (10000*np.exp((tbl['log_close']- tbl['log_close'][0]))).astype(int).astype(str)
    tbl2['val'] = (10000*tbl['Close']/tbl['Close'][0]).astype(int).astype(str)
    # check if mask is a property of the column
    if hasattr(tbl2['div'], 'mask'):
        tbl2 = tbl2[~tbl2['div'].mask]
    

    v1 =  '["'+'","'.join(tbl2['date'])+'"];'
    v2 = ' [' + ','.join(tbl2['val']) + '];'
    v3 = ' [' + ','.join(tbl2['div']) + '];'

    dict_params = {
        'long_name': info['longName'],
        'ticker': ticker,
        'dates1': v1,
        'close1': v2,
        'div1': v3,
        'log_close': ' [' + ','.join((tbl['log_close']).astype(str)) + '];',
        'log_sp500': ' [' + ','.join((tbl['log_sp500']).astype(str)) + '];',
        'sectors': '"'+'","'.join(info['sectors']['sector'])+'"',
        'fractions_sectors': ','.join((info['sectors']['frac']*100).astype(str)),
        'slopes': slopes,
        'sigmas': sigmas,
        'tickers': tickers,
        'covid_dip':  np.round(info['events']['COVID-19']*100,1).astype(str),
        'tarif_dip':  np.round(info['events']['Rose Garden']*100,1).astype(str),
        'croissance': np.round(info['compare_sp500']['ticker_slope']*100,1).astype(str),
        'volatilite': np.round(info['compare_sp500']['ticker_stdv']*100,1).astype(str),
        'croissance_sp500': np.round(info['compare_sp500']['spx_slope']*100,1).astype(str),
        'volatilite_sp500': np.round(info['compare_sp500']['spx_sigma']*100,1).astype(str),
        'duree': np.round((np.max(tbl['mjd'])-np.min(tbl['mjd']))/365.24,2).astype(str)
    }

    output_from_parsed_template = template.render(params=dict_params)

    with open(outname, "w") as fh:
        fh.write(output_from_parsed_template)



env = Environment(loader=FileSystemLoader("/Users/eartigau/bydoux/templates"))
template = env.get_template("tableau_template.html")

# translate column names:
for col in tbl_bulk.colnames:
    col_fr = translator(col)
    if col != col_fr:
        tbl_bulk.rename_column(col, col_fr)



keys = np.array(tbl_bulk.colnames)
keys_is_float = np.zeros(len(keys), dtype=bool)
for i, key in enumerate(keys):
    if tbl_bulk[key].dtype == float:
        keys_is_float[i] = True

v_all = ''
for iline in range(len(tbl_bulk)):
    v = '          <tr>'
    for key in keys:
        tmp = tbl_bulk[key][iline].astype(str)
        if key == 'Ticker':
            tmp = '<a href="' + tbl_bulk[key][iline].replace('.','_') + '.html">' + tmp + '</a>'
        v += '<td>' + tmp + '</td>'
    v += '</tr>\n'
    v_all += v
head = '          <tr>'
for key in keys:
    key2 = key+''
    if 'Croissance' in key:
        key2 = key+'<br>[%/an]'
    if 'Volatilité' in key:
        key2 = key+'<br>[%/an]'
    if 'techno' in key:
        key2 = key+'<br>[%]'

    head = head + '<th>' + key2 + '</th>'
head += '</tr>\n'

asc_desc = ''
sorting = ''
masks = []
mask2 = []
buttons = ''
# decider si c'est ascendant ou descendant
for i in range(len(keys)):
    # si pas float on continue
    if not keys_is_float[i]:
        continue
    if ('Volat' in keys[i]) or ('gini' in keys[i]) or ('sigma' in keys[i]):
        asc_desc += f'          {i}: "asc",   // {keys[i]} : plus petit = vert\n'
    else:
        asc_desc += f'          {i}: "desc",  // {keys[i]} : plus grand = vert\n'

    tmp = f"""
                var min{keys[i]} = parseFloat($('#min{keys[i]}').val(),10);
                var max{keys[i]} = parseFloat($('#max{keys[i]}').val(),10);
                var {keys[i]} = parseFloat(data[{i}]) || 0;
        """
    sorting += tmp

    v1 = f'            (isNaN(min{keys[i]}) || {keys[i]} >= min{keys[i]}) '
    v2 = f'            (isNaN(max{keys[i]}) || {keys[i]} <= max{keys[i]}) '
    masks.append(v1)
    masks.append(v2)
    v1b = f'#min{keys[i]},#max{keys[i]}'
    mask2.append(v1b)

    tmp = f"""
        <div class="filter-item">
        <div><label>{keys[i]} </label><input type="number" size = "5" id="min{keys[i]}"></div>
        <div><label>&#x2192; </label><input type="number" size = "5" id="max{keys[i]}"></div><br>
        </div>
        """
    buttons += tmp

masks = np.asarray(masks)
masks = ' && \n'.join(masks)
mask2 = ','.join(mask2)
mask2 = "'" + mask2 + "'"

dict_params = {'table':v_all
               ,'head': head
               ,'n_etf': len(tbl_bulk)
               ,'date': datetime.now().strftime("%Y-%m-%d %H:%M")
               ,'n_float': np.sum(keys_is_float)
               ,'n_other': np.sum(~keys_is_float)
                ,'asc_desc': asc_desc
                ,'sorting': sorting
                ,'masks': masks
                ,'mask2': mask2
                ,'buttons': buttons
               }
output_from_parsed_template = template.render(params=dict_params)
with open(f"/Users/eartigau/bydoux/website/table_fnb.html", "w") as fh:
    fh.write(output_from_parsed_template)