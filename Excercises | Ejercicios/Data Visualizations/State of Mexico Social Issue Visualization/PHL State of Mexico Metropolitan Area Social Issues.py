from bs4 import BeautifulSoup
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import re
import requests

folder_path = r"C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\COVID-19"


def load_heavy_covid_data(folder_path, chunksize):
    file_path = f"{folder_path}/COVID19MEXICO2021.csv"
    
    columns = ['ENTIDAD_UM', 'ENTIDAD_RES', 'MUNICIPIO_RES', 'HIPERTENSION', 
               'CLASIFICACION_FINAL', 'EDAD', 'RENAL_CRONICA', 'OBESIDAD', 
               'INDIGENA', 'SEXO', 'INTUBADO', 'TABAQUISMO', 
               'NEUMONIA', 'CARDIOVASCULAR', 'EPOC', 'DIABETES', 'ASMA', 
               'EMBARAZO', 'FECHA_DEF']



    string_columns = ['FECHA_DEF', ]

    dtypes = {col: int for col in columns if col not in string_columns}
    for col in string_columns:
        dtypes[col] = object

    chunks = []
    for chunk in pd.read_csv(file_path, usecols=columns, chunksize=chunksize):
        chunk = chunk.astype(dtypes)
        chunk = chunk[(chunk['ENTIDAD_RES'] == 15) & 
                      (chunk['CLASIFICACION_FINAL'] != 7) & 
                      (chunk['MUNICIPIO_RES'] != 999)]
        chunks.append(chunk)

    df = pd.concat(chunks)
    df.reset_index(drop=True, inplace=True)

    return df

chunksize = 12500
st_mex_data = load_heavy_covid_data(folder_path, chunksize)


def load_data_from_excel(folder_path, nrows, skip_rows):
    file_path = f"{folder_path}/201128 Catalogos.xlsx"
    columns_to_read = ['CLAVE_MUNICIPIO', 'MUNICIPIO']

    sheet_names = pd.ExcelFile(file_path).sheet_names
    data = pd.read_excel(file_path, sheet_name=sheet_names[-1], 
                      usecols=columns_to_read, skiprows=skip_rows, nrows=nrows)
    return data

skip_rows = range(1, 676)
nrows = 125
municipalities = load_data_from_excel(folder_path, nrows, skip_rows)

def processed_pop_data(folder_path):
    file_path = f"{folder_path}/conjunto_de_datos_iter_15CSV20.csv"
    data = pd.read_csv(file_path, usecols=['NOM_LOC', 'NOM_MUN', 'POBTOT'])

    data['NOM_MUN'] = data['NOM_MUN'].str.upper()
    md_data = data[data['NOM_LOC'].str.strip() == 'Total del Municipio'].copy()
    md_data.drop('NOM_LOC', axis=1, inplace=True)
    md_data.reset_index(drop=True, inplace=True)
    
    return md_data

population_data = processed_pop_data(folder_path)

def prp_covid_data(st_mex_data, municipalities, population_data):
    
    merged_data = pd.merge(st_mex_data, municipalities,
                           left_on='MUNICIPIO_RES', right_on='CLAVE_MUNICIPIO',
                           how='inner')

    second_merge = pd.merge(merged_data, population_data,
                            left_on='MUNICIPIO', right_on='NOM_MUN',
                            how='inner')

    second_merge['Deaths'] = second_merge['FECHA_DEF'].apply(
                                       lambda x: 0 if x == '9999-99-99' else 1)

    covid_cases = second_merge.groupby('MUNICIPIO').size().reset_index(
                                                         name='covid-19_cases')
    deaths = second_merge.groupby('MUNICIPIO')['Deaths'].sum().reset_index()

    result = pd.merge(covid_cases, deaths, on='MUNICIPIO')
    result = pd.merge(result, population_data[['NOM_MUN', 'POBTOT']], 
                                       left_on='MUNICIPIO', right_on='NOM_MUN')
    result.drop('NOM_MUN', axis=1, inplace=True)

    return result

enh_st_mex_data = prp_covid_data(st_mex_data, municipalities, population_data)

url = 'http://plataforma.seduym.edomex.gob.mx/SIGZonasMetropolitanas/PEIM/descriptiva.do'
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    
    paragraphs = soup.find_all('p')

    all_text = [paragraph.get_text() for paragraph in paragraphs]
    
    if len(all_text) >= 2:
        first_paragraph = all_text[1]
        
def process_paragraph_from_Acolman(first_paragraph):
    modified_paragraph = first_paragraph.replace(' y ', ', ')
    modified_paragraph = modified_paragraph.replace("z ", "z,")

    lines = modified_paragraph.split(',')
    found_acolman = False
    fil_lines = []

    for line in lines:
        if found_acolman:
            fil_lines.append(line)
        elif "Acolman" in line:
            found_acolman = True
            fil_lines.append(line.split("Acolman", 1)[1])
    if found_acolman:
        fil_lines = ["Acolman" if line.strip(
                                       ) == "" else line for line in fil_lines]
        fil_lines = [line.lstrip() for line in fil_lines]
        fil_lines = [re.sub(r'\.\s*', '', line) for line in fil_lines]
        fil_lines = [line.upper() for line in fil_lines]
        fil_lines = [line.replace("TEOTIHUACAN", "TEOTIHUACÁN").replace(
                     "COACALCO DE BERRIOZABAL", "COACALCO DE BERRIOZÁBAL"
                                                       ) for line in fil_lines]
        
    return fil_lines

processed_lines = process_paragraph_from_Acolman(first_paragraph)

enh_st_mex_data['ZMVM Municipality'] = enh_st_mex_data['MUNICIPIO'].isin(
                                                   processed_lines).astype(int)

enh_st_mex_data['High_pop_Municipality'] = (enh_st_mex_data['POBTOT'
                                                        ] >= 50000).astype(int)

def prep_state_shapefile(shapefile_path, state_name):
    sm_gpd = gpd.read_file(shapefile_path)
    sm_gpd = sm_gpd[sm_gpd['ADM1_ES'] == state_name].reset_index(drop=True)
    sm_gpd['ADM2_ES'] = sm_gpd['ADM2_ES'].str.upper()

    columns_to_keep = ['ADM2_ES', 'geometry']
    sm_gpd = sm_gpd[columns_to_keep]
    
    return sm_gpd

def comp_state_mex_data(muni_shapefile_path, state_name):
    state_mex_data = prep_state_shapefile(muni_shapefile_path, state_name)

    edo_mex_f_gpd = pd.merge(enh_st_mex_data, state_mex_data, 
                          left_on='MUNICIPIO', right_on='ADM2_ES', how='inner')

    edo_mex_f_gpd = edo_mex_f_gpd.set_geometry('geometry')

    covid_cases = edo_mex_f_gpd['covid-19_cases']
    q1, q2, q3 = np.percentile(covid_cases.dropna(), [25, 50, 75])

    edo_mex_f_gpd['Cases_quartile'] = np.where(covid_cases <= q1, 
          'lowest quartile', np.where(covid_cases <= q2, 'middle-low quartile',
          np.where(covid_cases <= q3, 'middle-high quartile', 'top quartile')))
    
    edo_mex_f_gpd.drop(columns=['ADM2_ES'], inplace=True)

    return edo_mex_f_gpd

muni_shapefile_path = r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\mex_admbnda_govmex_20210618_shp\mex_admbnda_adm2_govmex_20210618.shp'
state_name = 'México'

full_state_mex_data = comp_state_mex_data(muni_shapefile_path, state_name)

# Map Chart

def plot_map_by_cases(data, ax):
    colors = {
        'lowest quartile': 'navajowhite',
        'middle-low quartile': 'orange',
        'middle-high quartile': 'darkorange',
        'top quartile': 'orangered'
    }

    
    labels = ["In the lowest Covid-19 case quartile", "In the lower Covid-19 case quartile",
              "In the higher Covid-19 case quartile", "In the top Covid-19 case quartile"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-low quartile', 'middle-high quartile',
        'top quartile']])

    data.plot(column='Cases_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.66, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})
    
# To also visualize the ZMVM municipalities on the map 

    zmvm_municipalities = data[data['ZMVM Municipality'] == 1]
    centroids = zmvm_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='black', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, '\u2022 ZMVM Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
        'Municipalities per Total registered Covid-19 cases',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

fig, ax = plt.subplots(figsize=(10, 10))
plot_map_by_cases(full_state_mex_data, ax)
plt.show()

###############################################################################

def load_heavy_covid_data(folder_path, chunksize):
    file_path = f"{folder_path}/COVID19MEXICO2021.csv"
    
    columns = ['ENTIDAD_UM', 'ENTIDAD_RES', 'MUNICIPIO_RES', 'HIPERTENSION', 
               'CLASIFICACION_FINAL', 'EDAD', 'RENAL_CRONICA', 'OBESIDAD', 
               'INDIGENA', 'SEXO', 'INTUBADO', 'TABAQUISMO', 
               'NEUMONIA', 'CARDIOVASCULAR', 'EPOC', 'DIABETES', 'ASMA', 
               'EMBARAZO', 'FECHA_DEF']

    string_columns = ['FECHA_DEF']

    dtypes = {col: int for col in columns if col not in string_columns}
    for col in string_columns:
        dtypes[col] = object

    chunks = []
    for chunk in pd.read_csv(file_path, usecols=columns, chunksize=chunksize):
        chunk = chunk.astype(dtypes)
        chunk = chunk[(chunk['ENTIDAD_RES'] == 15) &  
                      (chunk['MUNICIPIO_RES'] != 999)]
        chunks.append(chunk)

    df = pd.concat(chunks)
    df.reset_index(drop=True, inplace=True)

    return df

chunksize = 12500
st_mex_data2 = load_heavy_covid_data(folder_path, chunksize)


def load_data_from_excel(folder_path, nrows, skip_rows):
    file_path = f"{folder_path}/201128 Catalogos.xlsx"
    columns_to_read = ['CLAVE_MUNICIPIO', 'MUNICIPIO']

    sheet_names = pd.ExcelFile(file_path).sheet_names
    data = pd.read_excel(file_path, sheet_name=sheet_names[-1], 
                      usecols=columns_to_read, skiprows=skip_rows, nrows=nrows)
    return data

skip_rows = range(1, 676)
nrows = 125
municipalities = load_data_from_excel(folder_path, nrows, skip_rows)

def processed_pop_data(folder_path):
    file_path = f"{folder_path}/conjunto_de_datos_iter_15CSV20.csv"
    data = pd.read_csv(file_path, usecols=['NOM_LOC', 'NOM_MUN', 'POBTOT'])

    data['NOM_MUN'] = data['NOM_MUN'].str.upper()
    md_data = data[data['NOM_LOC'].str.strip() == 'Total del Municipio'].copy()
    md_data.drop('NOM_LOC', axis=1, inplace=True)
    md_data.reset_index(drop=True, inplace=True)
    
    return md_data

population_data = processed_pop_data(folder_path)

def prp_covid_data(st_mex_data2, municipalities, population_data):
    
    merged_data = pd.merge(st_mex_data2, municipalities,
                           left_on='MUNICIPIO_RES', right_on='CLAVE_MUNICIPIO',
                           how='inner')

    second_merge = pd.merge(merged_data, population_data,
                            left_on='MUNICIPIO', right_on='NOM_MUN',
                            how='inner')

    second_merge['Deaths'] = second_merge['FECHA_DEF'].apply(
                                       lambda x: 0 if x == '9999-99-99' else 1)

    covid_cases = second_merge.groupby('MUNICIPIO').size().reset_index(
                                                         name='covid-19_cases')
    deaths = second_merge.groupby('MUNICIPIO')['Deaths'].sum().reset_index()
    obesity = second_merge.groupby('MUNICIPIO')['OBESIDAD'].sum().reset_index()
    smoker = second_merge.groupby('MUNICIPIO')['TABAQUISMO'].sum().reset_index()

    # Merge multiple DataFrames using a sequence and 'on' as the common column
    result = pd.merge(covid_cases, deaths, on='MUNICIPIO')
    result = pd.merge(result, obesity, on='MUNICIPIO')
    result = pd.merge(result, smoker, on='MUNICIPIO')

    # Merge with population_data and drop 'NOM_MUN' column
    result = pd.merge(result, population_data[['NOM_MUN', 'POBTOT']], 
                                       left_on='MUNICIPIO', right_on='NOM_MUN')
    result.drop('NOM_MUN', axis=1, inplace=True)

    return result

enh_st_mex_data2 = prp_covid_data(st_mex_data2, municipalities, population_data)


url = 'http://plataforma.seduym.edomex.gob.mx/SIGZonasMetropolitanas/PEIM/descriptiva.do'
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    
    paragraphs = soup.find_all('p')

    all_text = [paragraph.get_text() for paragraph in paragraphs]
    
    if len(all_text) >= 2:
        first_paragraph = all_text[1]
        
def process_paragraph_from_Acolman(first_paragraph):
    modified_paragraph = first_paragraph.replace(' y ', ', ')
    modified_paragraph = modified_paragraph.replace("z ", "z,")

    lines = modified_paragraph.split(',')
    found_acolman = False
    fil_lines = []

    for line in lines:
        if found_acolman:
            fil_lines.append(line)
        elif "Acolman" in line:
            found_acolman = True
            fil_lines.append(line.split("Acolman", 1)[1])
    if found_acolman:
        fil_lines = ["Acolman" if line.strip(
                                       ) == "" else line for line in fil_lines]
        fil_lines = [line.lstrip() for line in fil_lines]
        fil_lines = [re.sub(r'\.\s*', '', line) for line in fil_lines]
        fil_lines = [line.upper() for line in fil_lines]
        fil_lines = [line.replace("TEOTIHUACAN", "TEOTIHUACÁN").replace(
                     "COACALCO DE BERRIOZABAL", "COACALCO DE BERRIOZÁBAL"
                                                       ) for line in fil_lines]
        
    return fil_lines

processed_lines = process_paragraph_from_Acolman(first_paragraph)

enh_st_mex_data2['ZMVM Municipality'] = enh_st_mex_data2['MUNICIPIO'].isin(
                                                   processed_lines).astype(int)

enh_st_mex_data2['High_pop_Municipality'] = (enh_st_mex_data2['POBTOT'
                                                        ] >= 50000).astype(int)

def prep_state_shapefile(shapefile_path, state_name):
    sm_gpd = gpd.read_file(shapefile_path)
    sm_gpd = sm_gpd[sm_gpd['ADM1_ES'] == state_name].reset_index(drop=True)
    sm_gpd['ADM2_ES'] = sm_gpd['ADM2_ES'].str.upper()

    columns_to_keep = ['ADM2_ES', 'geometry']
    sm_gpd = sm_gpd[columns_to_keep]
    
    return sm_gpd

def comp_state_mex_data(muni_shapefile_path, state_name):
    state_mex_data = prep_state_shapefile(muni_shapefile_path, state_name)

    edo_mex_f_gpd = pd.merge(enh_st_mex_data2, state_mex_data, 
                          left_on='MUNICIPIO', right_on='ADM2_ES', how='inner')

    edo_mex_f_gpd = edo_mex_f_gpd.set_geometry('geometry')

    covid_cases = edo_mex_f_gpd['covid-19_cases']
    q1, q2, q3 = np.percentile(covid_cases.dropna(), [25, 50, 75])

    edo_mex_f_gpd['Cases_quartile'] = np.where(covid_cases <= q1, 
          'lowest quartile', np.where(covid_cases <= q2, 'middle-low quartile',
          np.where(covid_cases <= q3, 'middle-high quartile', 'top quartile')))
    
    obesity = edo_mex_f_gpd['OBESIDAD']
    q1, q2, q3 = np.percentile(obesity.dropna(), [25, 50, 75])

    edo_mex_f_gpd['Obesity quartile'] = np.where(covid_cases <= q1, 
          'lowest quartile', np.where(obesity <= q2, 'middle-low quartile',
          np.where(obesity <= q3, 'middle-high quartile', 'top quartile')))
    
    smoker = edo_mex_f_gpd['TABAQUISMO']
    q1, q2, q3 = np.percentile(smoker.dropna(), [25, 50, 75])

    edo_mex_f_gpd['Smoker_quartile'] = np.where(smoker <= q1, 
          'lowest quartile', np.where(smoker <= q2, 'middle-low quartile',
          np.where(smoker <= q3, 'middle-high quartile', 'top quartile')))
    
    edo_mex_f_gpd.drop(columns=['ADM2_ES'], inplace=True)

    return edo_mex_f_gpd

muni_shapefile_path = r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\mex_admbnda_govmex_20210618_shp\mex_admbnda_adm2_govmex_20210618.shp'
state_name = 'México'

full_state_mex_data2 = comp_state_mex_data(muni_shapefile_path, state_name)

# Map Chart

def plot_map_by_obesity(data, ax):
    colors = {
        'lowest quartile': 'blanchedalmond',
        'middle-low quartile': 'pink',
        'middle-high quartile': 'hotpink',
        'top quartile': 'deeppink'
    }

    
    labels = ["In the lowest obese population quartile", "In the lower obese population quartile",
              "In the higher obese population quartile", "In the top obese population quartile"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-low quartile', 'middle-high quartile',
        'top quartile']])

    data.plot(column='Obesity quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.66, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})
    
# To also visualize the ZMVM municipalities on the map 

    zmvm_municipalities = data[data['ZMVM Municipality'] == 1]
    centroids = zmvm_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='black', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, '\u2022 ZMVM Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
        'Municipalities per population with obesity accessing medical services',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

fig, ax = plt.subplots(figsize=(10, 10))
plot_map_by_obesity(full_state_mex_data2, ax)
plt.show()

###############################################################################
    
# Map Chart

def plot_map_by_smoking(data, ax):
    colors = {
     'lowest quartile': 'khaki',
     'middle-low quartile': 'greenyellow',
     'middle-high quartile': 'chartreuse',
     'top quartile': 'lime'
    }

    
    labels = ["In the lowest smoker population quartile", "In the lower smoker population quartile",
              "In the higher smoker population quartile", "In the top smoker population quartile"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-low quartile', 'middle-high quartile',
        'top quartile']])

    data.plot(column='Smoker_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.66, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})
    
# To also visualize the ZMVM municipalities on the map 

    zmvm_municipalities = data[data['ZMVM Municipality'] == 1]
    centroids = zmvm_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='black', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, '\u2022 ZMVM Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
        'Municipalities per smoker population accessing medical services',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

fig, ax = plt.subplots(figsize=(10, 10))
plot_map_by_smoking(full_state_mex_data2, ax)
plt.show()
###############################################################################

def load_filtered_data(csv_file_path):
    data = pd.read_csv(csv_file_path, encoding='latin1')

    filtered_data = data[(data.iloc[:, 0].isin([2020, 2021])) & 
                         (data.iloc[:, 2] == 'México') &
                         (data.iloc[:, 6] == 'Homicidio')]

    filtered_data['Total homicides'] = filtered_data.iloc[:, -12:].sum(axis=1)

    filtered_data['Municipio'] = filtered_data['Municipio'].str.upper()

    grouped_data = filtered_data.groupby(filtered_data.iloc[:, 4])['Total homicides'].sum().reset_index()

    return grouped_data

csv_file_path = r"C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\COVID-19\Municipal-Delitos-2015-2023_oct2023.csv"

grouped_data = load_filtered_data(csv_file_path)

merged_data = pd.merge(full_state_mex_data2, grouped_data,
                       left_on='MUNICIPIO', right_on='Municipio', how='inner')

homicides = merged_data['Total homicides']
q1, q2, q3 = np.percentile(homicides.dropna(), [25, 50, 75])

merged_data['Homicides_quartile'] = np.where(homicides <= q1, 
      'lowest quartile', np.where(homicides <= q2, 'middle-low quartile',
      np.where(homicides <= q3, 'middle-high quartile', 'top quartile')))

###############################################################################

# Map Chart

def plot_map_by_homicides(data, ax):
    colors = {
        'lowest quartile': 'peachpuff',
        'middle-low quartile': 'lightpink',
        'middle-high quartile': 'lightcoral',
        'top quartile': 'red'
    }

    
    labels = ["In the lowest homicide count quartile", "In the lower homicide count quartile",
              "In the higher homicide count quartile", "In the top homicide count quartile"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-low quartile', 'middle-high quartile',
        'top quartile']])

    data.plot(column='Homicides_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.66, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})
    
# To also visualize the ZMVM municipalities on the map 

    zmvm_municipalities = data[data['ZMVM Municipality'] == 1]
    centroids = zmvm_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='black', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, '\u2022 ZMVM Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
        'Municipalities per registered Homicide Count',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

fig, ax = plt.subplots(figsize=(10, 10))
plot_map_by_homicides(merged_data, ax)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

plot_map_by_cases(full_state_mex_data, axes[0, 0])
plot_map_by_obesity(full_state_mex_data2, axes[0, 1])
plot_map_by_smoking(full_state_mex_data2, axes[1, 0])
plot_map_by_homicides(merged_data, axes[1, 1])


for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])


plt.tight_layout()

plt.show()



