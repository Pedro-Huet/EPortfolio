from bs4 import BeautifulSoup
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import re
import requests
import seaborn as sns
from shiny import App, render, ui
import statsmodels.api as sm

folder_path = r"C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\COVID-19"

# Since my csv file weighs more than 1.3gbs and my computer lacks the vram to
# open such a large Dataframe, I'm importing only specific columns and opening 
# the data in chunks. The selected chunk size is optimal for my pc to load

def load_heavy_covid_data(folder_path, chunksize):
    file_path = f"{folder_path}/COVID19MEXICO2021.csv"
    
    columns = ['ENTIDAD_RES', 'MUNICIPIO_RES', 'CLASIFICACION_FINAL',
               'FECHA_DEF']

# I'm also specifying the type of columns that are being used in my dataframe 
# to accelerate the opening of my dataframe

    string_columns = ['FECHA_DEF']

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

# In order to find the name of each of the municipalities, I use a "dictionary"  
# csv tied to the aforementioned database with the names (crosswalk)

def load_data_from_excel(folder_path, nrows, skip_rows):
    file_path = f"{folder_path}/201128 Catalogos.xlsx"
    columns_to_read = ['CLAVE_MUNICIPIO', 'MUNICIPIO']

    sheet_names = pd.ExcelFile(file_path).sheet_names
    data = pd.read_excel(file_path, sheet_name=sheet_names[-1], 
                      usecols=columns_to_read, skiprows=skip_rows, nrows=nrows)
    return data

# Skipping the rows that contain information of municipalities in other states

skip_rows = range(1, 676)
nrows = 125
municipalities = load_data_from_excel(folder_path, nrows, skip_rows)

# I import the population data for each of the municipalities of the State of 
# Mexico using Census Data

def processed_pop_data(folder_path):
    file_path = f"{folder_path}/conjunto_de_datos_iter_15CSV20.csv"
    data = pd.read_csv(file_path, usecols=['NOM_LOC', 'NOM_MUN', 'POBTOT'])
    
# To make the municipality strings match, municipalities are set to all caps

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

# To label which municipalities are considered the Metropolitan Zone of the
# Valley of Mexico, we perform web scraping:

url = 'http://plataforma.seduym.edomex.gob.mx/SIGZonasMetropolitanas/PEIM/descriptiva.do'
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    
# We get all the text and close-in on the first paragraph 

    paragraphs = soup.find_all('p')

    all_text = [paragraph.get_text() for paragraph in paragraphs]
    
    if len(all_text) >= 2:
        first_paragraph = all_text[1]
        
# Parsing through and cleaning the first paragraph to only keep the name of 
# the municipalities
        
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

# Making sure the names of the municipalities all coincide for the merge: 
# municipalities with all caps and adding accents to account for regional 
# writing and pronunciation of some municipalities

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

# We make import the shapefile amd make sure the municipality names match

def prep_state_shapefile(shapefile_path, state_name):
    sm_gpd = gpd.read_file(shapefile_path)
    sm_gpd = sm_gpd[sm_gpd['ADM1_ES'] == state_name].reset_index(drop=True)
    sm_gpd['ADM2_ES'] = sm_gpd['ADM2_ES'].str.upper()
    
# We keep only relevant columns

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

    covid_deaths = edo_mex_f_gpd['Deaths']
    q1, q2, q3 = np.percentile(covid_deaths.dropna(), [25, 50, 75])

    edo_mex_f_gpd['Deaths_quartile'] = np.where(covid_deaths <= q1, 
         'lowest quartile', np.where(covid_deaths <= q2, 'middle-low quartile', 
         np.where(covid_deaths <= q3, 'middle-high quartile', 'top quartile')))
    
    edo_mex_f_gpd.drop(columns=['ADM2_ES'], inplace=True)

    return edo_mex_f_gpd

muni_shapefile_path = r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\mex_admbnda_govmex_20210618_shp\mex_admbnda_adm2_govmex_20210618.shp'
state_name = 'México'

full_state_mex_data = comp_state_mex_data(muni_shapefile_path, state_name)

# Descriptive Statistics Charts

# Separate Plots

def plot_covid_data_by_ZMVM(data, variable, ylabel):
    grouped_data = data.groupby('ZMVM Municipality')[variable].sum()
    grouped_data.index = grouped_data.index.map(
        {1: 'ZMVM Municipalities', 0: 'Non ZMVM Municipalities'})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped_data.plot(kind='bar', color='royalblue' if variable == 'covid-19_cases' else 'firebrick', ax=ax)
    ax.set_title(f'Total {ylabel} by ZMVM Municipality')
    ax.set_xlabel('ZMVM Municipality')
    ax.set_ylabel(f'Total {ylabel}')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()

def plot_covid_vs_population(data, y_variable, ylabel, color):
    data['POBTOT_in_tens_of_thousands'] = data['POBTOT'] / 10000
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        x='POBTOT_in_tens_of_thousands',
        y=y_variable,
        data=data,
        scatter_kws={'s': 50, 'color': color},
        line_kws={'color': 'black'},
        ax=ax
    )
    ax.set_title(f'Linear Relationship between Population and {ylabel}')
    ax.set_xlabel('Population (in tens of thousands)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()

plot_covid_data_by_ZMVM(full_state_mex_data, 'covid-19_cases', 'Covid-19 Cases')
plot_covid_data_by_ZMVM(full_state_mex_data, 'Deaths', 'Covid-19 Deaths')

plot_covid_vs_population(full_state_mex_data, 'covid-19_cases', 'Covid-19 Cases', 'forestgreen')
plot_covid_vs_population(full_state_mex_data, 'Deaths', 'Covid-19 Deaths', 'darkred')


# Joint Plot

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

def plot_covid_data_by_ZMVM(data, variable, ylabel, ax):
    grouped_data = data.groupby('ZMVM Municipality')[variable].sum()
    grouped_data.index = grouped_data.index.map(
                      {1: 'ZMVM Municipalities', 0: 'Non ZMVM Municipalities'})
    grouped_data.plot(kind='bar',
     color='royalblue' if variable == 'covid-19_cases' else 'firebrick', ax=ax)
    ax.set_title(f'Total {ylabel} by ZMVM Municipality')
    ax.set_xlabel('ZMVM Municipality')
    ax.set_ylabel(f'Total {ylabel}')
    ax.tick_params(axis='x', rotation=0)

def plot_covid_vs_population(data, y_variable, ylabel, color, ax):
    data['POBTOT_in_tens_of_thousands'] = data['POBTOT'] / 10000
    sns.regplot(
        x='POBTOT_in_tens_of_thousands',
        y=y_variable,
        data=data,
        scatter_kws={'s': 50, 'color': color},
        line_kws={'color': 'black'},
        ax=ax
    )
    ax.set_title(f'Linear Relationship between Population and {ylabel}')
    ax.set_xlabel('Population (in tens of thousands)')
    ax.set_ylabel(ylabel)

plot_covid_data_by_ZMVM(full_state_mex_data, 'covid-19_cases', 
                                                   'Covid-19 Cases', axs[0, 0])
plot_covid_data_by_ZMVM(full_state_mex_data, 'Deaths',
                                                  'Covid-19 Deaths', axs[0, 1])
plot_covid_vs_population(full_state_mex_data, 'covid-19_cases', 
                                    'Covid-19 Cases', 'forestgreen', axs[1, 0])
plot_covid_vs_population(full_state_mex_data, 'Deaths', 'Covid-19 Deaths',
                                                          'darkred', axs[1, 1])

plt.tight_layout()
plt.show()

# Map Chart

def plot_map_by_cases_zmvm(data, ax):
    colors = {
        'lowest quartile': 'green',
        'middle-low quartile': 'yellow',
        'middle-high quartile': 'darkorange',
        'top quartile': 'red'
    }

    labels = ["Least Covid-19 cases", "Lower amount of Covid-19 cases",
              "Higher amount of Covid-19 cases", "Most Covid-19 cases"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-high quartile', 'middle-low quartile',
        'top quartile']])

    data.plot(column='Cases_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.75, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})
    
# To also visualize the ZMVM municipalities on the map 

    zmvm_municipalities = data[data['ZMVM Municipality'] == 1]
    centroids = zmvm_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='black', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, 'ZMVM Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
        'State of Mex Municipalities per Covid-19 cases (2020 - 2021)',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

plt.show()

def plot_map_by_deaths_zmvm(data, ax):
    colors = {
        'lowest quartile': 'green',
        'middle-low quartile': 'yellow',
        'middle-high quartile': 'darkorange',
        'top quartile': 'red'
    }

    labels = ["Least Covid-19 deaths", "Lower amount of Covid-19 deaths",
              "Higher amount of Covid-19 deaths", "Most Covid-19 deaths"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-high quartile', 'middle-low quartile',
        'top quartile']])

    data.plot(column='Deaths_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.75, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})

    zmvm_municipalities = data[data['ZMVM Municipality'] == 1]
    centroids = zmvm_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='black', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, 'ZMVM Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
       'State of Mex Municipalities per Covid-19 deaths (2020 - 2021)',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

def plot_map_by_cases_pop(data, ax):
    colors = {
        'lowest quartile': 'green',
        'middle-low quartile': 'yellow',
        'middle-high quartile': 'darkorange',
        'top quartile': 'red'
    }

    labels = ["Least Covid-19 cases", "Lower amount of Covid-19 cases",
              "Higher amount of Covid-19 cases", "Most Covid-19 cases"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-high quartile', 'middle-low quartile',
        'top quartile']])

    data.plot(column='Cases_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.75, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})

# To also visualize the Highly populated municipalities on the map

    h_pop_municipalities = data[data['High_pop_Municipality'] == 1]
    centroids = h_pop_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='blue', marker='D', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, 'Highly Populated Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
        'State of Mex Municipalities per Covid-19 cases (2020 - 2021)',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

def plot_map_by_deaths_pop(data, ax):
    colors = {
        'lowest quartile': 'green',
        'middle-low quartile': 'yellow',
        'middle-high quartile': 'darkorange',
        'top quartile': 'red'
    }

    labels = ["Least Covid-19 deaths", "Lower amount of Covid-19 deaths",
              "Higher amount of Covid-19 deaths", "Most Covid-19 deaths"]

    colormap = ListedColormap([colors[category] for category in [
        'lowest quartile', 'middle-high quartile', 'middle-low quartile',
        'top quartile']])

    data.plot(column='Deaths_quartile', cmap=colormap, legend=True,
              categorical=True, edgecolor='black', linewidth=0.5,
              missing_kwds={'color': 'lightgrey'}, ax=ax,
              legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (0.75, 0.09),
                           'title': 'Quartile Categories', 'labels': labels})

    h_pop_municipalities = data[data['High_pop_Municipality'] == 1]
    centroids = h_pop_municipalities['geometry'].centroid
    ax.scatter(centroids.x, centroids.y, color='blue', marker='D', s=15)

    label_x = data.geometry.bounds['maxx'].max()
    label_y = data.geometry.bounds['miny'].min()
    ax.text(label_x, label_y, 'Highly Populated Municipality', fontsize=10,
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(
       'State of Mex Municipalities per Covid-19 deaths (2020 - 2021)',
        fontsize=16, ha='center', x=0.5, y=1.05)

    return ax

def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# To compare the effect of both variables, we display them on the same graph

#Separate Maps

fig, ax = plt.subplots(figsize=(8, 7))
plot_map_by_cases_zmvm(full_state_mex_data, ax)
remove_ticks(ax)  # Apply the function to remove ticks
plt.show()

fig, ax = plt.subplots(figsize=(8, 7))
plot_map_by_deaths_zmvm(full_state_mex_data, ax)
remove_ticks(ax)  # Apply the function to remove ticks
plt.show()

fig, ax = plt.subplots(figsize=(8, 7))
plot_map_by_cases_pop(full_state_mex_data, ax)
remove_ticks(ax)  # Apply the function to remove ticks
plt.show()

fig, ax = plt.subplots(figsize=(8, 7))
plot_map_by_deaths_pop(full_state_mex_data, ax)
remove_ticks(ax)  # Apply the function to remove ticks
plt.show()

#Joint Map

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

plot_map_by_cases_zmvm(full_state_mex_data, axs[0, 0])
remove_ticks(axs[0, 0])

plot_map_by_deaths_zmvm(full_state_mex_data, axs[0, 1])
remove_ticks(axs[0, 1])

plot_map_by_cases_pop(full_state_mex_data, axs[1, 0])
remove_ticks(axs[1, 0])

plot_map_by_deaths_pop(full_state_mex_data, axs[1, 1])
remove_ticks(axs[1, 1])

plt.tight_layout()
plt.show()

# Regression Analysis

# Simple linear regressions between the two main variables:
# Population in a municipality and ZMVM Municipality

def perform_linear_regression_to_image(X, y, filename):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.text(0.1, 0.1, str(summary), fontsize=10, family='monospace')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.show()

variable_combinations = [
    ('POBTOT_in_tens_of_thousands', 'covid-19_cases', 'summary_cases.png'),
    ('POBTOT_in_tens_of_thousands', 'Deaths', 'summary_deaths.png'),
    ('ZMVM Municipality', 'covid-19_cases', 'summary_cases_zmvm.png'),
    ('ZMVM Municipality', 'Deaths', 'summary_deaths_zmvm.png')
]

for predictor, target, filename in variable_combinations:
    X = full_state_mex_data[[predictor]]
    y = full_state_mex_data[target]
    perform_linear_regression_to_image(X, y, filename)


# Multivariate linear regressions between with two main variables:

def regression_summary_to_image(X, y, filename):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.text(0.1, 0.1, str(summary), fontsize=10, family='monospace')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.show()

# Perform regression and generate summary images
variables = [
    (full_state_mex_data[['POBTOT_in_tens_of_thousands', 'ZMVM Municipality']], 
     full_state_mex_data['covid-19_cases'], 'summary_cases.png'),
    
    (full_state_mex_data[['POBTOT_in_tens_of_thousands', 'ZMVM Municipality']], 
     full_state_mex_data['Deaths'], 'summary_deaths.png')
]

for X, y, filename in variables:
    regression_summary_to_image(X, y, filename)

###############################################################################

h_logo = 'https://d11jve6usk2wa9.cloudfront.net/platform/10747/assets/logo.png'

app_ui = ui.page_fluid(ui.panel_title("The state of the State of Mexico \
                                      during a state of emergency", 
                                         "Pedro H. Final Project"),
    ui.row(
        ui.column(4, ui.img(src=h_logo, height=100, width=288)),
        ui.column(4, ui.h1('Final Project'), ui.hr())),
    ui.row(ui.layout_sidebar(
        ui.panel_sidebar(ui.output_image(id="logo")),
        ui.panel_main(ui.output_text(id="student_info"), ui.output_text(
        id="course_info"), ui.output_text(id="term_info")
                  ))),
     ui.row(ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_select(id="variable_select", 
                    label="Select your variable of interest", choices=[
                                      "ZMVM Municipality", "Population Size"], 
                                                 selected="ZMVM Municipality"),
                ui.input_select(id="metric_select", 
                                label="Select your metric of of interest", 
                                 choices=['Covid-19 cases', 'Covid-19 deaths'], 
                                                    selected='Covid-19 cases'),
                ui.input_select(id="evidence_select", 
                                label="Select the evidence", choices=[
                                    'Descriptive Statistics', 'Mapping', 
                            'Regression'], selected='Descriptive Statistics')),
                 ui.panel_main(
                     ui.output_text("Description"),
                     ui.output_image("plot")
             )
             )
         )
     )

def server(input, output, session):
    
    @output
    @render.image
    def logo():
        ofile = r"C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\harris logo.png"
    
        return {'src': ofile, 'contentType': 'image/png'}
    
    @output
    @render.text
    def student_info():
        s_info = 'Student: Pedro Huet López'
        return s_info
    
    @output
    @render.text
    def course_info(): 
        c_info = 'Course: Data and Programming for Public Policy \
        II - Python Programming'
        return c_info
    @output
    @render.text
    def term_info():
        q_info = 'Quarter: Autumn 2023'
        return q_info
    
    @output
    @render.text
    def Description():
        info = {
            'ZMVM Municipality': 'ZMVM municipalities seem to be highly \
                                    associated with covid-19 cases and deaths',
            'Population Size': 'The evidence suggests that, the higher the \
            population in the municipality, the more covid-19 cases and deaths'
            }
        return info.get(input.variable_select())


    @output
    @render.image()
    def plot():
        variable_select = input.variable_select()
        metric_select = input.metric_select()
        evidence_select = input.evidence_select()
        
        plots_im = {
            ('ZMVM Municipality', 'Covid-19 cases', 'Descriptive Statistics'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\ZMVM Descriptive Statistics covid-19.png',
            ('ZMVM Municipality', 'Covid-19 cases', 'Mapping'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\\ZMVM Map covid-19 cases.png',
            ('ZMVM Municipality', 'Covid-19 cases', 'Regression'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\\OLS Regression ZMVM Cases.png',
            ('ZMVM Municipality', 'Covid-19 deaths', 'Descriptive Statistics'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\ZMVM Descriptive Statistics deaths.png',
            ('ZMVM Municipality', 'Covid-19 deaths', 'Mapping'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\ZMVM Map deaths.png',
            ('ZMVM Municipality', 'Covid-19 deaths', 'Regression'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\OLS Regression ZMVM Deaths.png',
            ('Population Size', 'Covid-19 cases', 'Descriptive Statistics'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\Pop Descriptive Statistics covid-19.png',
            ('Population Size', 'Covid-19 cases', 'Mapping'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\Pop Map covid-19 cases.png',
            ('Population Size', 'Covid-19 cases', 'Regression'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\OLS Regression Total Population Cases.png',
            ('Population Size', 'Covid-19 deaths', 'Descriptive Statistics'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\Pop Descriptive Statistics deaths.png',
            ('Population Size', 'Covid-19 deaths', 'Mapping'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\Pop Map covid-19 deaths.png',
            ('Population Size', 'Covid-19 deaths', 'Regression'): r'C:\Users\pedro\OneDrive\Documentos\Relevant Documents\School Work\Programming with Python\Final Project\my_app1\OLS Regression Total Population Deaths.png',
            }

        src = plots_im.get((variable_select, metric_select, evidence_select))
        return {'src':src, 'width':'60%'}

app = App(app_ui, server)

# Sources:
# Homework 1
# Homework 2
# Homework 3
# Homework 4
# https://www.statsmodels.org/stable/index.html