import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.optimize as opt
from errors import error_prop

from sklearn.impute import SimpleImputer

# Define error ranges calculation function
def err_ranges(x, func, param, sigma):
    """
    Calculate the error ranges for the provided function and parameters.
    
    Parameters:
    - x (array-like): Independent variable values.
    - func (callable): Function used for fitting.
    - param (array-like): Optimized parameters.
    - sigma (array-like): Standard deviations of the parameters.
    
    Returns:
    - lower (array-like): Lower bound of the error range.
    - upper (array-like): Upper bound of the error range.
    """
    import itertools
    
    lower = func(x, *param)
    upper = lower
    uplow = []
    for p, s in zip(param, sigma):
        p_min = p - s
        p_max = p + s
        uplow.append((p_min, p_max))
        
    pmix = list(itertools.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    
    return lower, upper

# Function to read and preprocess data
def read_data(file_paths, selected_countries, year):
    dfs = []
    dataframes_dict_transpose = {}
    exclude_columns = ['Country Code', 'Indicator Name', 'Indicator Code']
    
    for path in file_paths:
        file_name = path.split('.')[0].replace(' ', '_')
        df = pd.read_csv(path, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 67" if x not in exclude_columns else True)
        df.set_index('Country Name', inplace=True)
        df = df.loc[selected_countries]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
        df = df[[str(year)]]
        df_trans = df.transpose()
        df_trans.dropna(axis=0, how="all", inplace=True)
        df_trans.reset_index(inplace=True)
        df.columns = [file_name]
        dfs.append(df)
        dataframes_dict_transpose[file_name] = df_trans
    
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], start=1):
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, suffixes=('', f'_file{i}'))
    
    merged_df.dropna(inplace=True)
    
    return merged_df, dataframes_dict_transpose

# Function to read and preprocess data for fitting
def read_data_for_fit(file_paths, selected_country, start_year, end_year):
    dataframes_dict = {}

    for file_path in file_paths:
        file_name = file_path.split('.')[0]
        df = pd.read_csv(file_path, skiprows=4, usecols=lambda x: x.strip() != "Unnamed: 67" if x not in ['Indicator Name', 'Indicator Code', 'Country Code'] else True)
        df.set_index("Country Name", inplace=True)
        
        if selected_country in df.index:
            df_country = df.loc[[selected_country], str(start_year):str(end_year)].transpose()
            df_country.fillna(df_country.mean(), inplace=True)
            if df_country.isnull().values.any():
                print(f"NaNs found in dataset {file_name} even after fillna: {df_country.isnull().sum()}")
            
            data = {'Year': df_country.index, file_name: df_country[selected_country].values}
            df_fit = pd.DataFrame(data).reset_index(drop=True)
            dataframes_dict[file_name] = df_fit

    return dataframes_dict

# Function to merge multiple DataFrames into one dataset based on a key indicator
def merge_data(dataframes_dict, key_indicator):
    result_df = dataframes_dict[key_indicator]
    
    for key, df in dataframes_dict.items():
        if key != key_indicator and not df.empty and 'Year' in df.columns:
            result_df = pd.merge(result_df, df, on='Year', how='outer', suffixes=('', f'_{key}'))
    
    result_df.drop(columns=[col for col in result_df.columns if col.startswith('index') or col.startswith('Year_')], inplace=True)
    result_df.set_index('Year', inplace=True)
    result_df.reset_index(inplace=True)
    result_df['Year'] = result_df['Year'].astype(int)
    
    return result_df.iloc[:, :2]

# Function to create and display a correlation heatmap
def create_correlation_heatmap(data, year):
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Indicators")
    plt.savefig(f"heatmap_{year}.png", bbox_inches='tight', dpi=300)
    plt.show()

# Function to create and display a scatter matrix
def create_scatter_matrix(data, year):
    pd.plotting.scatter_matrix(data, figsize=(9.0, 9.0))
    plt.tight_layout()
    plt.savefig(f"scatter_matrix_{year}.png", bbox_inches='tight', dpi=300)
    plt.show()

# Function to perform clustering and create cluster plots
def perform_clustering(data, cluster_indicators, year):
    df_fit = data[cluster_indicators].copy()
    scaler = StandardScaler()
    df_fit_scaled = scaler.fit_transform(df_fit)
    
    silhouette_scores = {}
    for ic in range(2, 7):
        kmeans = KMeans(n_clusters=ic, n_init=10)
        kmeans.fit(df_fit_scaled)
        labels = kmeans.labels_
        if len(np.unique(labels)) > 1:
            silhouette_scores[ic] = silhouette_score(df_fit_scaled, labels)
        else:
            silhouette_scores[ic] = -1
    
    for clusters, score in silhouette_scores.items():
        print(f"Silhouette Score for {clusters} clusters: {score}")

    nc = 2 if year == 2012 else 2
    kmeans = KMeans(n_clusters=nc, n_init=10)
    kmeans.fit(df_fit_scaled)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    plt.figure(figsize=(6.0, 6.0))
    for i in range(nc):
        plt.scatter(df_fit_scaled[labels == i, 0], df_fit_scaled[labels == i, 1], label=f'Cluster {i+1}')
    plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=100, label='Centers')

    plt.xlabel(cluster_indicators[0])
    plt.ylabel(cluster_indicators[1])
    plt.legend()
    plt.title(f"Country clusters based on {cluster_indicators[0]} & {cluster_indicators[1]} {year}")
    plt.savefig(f"cluster_{year}.png", bbox_inches='tight', dpi=300)
    plt.show()

    data["Cluster_Labels"] = labels
    data.to_excel(f"cluster_{year}.xlsx")

# Exponential growth function
def exp_growth(t, scale, growth):
    return scale * np.exp(growth * (t - 1950))

def create_data_fit_graph(df_fitted, indicator, country):
    
    """Create line graph with fitting data towards prediction for 10 years
    """

    initial_guess = [1.0, 0.02]
    popt, pcovar = opt.curve_fit(exp_growth, df_fitted["Year"], df_fitted[indicator], p0 = initial_guess, maxfev = 10000)
    print("Fit parameters:", popt)
    # Create a new column with the fitted values
    df_fitted["pop_exp"] = exp_growth(df_fitted["Year"], *popt)
    #plot
    plt.figure()
    plt.plot(df_fitted["Year"], df_fitted[indicator], label = "data")
    plt.plot(df_fitted["Year"], df_fitted["pop_exp"], label = "fit")
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel(indicator)
    plt.title(f"{indicator} in {country} ")

    plt.savefig(f"{indicator}_{country}_fit_graph.png", bbox_inches = 'tight', dpi = 300)
    plt.show()
    print()
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.linspace(2000, 2033)
    pop_exp_growth = exp_growth(years, *popt)
    sigma = error_prop(years, exp_growth, popt, pcovar)
    low = pop_exp_growth - sigma
    up = pop_exp_growth + sigma
    plt.figure()
    plt.title(f"{indicator} in {country} ")
    plt.plot(df_fitted["Year"], df_fitted[indicator], label = "data")
    plt.plot(years, pop_exp_growth, label = "Forecast")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha = 0.5, color = "y", label = "95% Confidence Interval")
    plt.legend(loc = "upper right")
    plt.xlabel("Year")
    plt.ylabel(indicator)
    plt.savefig(f"{indicator}_{country}_predict_graph.png", bbox_inches = 'tight', dpi = 300)
    plt.show()
    # Predict future values
    pop_2030 = exp_growth(np.array([2030]), *popt)
    # Assuming you want predictions for the next 10 years
    sigma_2030 = error_prop(np.array([2030]), exp_growth, popt, pcovar)
    print(f"{indicator} in")
    print("2030:", exp_growth(2030, *popt) / 1.0e6, "Mill.")

    # for next 10 years
    print(f"{indicator} in")
    for year in range(2024, 2034):
        print(f"{indicator} in",year)
        print("2030:", exp_growth(year, *popt) / 1.0e6, "Mill.")
# Function to predict and plot data for the next 50 years
def predict_next_50_years(df_fitted, indicator, country):
    initial_guess = [1.0, 0.02]
    popt, pcovar = opt.curve_fit(exp_growth, df_fitted["Year"], df_fitted[indicator], p0=initial_guess, maxfev=10000)
    years = np.arange(df_fitted["Year"].iloc[-1], df_fitted["Year"].iloc[-1] + 50)
    df_future = pd.DataFrame({"Year": years})
    df_future[indicator] = exp_growth(years, *popt)
    sigma = error_prop(df_future["Year"], exp_growth, popt, pcovar)
    lower, upper = df_future[indicator] - sigma, df_future[indicator] + sigma
    df_future["Lower_Error"] = lower
    df_future["Upper_Error"] = upper

    plt.figure()
    plt.plot(df_future["Year"], df_future[indicator], label="Predicted")
    plt.fill_between(df_future["Year"], df_future["Lower_Error"], df_future["Upper_Error"], color='yellow', alpha=0.7, label="Error range")
    plt.xlabel("Year")
    plt.ylabel(indicator)
    plt.legend()
    plt.title(f"{country} {indicator} Future Prediction")
    plt.savefig(f"{country}_{indicator}_future_prediction.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # File paths
    file_paths = ['Fossil fuel.csv', 'GDP growth.csv', 'CO2 emissions.csv', 'Forest area.csv', 'Access to electricity.csv']
    selected_countries = [
        "United States", "China", "Germany", "Japan", "India", "United Kingdom", 
        "France", "Brazil", "Italy", "Canada", "Russian Federation", "Indonesia", 
        "Mexico", "Spain", "Australia", "Saudi Arabia", "Turkiye", "Nigeria", 
        "Thailand"
    ]
    
    year_2012 = 2010
    year_2022 = 2022
    start_year = 2000
    end_year = 2022

    # Dataset for 2012
    df_10, dataframes_dict_transpose_12 = read_data(file_paths, selected_countries, year_2012)

    # Dataset for 2022
    df_22, dataframes_dict_transpose_22 = read_data(file_paths, selected_countries, year_2022)
    
    # Data for Germany from 2000 to 2022
    country = "Germany"
    dict_Germany = read_data_for_fit(file_paths, country, start_year, end_year)
    
    # Data for United States from 2000 to 2022
    country = "United States"
    dict_United_States = read_data_for_fit(file_paths, country, start_year, end_year)

    # Merging data for different indicators
    data_Fosil_Germany = merge_data(dict_Germany, "Fossil fuel")
    data_Fosil_United_States = merge_data(dict_United_States, "Fossil fuel")
    
    data_electricity_Germany = merge_data(dict_Germany, "Access to electricity")
    data_electricity_United_States = merge_data(dict_United_States, "Access to electricity")
    
    data_GDP_growth_Germany = merge_data(dict_Germany, "GDP growth")
    data_GDP_growth_United_States = merge_data(dict_United_States, "GDP growth")
     
    data_CO2_emissions_Germany = merge_data(dict_Germany, "CO2 emissions")
    data_CO2_emissions_United_States = merge_data(dict_United_States, "CO2 emissions")
    
    # Create correlation heatmap for the years 2012 and 2022
    create_correlation_heatmap(df_10, year_2012)
    create_correlation_heatmap(df_22, year_2022)
    
    # Create scatter matrix for the years 2012 and 2022
    create_scatter_matrix(df_10, year_2012)
    create_scatter_matrix(df_22, year_2022)
    
    # Create country clusters for the years 2012 and 2022
    cluster_indicators = ["CO2_emissions", "Fossil_fuel"]
    perform_clustering(df_10, cluster_indicators, year_2012)
    perform_clustering(df_22, cluster_indicators, year_2022)
    
    # Create data fit graphs
    create_data_fit_graph(data_Fosil_Germany, "Fossil fuel", "Germany")
    create_data_fit_graph(data_Fosil_United_States, "Fossil fuel", "United States")
    create_data_fit_graph(data_CO2_emissions_United_States, "CO2 emissions", "United States")
    create_data_fit_graph(data_CO2_emissions_Germany, "CO2 emissions", "Germany")

   
