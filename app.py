
import streamlit as st
import pandas as pd
from app.src import config
import base64
import plotly.express as px
import ast
import matplotlib.pyplot as plt
from matchms import Fragments, calculate_scores, Spectrum
import numpy as np

from io import BytesIO
from PIL import Image
import io

# Define the plot_kmd functions directly in the code
def plot_kmd(data, size, fragment="44"):
    kmd_f = "_".join(("kmd", fragment))
    plt.figure(figsize=(6,4))
    plt.scatter(data['precMz'][0:size], data[kmd_f][0:size], marker='.', color='green')
    plt.title("KMD")
    plt.xlabel("precMz")
    plt.ylabel(kmd_f)
    # Save the plot to an in-memory buffer with lower dpi

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Display the saved plot in Streamlit with adjusted width
    st.image(buffer.getvalue(), use_column_width=True)

def plot_kmd2(data, size, fragment="44"):
    kmd_f = "_".join(("kmd", fragment))
    km_f = "_".join(("km", fragment))
    x = km_f
    y = kmd_f
    plt.figure(figsize=(6,4))
    plt.scatter(data[x][0:size], data[y][0:size], marker='.', color='green')
    plt.title("KMD")
    plt.xlabel(x)
    plt.ylabel(y)
    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    # Display the saved plot in Streamlit with adjusted width
    st.image(buffer.getvalue(), use_column_width=True)
def plot_spectrum(row):
    mz_array = row["m/z array"]
    intensity_array = row["intensity array"]

    value_list = row["m/z array"].strip('[]').split()

    # Convert the list of values to floats and then create a NumPy array
    mz_array = np.array([float(value) for value in value_list])

    # Remove '[' and ']' characters and split the string into a list of values
    value_list = row["intensity array"].strip('[]').split()

    # Convert the list of values to floats and then create a NumPy array
    numpy_array = np.array([float(value) for value in value_list])


    # Create the Spectrum object
    spectrum = Spectrum(mz=mz_array, intensities=numpy_array)


    # Get the plot as a Matplotlib figure and axes
    fig, ax = spectrum.plot(figsize=(4, 2), dpi=100)

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)



def plot_spectrum_comp(row1, row2):
    spec1 = Spectrum(mz=row1["m/z array"], intensities=row1["intensity array"])
    spec2 = Spectrum(mz=row2["m/z array"], intensities=row2["intensity array"])
    return spec1.plot_against(spec2, figsize=(8, 6), dpi=100)

st.set_page_config(
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)

st.markdown(config.INCLUDE_CSS, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("logo.png", width = 150)

with col3:
    st.write(' ')

menu = st.sidebar.radio("Select an option:", ("Display Compounds", "Data Visualization", "Data Metrics"))

if menu == "Display Compounds":
    df = pd.read_csv("app/labels_processed.csv")
    st.write("## Data Frame")

    # Add a text input for searching compounds
    search_compound = st.text_input("Search for a compound by name:")
    
    if search_compound:
        # Filter the DataFrame based on the entered compound name
        filtered_df = df[df["Compound"].str.contains(search_compound, case=False)]
        st.write(filtered_df)
    else:
        edited_df = st.write(df)  # Display the entire DataFrame

if menu == "Data Visualization":
    # url = "https://drive.google.com/file/d/1Ab_DJWWvfBPkNx0ApWLUBnSsvBe5wz2X/view?usp=sharing"
    # file_id = url.split('/')[-2]
    # output_file = "data_ready.csv"
    # gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)

    df = pd.read_csv("app/data_ready.csv")
    print(df.size)
    # Pagination
    PAGE_SIZE = 200
    total_rows = df.shape[0]
    num_pages = int(total_rows / PAGE_SIZE)
    if total_rows % PAGE_SIZE > 0:
        num_pages += 1
    page_num = st.number_input("Enter page number", min_value=1, max_value=num_pages, value=1)

    # Add a text input for searching compounds
    search_compound = st.text_input("Search for a row by source name:")
    
    if search_compound:
        start_idx = (page_num - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)

        # Display current page data
        st.write(f"Showing rows {start_idx + 1} to {end_idx} (out of {total_rows})")
        df = df.iloc[start_idx:end_idx]
        # Filter the DataFrame based on the entered compound name
        filtered_df = df[df["source"].str.contains(search_compound, case=False)]
        st.write(filtered_df)
    else:
        start_idx = (page_num - 1) * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_rows)

        st.write(f"Showing rows {start_idx + 1} to {end_idx} (out of {total_rows})")
        edited_df = st.write(df.iloc[start_idx:end_idx])
        # edited_df = st.write(df)  # Display the entire DataFrame
    # start_idx = (page_num - 1) * PAGE_SIZE
    # end_idx = min(start_idx + PAGE_SIZE, total_rows)

    # Display current page data
    # st.write(f"Showing rows {start_idx + 1} to {end_idx} (out of {total_rows})")
    # edited_df = st.write(df.iloc[start_idx:end_idx])
    # Create a section to display the details of a selected row
    # Create a section to display the details of a selected row
    
    
    selected_row_idx = st.number_input("Select a row index to view details:", min_value=0, max_value=len(df) - 1, value=0)
    row2_idx = st.number_input("Select the second row index for spectrum comparison:", min_value=0, max_value=len(df) - 1, value=1)

    if selected_row_idx:
        selected_row = df.iloc[selected_row_idx - 1]
        st.write("### Details")
        # Create a layout with three columns
        col1, col2, col3 = st.columns([1, 2, 1])
        column1 ,column2,column3 = st.columns([1, 2, 1])
        with col1 : 
            if st.button("Show Spectrum Graph"):
                selected_row = df.iloc[selected_row_idx]
                with column2 : 
                    plot_spectrum(selected_row)  # Assuming you have a plot_spectrum function
            


        
        # No specific results column, display other columns if needed
        if row2_idx : 
            with col2 : 
                if st.button("Visualize Spectrum Comparison"):
                    row1 = df.iloc[selected_row_idx]
                    row2 = df.iloc[row2_idx]

                    mz_array = row1["m/z array"]
                    intensity_array = row1["intensity array"]

                    value_list = row1["m/z array"].strip('[]').split()

                    # Convert the list of values to floats and then create a NumPy array
                    mz_array_1 = np.array([float(value) for value in value_list])

                    # Remove '[' and ']' characters and split the string into a list of values
                    value_list = row1["intensity array"].strip('[]').split()

                    # Convert the list of values to floats and then create a NumPy array
                    numpy_array_1 = np.array([float(value) for value in value_list])


                    mz_array_2 = row2["m/z array"]
                    intensity_array = row2["intensity array"]

                    value_list = row2["m/z array"].strip('[]').split()

                    # Convert the list of values to floats and then create a NumPy array
                    mz_array_2 = np.array([float(value) for value in value_list])

                    # Remove '[' and ']' characters and split the string into a list of values
                    value_list = row2["intensity array"].strip('[]').split()

                    # Convert the list of values to floats and then create a NumPy array
                    numpy_array_2 = np.array([float(value) for value in value_list])


                    # Create Spectrum instances
                    spec1 = Spectrum(mz=mz_array_1, intensities=numpy_array_1)
                    spec2 = Spectrum(mz=mz_array_2, intensities=numpy_array_2)
                    with column2 : 

                        # Visualize spectrum comparison
                        fig, _ = spec1.plot_against(spec2, figsize=(4, 2), dpi=100)
                        st.pyplot(fig)






        # Get the column names that start with "y"
        y_columns = [col_name for col_name in selected_row.index if col_name.startswith("y")]
        y_columns.sort()

        # Exclude columns "Diff_exact_mass" and "Mass_error_ppm"
        other_columns = [col_name for col_name in selected_row.index if col_name not in y_columns and col_name not in ["diff_exact_mass", "mass_error_ppm"]]

        # Calculate the maximum length of y_column_name
        max_y_column_length = max(len(y_column_name.capitalize()) for y_column_name in y_columns)

        # Calculate the padding for centering
        padding = max_y_column_length // 2

        # Create a layout with three columns
        col1, col2, col3 = st.columns([1, 2, 1])

        # Display "Y columns" centered in the center column
        with col2:
            st.write(" " * padding, "## Y Columns")
            for y_column_name in y_columns:
                st.write(" " * padding, f"**{y_column_name.capitalize()}**: {selected_row[y_column_name]}")

        # Display other values in the first column
        st.write("### Other Columns")
        for column_name in other_columns:
            column_value = selected_row[column_name]
            st.write(f"{column_name.capitalize()}: {column_value}")

        # Leave the third column empty
        with col3:
            st.write("")

if menu == "Data Metrics":

    df = pd.read_csv("app/data_ready.csv")
    df_fragment = pd.read_csv("app/fragments.csv")
    fragment_list = df_fragment['0'].tolist()

    selected_fragment = st.selectbox("Select a fragment number", options=fragment_list)
    selected_fragment_index = st.number_input("Select a row index:", min_value=0, max_value=53, value=0)
    if selected_fragment_index   : 
        selected_fragment = df_fragment['0'][selected_fragment_index]
        plot_kmd(df, size=df.shape[0], fragment=str(selected_fragment))
        plot_kmd2(df, size=df.shape[0], fragment=str(selected_fragment) ) 
    if selected_fragment : 
        
        plot_kmd(df, size=df.shape[0], fragment=str(selected_fragment))
        plot_kmd2(df, size=df.shape[0], fragment=str(selected_fragment) ) 


