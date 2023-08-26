# import streamlit as st


# from src import config
# import pandas as pd
# import base64
# st.set_page_config(
#     page_title=config.APPLICATION_TITLE,
#     layout=config.LAYOUT,
#     initial_sidebar_state=config.SIDEBAR_STATE,
# )

# st.markdown(config.INCLUDE_CSS, unsafe_allow_html=True)

# st.sidebar.image(config.LOGO)
# st.markdown(config.APPLICATION_HEADER, unsafe_allow_html=True)



# st.title("Data Visualization and Modification")

# # Upload file
# compound_uploaded_file = st.file_uploader("Choose a file to upload your Compound data", type=["csv", "xlsx"])
# if compound_uploaded_file is not None:
#     c_df = pd.read_csv(compound_uploaded_file) 
#     st.write("## Compound Data")
#     st.write(c_df)

# if st.button('Display Compounds'):
#     df = pd.read_csv("labels_processed.csv")
#     st.write("## Data Frame")
#     edited_df = st.write(df) # 👈 An editable dataframe
# if st.button('Display CSV Data'):
#     df = pd.read_csv("inputs_semisup.csv")
#     st.write("## Data Frame")
#     edited_df = st.experimental_data_editor(df) # 👈 An editable dataframe

import streamlit as st
import pandas as pd
from src import config
import base64

st.set_page_config(
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)

st.markdown(config.INCLUDE_CSS, unsafe_allow_html=True)

# Centered logo
st.markdown(
    f"<div style='text-align:center;padding-top:10px;padding-bottom:10px;'><img src='app/images/logo.png'/></div>",
    unsafe_allow_html=True,
)

st.markdown(config.APPLICATION_HEADER, unsafe_allow_html=True)

# Sidebar menu
# st.sidebar.image(config.LOGO, use_column_width=True)
menu = st.sidebar.radio("Select an option:", ("Display Compounds", "Display CSV Data"))

# Upload file
compound_uploaded_file = st.file_uploader("Choose a file to upload your Compound data", type=["csv", "xlsx"])
if compound_uploaded_file is not None:
    c_df = pd.read_csv(compound_uploaded_file)
    st.write("## Compound Data")
    st.write(c_df)

if menu == "Display Compounds":
    df = pd.read_csv("labels_processed.csv")
    st.write("## Data Frame")
    edited_df = st.write(df)  # 👈 An editable dataframe

if menu == "Display CSV Data":
    df = pd.read_csv("inputs_semisup.csv")
    st.write("## Data Frame")
    edited_df = st.experimental_data_editor(df)  # 👈 An editable dataframe


    # Download modified file
    # csv = edited_df.to_csv(index=False)
    # b64 = base64.b64encode(csv.encode()).decode()
    # href = f'<a href="data:file/csv;base64,{b64}" download="modified_data.csv">Download modified data</a>'
    # st.markdown(href, unsafe_allow_html=True)

    
    # Choose a line and column to modify
    # line = st.number_input("Enter line number", min_value=1, max_value=len(df))
    # col = st.selectbox("Select a column to modify", df.columns)

    # # Modify the data frame
    # new_value = st.text_input("Enter new value", value=df.loc[line-1, col])
    # df.loc[line-1, col] = new_value
    # st.write("## Modified Data Frame")
    # st.write(df)

    # # Download modified file
    # csv = df.to_csv(index=False)
    # b64 = base64.b64encode(csv.encode()).decode()
    # href = f'<a href="data:file/csv;base64,{b64}" download="modified_data.csv">Download modified data</a>'
    # st.markdown(href, unsafe_allow_html=True)