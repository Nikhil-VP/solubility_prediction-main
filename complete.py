import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load data from Excel sheet
excel_file = 'actualvspredictcomplete.csv'
df = pd.read_csv(excel_file)

# Load the trained model
model = joblib.load("D:/solubility_prediction-main/solubility_prediction-main/trained_model (1).pkl")


def about_page():
    st.title("About This App")

    st.markdown(
        "This web app allows you to predict the A log P value for a molecule based on its features. "
        "It also provides visualizations such as scatter plots and histograms of the predicted vs actual A log P values."
        )
    st.markdown("\nPositive AlogP: Indicates a compound is more hydrophobic and has a greater affinity for the organic phase.")
    st.markdown( "\nNegative AlogP: Suggests a compound is more hydrophilic and prefers the aqueous phase.")

    st.header("How to Use")
    st.markdown(
        "1. **A log P Prediction Page:** Enter molecular features and click 'Predict A log P' to get the predicted A log P value.\n"
        "2. **Linear Regression Graph Page:** Visualize the linear regression graph with predicted vs actual A log P values.\n"
        "3. **Scatter Plot and Histograms Page:** Explore scatter plots and histograms of the predicted vs actual A log P values.\n"
        "4. **Plot Selected Data Page:** Plot selected data points from dataset.\n"
        "5. **Read Data Page:** Read and search data from dataset."
    )

    st.header("Built with")
    st.markdown(
        "- [Streamlit](https://streamlit.io/) - For creating interactive web apps with Python.\n"
        "- [Pandas](https://pandas.pydata.org/) - For data manipulation and analysis.\n"
        "- [Plotly Express](https://plotly.com/python/plotly-express/) - For creating interactive plots.\n"
        "- [Scikit-learn](https://scikit-learn.org/) - For machine learning models."
    )
    st.header("Project By")
    st.markdown("Govardhan Dharmendra Hegde (RVCE23BCY018)")
    st.markdown("Nikhil V P (RVCE23BCY037)")
    st.markdown("Nuthan B (RVCE23BCY038)")
    st.markdown("Mohammed Abdul Razzaq (RVCE23BCY049)")


def predict_A_log_P(molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms):
    try:
        feature_vector = np.array(
            [[molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings, heavy_atoms]])
        predicted_A_log_P = model.predict(feature_vector)
        return predicted_A_log_P[0][0]
    except ValueError:
        return None


def plot_linear_regression_graph(subset_size, new_point_x=None):
    df_subset = df.head(subset_size)
    predicted_log_p = df_subset['Predicted AlogP'].values.reshape(-1, 1)
    actual_log_p = df_subset['Actual AlogP'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(predicted_log_p, actual_log_p)
    predictions = model.predict(predicted_log_p)

    fig, ax = plt.subplots(figsize=(8, 6))

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    equation = f'Actual log P = {slope:.2f} * Predicted log P + {intercept:.2f}'
    r_squared = r2_score(actual_log_p, predictions)
    equation_text = f'Equation: {equation}\nR-squared: {r_squared:.2f}'
    ax.text(0.5, -0.2, equation_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.scatter(predicted_log_p, actual_log_p, label='Actual vs Predicted', s=5)
    ax.plot(predicted_log_p, predictions, color='red', label='Linear Regression Line')

    if new_point_x is not None:
        new_point_y = model.predict(np.array([[new_point_x]]))
        ax.scatter(new_point_x, new_point_y, color='black', label='AlogP')
        a = f'Actual AlogP:{new_point_y}'
        ax.text(0.5, -0.25, a, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlabel('Predicted log P')
    ax.set_ylabel('Actual log P')
    ax.set_title(f'Linear Regression: Actual vs Predicted log P ({subset_size} Values)')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# New Function to Read Data from a Different Excel Files
def read_data_from_excel(file_path):
    try:
        df_new = pd.read_excel(file_path)
        return df_new
    except Exception as e:
        st.error(f"Error reading data from Excel file: {e}")
        return None


def read_data_page():
    st.title("Read Data ")
    st.markdown("")

    file_path = "All Drugs.csv"  # Specify the file path

    if st.button("Read Data"):
        df_new = pd.read_csv(file_path, delimiter="\t")
        st.success("Data read successfully!")

    if len(df_new) > 10:
        st.write("Displaying the most recent 10 entries:")
        st.dataframe(df_new.tail(10))
    else:
        st.write("Displaying all entries:")
        st.dataframe(df_new)



            # Search Bar for A log P Value
            #search_value = st.text_input("Enter A log P value to search:")
        #     if search_value:
        #         try:
        #             search_value = float(search_value)
        #             print(df_new['AlogP'])
        #             filtered_df = df_new.loc[df_new['AlogP'] == search_value]
        #             st.write(f"Debug: Filtered DataFrame: {filtered_df}")
        #             if not filtered_df.empty:
        #                 st.success("Search results:")
        #                 st.dataframe(filtered_df)
        #             else:
        #                 st.warning("No matching records found.")
        #         except ValueError:
        #             st.warning("Please enter a valid numeric A log P value.")
        #     else:
        #         st.info("Enter an A log P value to search.")
        # except Exception as e:
        #     st.error(f"Error reading data from Excel file: {e}")


def scatter_plot_and_histograms():
    st.title("Scatter Plot and Histograms")

    scatter_fig = px.scatter(x=df['Predicted AlogP'], y=df['Actual AlogP'], title='Actual vs Predicted A log P',
                             labels={'x': 'Predicted A log P', 'y': 'Actual A log P'}, template='plotly_dark')
    scatter_fig.update_layout(showlegend=True)
    st.plotly_chart(scatter_fig)

    hist_actual_fig = px.histogram(x=df['Actual AlogP'], nbins=20, color_discrete_sequence=['blue'],
                                   labels={'x': 'Actual A log P'}, template='plotly_dark')
    hist_actual_fig.update_layout(barmode='overlay', title='Distribution of Actual A log P')
    st.plotly_chart(hist_actual_fig)

    hist_pred_fig = px.histogram(x=df['Predicted AlogP'], nbins=20, color_discrete_sequence=['green'],
                                 labels={'x': 'Predicted A log P'}, template='plotly_dark')
    hist_pred_fig.update_layout(barmode='overlay', title='Distribution of Predicted A log P')
    st.plotly_chart(hist_pred_fig)


def plot_selected_data_page():
    st.title("Plot Selected Data Page")
    st.markdown("This page allows you to plot selected data points from an Excel file.")

    # Read the Excel file into a pandas DataFrame
    file_path = 'solubility_data.csv'
    dataset = pd.read_csv(file_path)

    # Ask the user for the number of data points to be taken
    num_data_points = st.number_input("Enter the number of data points to plot:", min_value=1, max_value=len(dataset),
                                      value=len(dataset))

    # Limit the dataset to the specified number of data points
    dataset = dataset.head(num_data_points)

    # Display column names for the user to choose the x-axis column
    st.write("Available columns:")
    for i, col in enumerate(dataset.columns):
        st.write(f"{i + 1}. {col}")

    # Ask the user to choose the column for the x-axis
    x_axis_column_index = st.number_input("Enter the number corresponding to the x-axis column:", min_value=1,
                                          max_value=len(dataset.columns), value=1) - 1
    X = dataset.iloc[:, x_axis_column_index]

    # Display column names for the user to choose the y-axis column
    st.write("Available columns:")
    for i, col in enumerate(dataset.columns):
        st.write(f"{i + 1}. {col}")

    # Ask the user to choose the column for the y-axis
    y_axis_column_index = st.number_input("Enter the number corresponding to the y-axis column:", min_value=1,
                                          max_value=len(dataset.columns), value=2) - 1
    Y = dataset.iloc[:, y_axis_column_index]

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(X, Y, '.')
    ax.set_title(f'{dataset.columns[y_axis_column_index]} vs {dataset.columns[x_axis_column_index]}')
    ax.set_ylabel(dataset.columns[y_axis_column_index])
    ax.set_xlabel(dataset.columns[x_axis_column_index])
    ax.grid(True)

    # Show the plot using st.pyplot
    st.pyplot(fig)


def main():
    palp = 0
    st.title("A log P Prediction")

    st.markdown(
        "This web app predicts the A log P value for a molecule based on its features. Enter the details and click 'Predict A log P.'")

    # Section for A log P prediction
    st.header("")

    col1, col2 = st.columns(2)

    with col1:
        molecular_weight = st.number_input("Molecular Weight", min_value=0.0)
        polar_surface_area = st.number_input("Polar Surface Area", min_value=0.0)
        hbd = st.number_input("Number of H-Bond Donors", min_value=0, step=1)
        hba = st.number_input("Number of H-Bond Acceptors", min_value=0, step=1)

    with col2:
        rotatable_bonds = st.number_input("Number of Rotatable Bonds", min_value=0, step=1)
        aromatic_rings = st.number_input("Number of Aromatic Rings", min_value=0, step=1)
        heavy_atoms = st.number_input("Number of Heavy Atoms", min_value=0.0)

    if st.button("Predict A log P"):
        result = predict_A_log_P(molecular_weight, polar_surface_area, hbd, hba, rotatable_bonds, aromatic_rings,
                                 heavy_atoms)

        if result is not None:
            st.success(f"Predicted A log P: {result:.2f}")
            palp = result
            if palp<0:
                st.success("The molecule may be Hydrophilic in nature")
            elif palp>0:
                st.success("The molecule may be Lipophilic in nature")
            else:
                st.success("The molecule may be equally partitioned in organic & aqueous phase")
        else:
            st.error("Invalid input. Please enter valid numeric values for all features.")

    st.markdown("---")  # Add a horizontal line to separate sections
    # Section for linear regression graph
    st.header("Linear Regression Graph")
    subset_size = st.slider("Select the number of values for the linear regression graph", min_value=10,
                            max_value=len(df), value=100)

    plot_linear_regression_graph(subset_size, new_point_x=palp)


# 'Read Data from Excel': read_data_page,
# Main Execution
if __name__ == "__main__":
    pages = {'About': about_page, 'A log P Prediction': main,
             'Scatter Plot and Histograms': scatter_plot_and_histograms,
             'Plot Selected Data': plot_selected_data_page, 'Dataset': read_data_page}
    selection = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[selection]()
