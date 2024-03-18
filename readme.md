In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

In [2]: import os
In [3]: os.listdir()
In [4]: df1= pd.read_csv('horseasses-population-in-nepal-by-district-.csv')
        df2= pd.read_csv('milk-animals-and-milk-production-in-nepal-by-district.csv')
        df3= pd.read_csv('net-meat-production-in-nepal-by-district-.csv')
        df4= pd.read_csv('production-of-cotton-in-nepal-by-district-.csv')
        df5= pd.read_csv('production-of-egg-in-nepal-by-district.csv')
        df6= pd.read_csv('rabbit-population-in-nepal-by-district-.csv')
        df7= pd.read_csv('wool-production-in-nepal-by-district-.csv')
        df8= pd.read_csv('yak-nak-chauri-population-in-nepal-by-district-.csv')

In [5]: print("df1: ")
        (df1.head(5))
        print("df2: ")
        display(df2.head(5))
        print("df3: ")
        display(df3.head(5))
        print("df4: ")
        display(df4.head(5))
        print("df5: ")
        display(df5.head(5))
        print("df6: ")
        display(df6.head(5))
        print("df7: ")
        display(df7.head(5))
        print("df8: ")
        display(df8.head(5))

In [6]: dfs = [df1, df2, df3, df4, df5, df6, df7, df8]
        merged_df = pd.merge(dfs[0], dfs[1], on='DISTRICT', how='outer')
        for df in dfs[2:]:
        merged_df = pd.merge(merged_df, df, on='DISTRICT', how='outer')
In [7]: merged_df
In [8]: merged_df.info()
In [9]: nan_counts = merged_df.isna().sum()
        nan_counts
In [10]:sns.heatmap(merged_df.isnull(), yticklabels = False)

In [11]:columns_to_delete = nan_counts[nan_counts > 54].index.tolist() #Data cleaning
        merged_df.drop(columns=columns_to_delete, inplace=True)
In [12]:merged_df.info()
In [13]:merged_df.isna().sum()
In [14]:merged_df['DISTRICT'] = merged_df['DISTRICT'].str.lower()
In [15]:merged_df
In [16]:merged_df = merged_df.drop_duplicates()
        columns_to_visualize = merged_df.columns[1:]
In [17]: num_rows = 7  # You can adjust this based on the number of columns you have
        num_cols = 3
        fig = make_subplots(rows=num_rows, cols=num_cols)

        for i, column in enumerate(columns_to_visualize):
        fig.add_trace(go.Box(y=merged_df[column], name=column, boxpoints='all'),
                  row=i // num_cols + 1, col=i % num_cols + 1)
        fig.update_layout(height=3000, width=1000, title_text="Box Plots")
        fig.show()
In [18]:numeric_columns =merged_df.select_dtypes(include=['int', 'float'])
        correlation_matrix = numeric_columns.corr()
        correlation_matrix
In [19]:heatmap_fig = px.imshow(
            correlation_matrix,
            x = correlation_matrix.columns,
            y = correlation_matrix.columns,
            )
        heatmap_fig.update_layout(title = "Correlation Matrix")
        heatmap_fig.show()
In [20]:print(merged_df.columns.tolist())
In [21]:merged_df.fillna(merged_df.mean(), inplace=True)
In [22]:features = ['MILKING  BUFFALOES NO.','BUFF MILK','MILKING  COWS NO.','COW MILK']
        X = pd.get_dummies(merged_df[features])
        target_column = 'TOTAL MILK PRODUCED'
In [23]:merged_df[features]
In [24]:merged_df.fillna(0,inplace=True)
In [25]:merged_df
In [26]:
# Define function for applying linear regression
def apply_linear_regression(merged_df, features, target_column, test_size=0.2, random_state=42):
    # Step 1: Split the data into features (X) and target variable (y)
    X = merged_df[features]
    y = merged_df[target_column]
    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Reshape X_train and X_test to be 2D arrays
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    lr_model = LinearRegression()  # Step 3: Choose a model (Linear Regression)
    lr_model.fit(X_train, y_train)  # Step 4: Train the model
    y_pred_lr = lr_model.predict(X_test) # Step 5: Make predictions


    # Calculate Mean Absolute Error
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    # Calculate Mean Squared Error
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    # Calculate Root Mean Squared Error
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

    # Calculate R-squared (R2) score
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f'Mean Absolute Error: {mae_lr:.2f}')
    print(f'Mean Squared Error: {mse_lr:.2f}')
    print(f'Root Mean Squared Error: {rmse_lr:.2f}')
    print(f'R-squared (R2) Score: {r2_lr:.2f}')

    return lr_model, y_test, y_pred_lr  # Return the trained model, true values, and predicted values for evaluation

# Apply linear regression
trained_lr_model, true_values, predicted_values = apply_linear_regression(merged_df, ['SHEEPS NO.'], 'SHEEP WOOL PRODUCED')

In [27]:province_mapping = {
    'Taplejung': '1',
    'Panchthar': '1',
    'Ilam': '1',
    'Jhapa': '1',
    'Morang': '1',
    'Sunsari': '1',
    'Dhankuta': '1',
    'Terhathum': '1',
    'Sankhuwasabha': '1',
    'Bhojpur': '1',
    'Solukhumbu': '1',
    'Khotang': '1',
    'Okhaldhunga': '1',
    'Udayapur': '1',
    'Bhaktapur': '3',
    'Dhading': '3',
    'Kathmandu': '3',
    'Kavrepalanchok': '3',
    'Lalitpur': '3',
    'Nuwakot': '3',
    'Rasuwa': '3',
    'Sindhupalchok': '3',
    'Chitwan': '3',
    'Dolakha': '3',
    'Makwanpur': '3',
    'Ramechhap': '3',
    'Sindhuli': '3',
    'Gorkha': '4',
    'Lamjung': '4',
    'Tanahun': '4',
    'Syangja': '4',
    'Kaski': '4',
    'Manang': '4',
    'Mustang': '4',
    'Parbat': '4',
    'Myagdi': '4',
    'Baglung': '4',
    'Nawalparasi East': '4',
    'Rupandehi': '5',
'Kapilvastu': '5',
    'Arghakhanchi': '5',
    'Gulmi': '5',
    'Palpa': '5',
    'Dang': '5',
    'Pyuthan': '5',
    'Rolpa': '5',
    'Banke': '5',
    'Bardiya': '5',
    'Rukum': '6',
    'Salyan': '6',
    'Dolpa': '6',
    'Jumla': '6',
    'Kalikot': '6',
    'Mugu': '6',
    'Humla': '6',
    'Jajarkot': '6',
    'Dailekh': '6',
    'Surkhet': '6',
    'Kailali': '7',
    'Achham': '7',
    'Doti': '7',
    'Bajhang': '7',
    'Bajura': '7',
    'Kanchanpur': '7',
    'Dadeldhura': '7',
    'Baitadi': '7',
    'Darchula': '7',
    'Saptari': '2',
    'Siraha': '2',
    'Dhanusha': '2',
    'Mahottari': '2',
    'Sarlahi': '2',
    'Rautahat': '2',
    'Bara': '2',
    'Parsa': '2',
}
merged_df['DISTRICT'] = merged_df['DISTRICT'].str.title()
merged_df['PROVINCE'] = merged_df['DISTRICT'].apply(lambda x: province_mapping.get(x, "Not a District"))
In [28]:col_order = ['DISTRICT', 'PROVINCE'] + [col for col in merged_df.columns if col not in ['DISTRICT', 'PROVINCE']]
        merged_df = merged_df[col_order]
        # Check the updated dataframe structure
        merged_df.head()
In [29]:merged_df.to_csv('cleaned_dataset.csv', index=False)

