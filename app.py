import pandas as pd
import dash
import plotly.express as px
from dash import dcc, html, Input, Output
from data_guide import GatherData
from utility import *

# borrowed from https://stackoverflow.com/questions/63459424/how-to-add-multiple-graphs-to-dash-app-on-a-single-browser-page
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = GatherData("./data/nyc_housing_data_SAMPLE.csv")
data.load_data()
data.preprocess_data()
df = data.remove_price_outliers(10000)


fig_dist = px.histogram(df, "SALE PRICE")
df_bar = pd.DataFrame([{"Condo":df['PROPERTY TYPE_Condo'].sum(), 
                        "Coop": df['PROPERTY TYPE_Coop'].sum(),
                        "Land": df['PROPERTY TYPE_Land'].sum(),
                        "MultiFamily": df['PROPERTY TYPE_MultiFamily'].sum(),
                        "SingleFamily": df['PROPERTY TYPE_SingleFamily'].sum(),
                        "Rental": df['PROPERTY TYPE_Rental'].sum()}])
df_bar = df_bar.transpose().reset_index()
df_bar.columns = ['PropertyType', 'Count']
fig_bar_pt = px.bar(df_bar, x="PropertyType", y="Count")

app.layout = html.Div(children=[
    html.Div([
        dcc.Dropdown(options=df['BOROUGH'].unique(), 
                     id='borough_homes',
                     placeholder="Select by borough")
                    ]),
    html.Div([
        dcc.Dropdown(options=['Large Home', 'Small Home', 'All Homes'], 
                      value='Small Home', id='size_homes',
                      placeholder="Select a home size")
                    ]),
    html.Div([
        dcc.Dropdown(['New Home','Old Home', 'All Homes'], 
                      value='New Home', id="new_homes",
                      placeholder="Select a home age")
                    ]),
    # All elements from the top of the page
    html.Div([
        html.Div([
            html.H1(children='Square Footage to Sale Price'),
            html.Div(children='''Understanding how square footage can affect Sale Price'''),
            dcc.Graph(
                id='scatter-plot-sf-sp'
            ),  
        ], className='six columns'),
        html.Div([
            html.H1(children='Distribution of Sale Price'),
            html.Div(children=''''''),
            dcc.Graph(
                id='distribution-sp',
                figure=fig_dist
            ),  
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
    html.Div([
            html.Div([
                html.H1(children='Property Type'),
                html.Div(children='''Count of Property Type'''),
                dcc.Graph(
                    id='bar-property-type',
                    figure=fig_bar_pt
                ),  
            ], className='six columns'),
            html.Div([
                html.H1(children='Predicted Vs Actual Sales Price'),
                html.Div(children='''Relationship of Predicted and Actual Sales Price'''),
                dcc.Graph(
                    id='scatter-predicted',
                    figure=fig_bar_pt
                ),  
            ], className='six columns'),
        ], className='row'),
])

@app.callback(
    Output('scatter-plot-sf-sp', 'figure'),
    Input('borough_homes', 'value'),
    Input('size_homes', 'value'),
    Input("new_homes", 'value')
)
def update_scatter_plot_sf(bor_value, size_value, age_value):
    if not bor_value:
        bor_mask = df['BOROUGH'].unique()
    else:
        bor_mask = [bor_value]
    size_mask = home_size_mask(size_value)
    age_mask = home_age_mask(age_value)
    df_scatter = df
    df_scatter = df_scatter[df_scatter['BOROUGH'].isin(bor_mask)]
    df_scatter = df_scatter[df_scatter["LARGE HOMES_Yes"].isin(size_mask)]
    df_scatter = df_scatter[df_scatter["NEW HOMES_Yes"].isin(age_mask)]
    fig_scatter = px.scatter(df_scatter, x="GROSS SQUARE FEET", y="SALE PRICE")
    return fig_scatter

if __name__ == '__main__':
    app.run_server(debug=True)