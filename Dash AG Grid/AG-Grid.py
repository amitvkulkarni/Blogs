import dash
import pandas as pd
from dash import Dash, html, dcc, dash_table
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)


defaultColDef = {
    "filter": True,
    "floatingFilter": True,
    "resizable": True,
    "sortable": True,
    "editable": True,
    "minWidth": 150,
}

# columnDefs = [
#             {"field": "country", "rowDrag": True},
#             {"field": "pop"},
#             {"field": "continent"},
#             {"field": "lifeExp"},
#             {"field": "gdpPercap"}

# ]

getRowStyle1 = {
    "styleConditions": [
        {
            "condition": "params.data.lifeExp > 40 && params.data.lifeExp <= 60",
            "style": {"backgroundColor": "lavenderblush"},
        },
        # {"condition": "params.data.lifeExp > 60", "style": {"backgroundColor": "lightcoral"}},
    ]
}

columnDefs = [
    {
        "headerName": "Region",
        "children": [
            {
                "field": "country",
                "minWidth": 170,
                "resizable": True,
                "checkboxSelection": True,
                "headerCheckboxSelection": True,
            },
            {"field": "continent", "resizable": True},
        ],
    },
    {
        "headerName": "Quality of Life",
        "children": [
            {"field": "pop"},
            {"field": "lifeExp"},
            {"field": "gdpPercap", "type": "numeric", "specifier": ",.2f"},
        ],
    },
]

app = dash.Dash(__name__)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Dash DataTable",
                            style={
                                "text-align": "center",
                                "font-size": 35,
                                "font-weight": "bold",
                                "color": "#f8f9fa",
                            },
                        ),
                    ]
                )
            ],
            className="study-browser-banner row",
        ),
        dbc.Row(
            dash_table.DataTable(
                columns=[
                    {
                        "name": i,
                        "id": i,
                        "deletable": True,
                        "selectable": True,
                        "type": "numeric",
                        "format": {
                            "specifier": ",.2f",
                        },
                    }
                    for i in df.columns
                ],
                data=df.to_dict("records"),
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                    # 'border': '1px solid black'
                },
                style_data={"whiteSpace": "normal", "height": "auto"},
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    },
                    {
                        "if": {
                            "filter_query": "{lifeExp} > 40 && {lifeExp} < 75",
                            "column_id": "lifeExp",
                        },
                        "backgroundColor": "tomato",
                        "color": "white",
                    },
                ],
            ),
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Dash AG Grid",
                            style={
                                "text-align": "center",
                                "font-size": 35,
                                "font-weight": "bold",
                                "color": "#f8f9fa",
                            },
                        ),
                    ]
                )
            ],
            className="study-browser-banner row",
        ),
        dbc.Row(
            dag.AgGrid(
                columnDefs=columnDefs,
                rowData=df.to_dict("records"),
                columnSize="sizeToFit",
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                    "filter": True,
                    "editable": True,
                },
                dashGridOptions={
                    "rowDragManaged": True,
                    "undoRedoCellEditing": True,
                    "undoRedoCellEditingLimit": 20,
                    "editType": "fullRow",
                    "rowSelection": "single",
                    "pagination": True,
                    "paginationAutoPageSize": True,
                },
                getRowStyle=getRowStyle1,
            ),
        ),
        html.Br(),
        html.Br(),
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True)
