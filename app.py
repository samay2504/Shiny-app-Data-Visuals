from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

app_ui = ui.page_fluid(
    ui.h1("CSV Explorer - Exploratory Data Analysis"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file", "Upload CSV File", accept=[".csv"]),
            ui.hr(),
            ui.h4("Analysis Options"),
            ui.input_select("analysis_type", "Analysis Type", 
                            choices=["Data Overview", "Distribution Analysis", "Correlation Analysis"], 
                            selected="Data Overview"),
            ui.panel_conditional(
                "input.analysis_type === 'Distribution Analysis'",
                ui.input_select("dist_var", "Select Variable", choices=[]),
                ui.input_checkbox("show_kde", "Show Density Curve", value=True),
                ui.input_slider("bins", "Number of bins", min=5, max=50, value=20)
            ),
            ui.panel_conditional(
                "input.analysis_type === 'Correlation Analysis'",
                ui.input_select("corr_x", "X Variable", choices=[]),
                ui.input_select("corr_y", "Y Variable", choices=[]),
                ui.input_checkbox("show_trend", "Show Trend Line", value=True)
            )
        ),
        ui.navset_tab(
            ui.nav_panel("Data Preview", ui.output_table("preview_df")),
            ui.nav_panel("Summary Statistics", ui.output_table("summary_stats")),
            ui.nav_panel("Visualization", ui.output_plot("plot")),
            ui.nav_panel("Insights", ui.output_text("insights"))
        )
    )
)

def server(input, output, session):

    @reactive.Calc
    def data():
        req_file = input.file()
        if req_file is None:
            return pd.DataFrame({
                'x': np.random.normal(0, 1, 100),
                'y': np.random.normal(0, 1, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
        file_path = req_file[0]['datapath']
        df = pd.read_csv(file_path)
        return df

    @reactive.Effect
    def _():
        df = data()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        ui.update_select("dist_var", 
                         choices=numeric_cols,
                         selected=numeric_cols[0] if numeric_cols else None)
        
        ui.update_select("corr_x", 
                         choices=numeric_cols,
                         selected=numeric_cols[0] if len(numeric_cols) > 0 else None)
        
        ui.update_select("corr_y", 
                         choices=numeric_cols,
                         selected=numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if len(numeric_cols) > 0 else None))

    @output
    @render.table
    def preview_df():
        df = data()
        return df.head(10)

    @output
    @render.table
    def summary_stats():
        df = data()
        stats_df = df.describe().T
        stats_df['missing'] = df.isnull().sum()
        stats_df['missing_percent'] = df.isnull().sum() / len(df) * 100
        stats_df['unique'] = df.nunique()
        return stats_df

    @output
    @render.plot
    def plot():
        df = data()
        analysis = input.analysis_type()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if analysis == "Data Overview":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                cat_col = cat_cols[0]
                df[cat_col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Count of {cat_col}")
                ax.set_ylabel("Count")
            else:
                num_col = df.select_dtypes(include=np.number).columns[0]
                ax.hist(df[num_col].dropna(), bins=20)
                ax.set_title(f"Distribution of {num_col}")
                ax.set_xlabel(num_col)
                ax.set_ylabel("Frequency")
        
        elif analysis == "Distribution Analysis":
            var = input.dist_var()
            data_to_plot = df[var].dropna()
            ax.hist(data_to_plot, bins=input.bins(), alpha=0.7, density=True)
            if input.show_kde():
                from scipy import stats
                kde = stats.gaussian_kde(data_to_plot)
                x_range = np.linspace(data_to_plot.min(), data_to_plot.max(), 1000)
                ax.plot(x_range, kde(x_range), 'r-')
            ax.set_title(f"Distribution of {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Density" if input.show_kde() else "Frequency")
        
        elif analysis == "Correlation Analysis":
            x_var = input.corr_x()
            y_var = input.corr_y()
            ax.scatter(df[x_var], df[y_var], alpha=0.6)
            if input.show_trend():
                from scipy import stats
                valid_data = df[[x_var, y_var]].dropna()
                if len(valid_data) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data[x_var], valid_data[y_var])
                    x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
                    ax.plot(x_range, intercept + slope * x_range, 'r-')
                    ax.text(0.05, 0.95, f'r = {r_value:.2f}, p = {p_value:.4f}', transform=ax.transAxes, verticalalignment='top')
            ax.set_title(f"{y_var} vs {x_var}")
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
        
        return fig

    @output
    @render.text
    def insights():
        df = data()
        insights_text = "Data Insights:\n\n"
        insights_text += f"- Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        missing = df.isnull().sum()
        if missing.sum() > 0:
            insights_text += f"- There are {missing.sum()} missing values in the dataset.\n"
            cols_with_missing = missing[missing > 0]
            insights_text += f"- Columns with missing values: {', '.join(cols_with_missing.index)}\n"
        else:
            insights_text += "- No missing values found in the dataset.\n"
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            insights_text += f"\nNumeric Variables ({len(num_cols)}):\n"
            for col in num_cols[:3]:
                insights_text += f"- {col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}, Std={df[col].std():.2f}\n"
            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr()
                corr_pairs = []
                for i in range(len(num_cols)):
                    for j in range(i+1, len(num_cols)):
                        corr_pairs.append((num_cols[i], num_cols[j], corr_matrix.iloc[i, j]))
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    insights_text += "\nStrongest correlations:\n"
                    for x, y, r in corr_pairs[:3]:
                        insights_text += f"- {x} and {y}: r = {r:.2f}\n"
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            insights_text += f"\nCategorical Variables ({len(cat_cols)}):\n"
            for col in cat_cols[:3]:
                value_counts = df[col].value_counts()
                top_val = value_counts.index[0]
                insights_text += f"- {col}: {df[col].nunique()} unique values. Most common: '{top_val}' ({value_counts.iloc[0]} occurrences)\n"
        return insights_text

app = App(app_ui, server)

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    try:
        app.run(host="0.0.0.0", port=port)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            import asyncio
            loop = asyncio.get_event_loop()
            loop.create_task(app._start_server(host="0.0.0.0", port=port))
            print(f"App is running. Visit http://127.0.0.1:{port}/ in your browser.")
        else:
            raise e
