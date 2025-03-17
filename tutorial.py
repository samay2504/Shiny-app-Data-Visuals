from shiny import App, render, ui
import pandas as np
import numpy as np
import matplotlib.pyplot as plt

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

app_ui = ui.page_fluid(
    ui.h1("My First Shiny App"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider("n", "Number of observations", min=10, max=500, value=100),
            ui.input_slider("bins", "Number of bins", min=5, max=50, value=20),
            ui.input_select("plot_type", "Plot type", 
                          choices=["Histogram", "Scatter Plot", "Line Plot"], 
                          selected="Histogram")
        ),
        ui.output_plot("plot")
    )
)

def server(input, output, session):
    @output
    @render.plot
    def plot():
        np.random.seed(19680801)
        x = np.random.normal(0, 1, input.n())

        fig, ax = plt.subplots()

        if input.plot_type() == "Histogram":
            ax.hist(x, bins=input.bins())
            ax.set_title(f"Histogram with {input.bins()} bins")
        elif input.plot_type() == "Scatter Plot":
            y = np.random.normal(0, 1, input.n())
            ax.scatter(x, y)
            ax.set_title(f"Scatter Plot with {input.n()} points")
        else:  # Line Plot
            ax.plot(range(input.n()), x)
            ax.set_title(f"Line Plot with {input.n()} points")
            
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        
        return fig

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
