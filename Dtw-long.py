if __name__ == '__main__':
    from bokeh.layouts import row
    from bokeh.plotting import figure, output_file, show
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import itertools
    from bokeh.transform import linear_cmap
    from bokeh.util.hex import hexbin
    from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
    from bokeh.layouts import column


    # select a palette
    from bokeh.palettes import Dark2_5 as palette
    # itertools handles the cycling
    import itertools  
    from math import pi
    import os

    dataframe_dir = os.path.join(os.getcwd(), "DTW_output")
    dataframe_name = os.path.join( dataframe_dir, "export_dataframe_long_checkpoint_79992.pt_test_data.csv.csv")
    df = pd.read_csv(dataframe_name)
    df['Original'] = df.Original.astype(str)
    df['Predict'] = df.Predict.astype(str)
    df['Score_long'] = df.Score_long.astype(float)


    # df = data.set_index('Predict')
    # df.columns.name = 'Original'

    # predicts = list(data.index)
    # originals = list(data.columns)

    print(df)
    # this is the colormap from the original NYTimes plot
    # colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    colors = ["#FAD817", "#E3BE24", "#CDA431", "#B78A3E", "#A1714B", "#8B5758", "#753D65", "#5F2372", "#490A80"]

    mapper = LinearColorMapper(palette=colors, low=df.Score_long.min(), high=df.Score_long.max())


    mean_long=df.groupby(['Original', 'Predict'], as_index=False).mean()

    H = mean_long.to_numpy()
    # print(H.shape)
    # print(H)
    H = np.reshape(H, (10, 10, 3))

    predicts = list(H[0,:,1])
    originals = list(H[:,0,0])

    print(H[:,:,1])
    # print(H[:,:,0])

    # print(predicted_ponct)
    # print(original_ponct)

    harvest = H[:,:,2]
    harvest = np.reshape(harvest, (100))
    # print(harvest)


    combination = [ x+y for x in originals for y in predicts]
    print(len(combination))

    # factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
    factors = combination

    # x =  [50, 40, 65, 10, 25, 37, 80, 60]
    x = harvest
    # print(harvest)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    dot = figure(title="Categorical Dot Plot", tools=TOOLS,  y_range=factors, x_range=[0,3])

    dot.segment(0, factors, x, factors, line_width=6, line_color="green", )
    dot.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=5, )

    # x = ["foo 123", "foo 123", "foo 123", "bar:0.2", "bar:0.2", "bar:0.2", "baz-10",  "baz-10",  "baz-10"]
    # y = ["foo 123", "bar:0.2", "baz-10",  "foo 123", "bar:0.2", "baz-10",  "foo 123", "bar:0.2", "baz-10"]

    factors = originals

    x = list(np.reshape(H[:,:,0],(100)))
    y = list(np.reshape(H[:,:,1],(100)))

    var_gradient=["#FAD817","#F7D418","#F4D11A","#F1CD1C","#EECA1E","#EBC61F","#E8C321","#E5BF23","#E2BC25","#DFB827", "#DCB528", "#D9B12A", "#D6AE2C", "#D3AA2E", "#D0A72F", "#CDA331", "#CAA033","#C79C35", "#C49937", "#C19538", "#BE923A", "#BB8E3C", "#B88B3E", "#B5873F", "#B28441", "#AF8043","#AC7D45", "#A97947", "#A67648", "#A3724A", "#A06F4C", "#9D6B4E", "#9A684F", "#976451", "#946153", "#915D55", "#8E5A57", "#8B5658", "#88535A", "#854F5C", "#824C5E", "#7F485F", "#7C4561", "#794163", "#763E65", "#733A67", "#703768", "#6D336A", "#6A306C", "#672C6E", "#64296F", "#612571", "#5E2273", "#5B1E75", "#581B77", "#551778", "#52147A", "#4F107C", "#4B0D7E", "#490A80"]


    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p = figure(title="Categorical Heatmap",
                x_range=predicts, y_range=list(reversed(originals)),
                x_axis_location="above", plot_width=900, plot_height=400,
                tools=TOOLS, toolbar_location='below',
                tooltips=[('couple', '@Original @Predict'), ('Score_long', '@Score_long%')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "20pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    p.rect(x="Predict", y="Original", width=1, height=1,
        source=df,
        fill_color={'field': 'Score_long', 'transform': mapper},
        line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="3.5pt",
                        ticker=BasicTicker(desired_num_ticks=len(colors)),
                        formatter=PrintfTickFormatter(format="%d%%"),
                        label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # show(p)      # show the plot

    #output_file("categorical.html", title="categorical.py example")
    # show(column(p, dot))
    show(row(p, dot, sizing_mode="scale_width"))  # open a browser
