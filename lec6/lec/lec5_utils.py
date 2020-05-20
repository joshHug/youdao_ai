import sys
import numpy as np

try:
    import plotly
    import plotly.graph_objs as go
except ImportError:
    print(
        "Plotly Package not found. Please run: pip install plotly",
        file=sys.stderr)


def loss_plot_3d(theta_1_series, theta_2_series, loss_function, x, y):
    """Plot 3D Surface.
        
    The function takes the following as argument:
        theta_1: a list or array of theta_1 value
        theta_2: a list or array of theta_2 value
        loss_function
        x: the original x input
        y: the original y output
    """
    plotly.offline.init_notebook_mode(connected=True)

    # Create loss surface
    t1_s = np.linspace(np.min(theta_1_series) - 0.1, np.max(theta_1_series) + 0.1)
    t2_s = np.linspace(np.min(theta_2_series) - 0.1, np.max(theta_2_series) + 0.1)

    x_s, y_s = np.meshgrid(t1_s, t2_s)
    data = np.stack([x_s.flatten(), y_s.flatten()]).T
    ls = []
    for t1, t2 in data:
        l = loss_function(t1, t2, x, y)
        ls.append(l)
    z = np.array(ls).reshape(50, 50)

    surface = go.Surface(x=t1_s, y=t2_s, z=z, colorscale='Viridis')

    # Our plot will be overlay of trace and surface
    data = [surface]

    layout = dict(
        width=800,
        height=700,
        autosize=True,
        title='Loss Surface',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(
                    x=-1.7428,
                    y=1.0707,
                    z=0.7100,
                )),
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'),
    )

    fig = dict(data=data, layout=layout)

    plotly.offline.iplot(fig)


def loss_contour_plot(theta_1_series, theta_2_series, loss_function,  x, y, flip_axes = False):
    """    
    The function takes the following as argument:
        theta_1: a list or array of theta_1 value
        theta_2: a list or array of theta_2 value
        loss_function
        x: the original x input
        y: the original y output
    """
    t1_s = np.linspace(np.min(theta_1_series) - 0.1, np.max(theta_1_series) + 0.1)
    t2_s = np.linspace(np.min(theta_2_series) - 0.1, np.max(theta_2_series) + 0.1)

    x_s, y_s = np.meshgrid(t1_s, t2_s)
    data = np.stack([x_s.flatten(), y_s.flatten()]).T
    ls = []
    for t1, t2 in data:
        l = loss_function(t1, t2, x, y)
        ls.append(l)
    z = np.array(ls).reshape(50, 50)

    if flip_axes:
        z = z.transpose()

    if not flip_axes:
        contour_x = t1_s
        contour_y = t2_s
    else:
        contour_x = t2_s
        contour_y = t1_s        

    lr_loss_contours = go.Contour(x=contour_x, 
                                  y=contour_y, 
                                  z=z, 
                                  colorscale='Viridis')
    
    contour_fig = go.Figure(data=[lr_loss_contours])
    contour_fig['layout']['yaxis']['autorange'] = "reversed"

    plotly.offline.iplot(contour_fig)