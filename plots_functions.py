import matplotlib.pyplot as plt
from misc import Kf_curve_plot, Kw_curve_plot, Gr_curve_plot, Es_curve_plot
from cmapCLC import vel_cmap

def bpt_diagram_scatter(x, y, ax=None):
  if ax is None:
    ax_flag = True
    fig, ax = plt.subplots()
  else:
    ax_flag = False
  ax.scatter(x, y, cmap=vel_cmap())
  ax.set_ylim(-2.5, 1.2)
  ax.set_xlim(-2.0, 0.5)
  Kf_curve_plot(ax=ax, x_min=-2.5, linestyle='dashed',
                c='k', linewidth=1)
  Kw_curve_plot(ax=ax, x_min=-2.5, linestyle='dashdot',
                c='k', linewidth=1)
  Gr_curve_plot(ax=ax, x_min=-2.5, linestyle='dotted',
                c='k', linewidth=1)
  Es_curve_plot(ax=ax, x_min=-2.5, linestyle='solid',
                c='k', linewidth=1)
  if ax_flag:
    return fig, ax 
  
