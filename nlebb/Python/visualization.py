import scipy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
matplotlib.rcParams["figure.raise_window"] = False # disable raising windows, when "show()" or "pause()" is called


class BeamVisualization():
    
    def __init__(self, time_scaling_factor=1):
        """animations run faster than real-time, if time_saling_factor > 1"""
        
        self.fig_index = 100
        self.max_deflection = 0 # maximum recorded deflection for visualization
        self.time_scaling_factor = time_scaling_factor
        self.figsize = (8,10)
    
    
    def visualize(self, approx, eval_points, blocking=True):
        
        results = approx.results
        
        if results is None:
            raise ValueError("solve the problem before visualizing the solution")
        
        is_dynamic = len(results["u"]) > 1 # determine, whether there is more than one set of coefficients in the results
        
        spec = matplotlib.gridspec.GridSpec(nrows=3, ncols=1,
                                            width_ratios=[1], wspace=0.5,
                                            hspace=0.25, height_ratios=[1, 1, 0.25],
                                            top=0.97, bottom=0)
        fig = plt.figure(self.fig_index, figsize=self.figsize)
        fig.clear()
        
        ax_values = fig.add_subplot(spec[0])
        ax_centerline = fig.add_subplot(spec[1])
        
        t_min = results["t"].min()
        t_max = results["t"].max()
        if is_dynamic:
            ax_slider = fig.add_subplot(spec[-1])
            slider = Slider(ax_slider, 'time in s', t_min, t_max, valinit=t_max)
        
        # function for interpolating solutions at time steps
        if is_dynamic:
            interp_func_u = scipy.interpolate.interp1d(results["t"], results["u"], kind="linear", axis=0)
            interp_func_w = scipy.interpolate.interp1d(results["t"], results["w"], kind="linear", axis=0)
        else:
            interp_func_u = lambda t: results["u"][0]
            interp_func_w = lambda t: results["w"][0]
        # define the function to update the plot (callback for the slider / for manually updating the displayed time)
        def update(time):
            coeffs_u = interp_func_u(time)
            coeffs_w = interp_func_w(time)
            u = approx.eval_solution(coeffs_u, eval_points)
            w = approx.eval_solution(coeffs_w, eval_points)
            self.max_deflection = max(np.max(np.abs(np.concatenate([u, w]))), self.max_deflection)
            
            # plot values of u and w
            ax_values.clear()
            ax_values.plot(eval_points, u)
            ax_values.plot(eval_points, w)
            ax_values.set_ylim([-1.1*self.max_deflection, 1.1*self.max_deflection])
            ax_values.set_xlabel("x in m")
            ax_values.set_ylabel("u, w in m")
            ax_values.set_title("values of u and w")
            ax_values.legend(["u", "w"])
            ax_values.grid(True)
            
            # plot centerline of the beam
            ax_centerline.clear()
            ax_centerline.plot(eval_points - u, w)
            ax_centerline.set_ylim([-1.1*self.max_deflection, 1.1*self.max_deflection])
            ax_centerline.set_xlabel("x in m")
            ax_centerline.set_ylabel("z in m")
            ax_centerline.set_title("centerline of the beam")
            ax_centerline.grid(True)
            
            fig.canvas.draw_idle()
        
        if is_dynamic:
            #update(0)
            #plt.pause(2e-1)
            t_start = time.time()
            while t_min + (time.time() - t_start)*self.time_scaling_factor < t_max:
                t_scaled = (time.time() - t_start)*self.time_scaling_factor
                update(t_scaled)
                slider.set_val(t_scaled)
                plt.pause(5e-2)
            
            slider.set_val(t_max)
            slider.on_changed(update) # attach the callback function to the slider
            
        update(t_max) # make sure that the last point in time is displayed at the end
        
        if blocking:
            plt.show()
        else:
            plt.pause(2)













