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
        self.max_displacement = 0 # maximum recorded displacement for visualization
        self.max_derivative = 0 # maximum recorded derivative for visualization
        self.time_scaling_factor = time_scaling_factor
        self.figsizes = {"beam": (8,10),
                         "energies": (8, 6)}
    
    
    def visualize(self, approx, eval_points, results, results_reference=None, blocking=True):
        
        is_dynamic = len(results["u"]) > 1 # determine, whether there is more than one set of coefficients in the results
        
        # ----------displacements, derivatives and centerline----------
        spec = matplotlib.gridspec.GridSpec(nrows=4, ncols=1,
                                            width_ratios=[1], wspace=0.5,
                                            hspace=0.35, height_ratios=[1, 1, 1, 0.25],
                                            top=0.97, bottom=0)
        fig = plt.figure(self.fig_index, figsize=self.figsizes["beam"])
        fig.clear()
        
        ax_displacements = fig.add_subplot(spec[0])
        ax_derivatives = fig.add_subplot(spec[1])
        ax_centerline = fig.add_subplot(spec[2])
        
        t_min = results["t"].min()
        t_max = results["t"].max()
        if is_dynamic:
            ax_slider = fig.add_subplot(spec[-1])
            slider = Slider(ax_slider, 'time in s', t_min, t_max, valinit=t_max)
        
       # function for interpolating solutions between time steps
        interp_functions = {}
        line_labels = ["u", "w"]
        if is_dynamic:
            for label in line_labels:
                interp_functions[label] = scipy.interpolate.interp1d(results["t"], results[label], kind="linear", axis=0)
                if results_reference is not None:
                    interp_functions[label + "_ref"] = scipy.interpolate.interp1d(results["t"], results_reference[label], kind="linear", axis=0)
        else:
            for label in line_labels:
                interp_functions[label] = lambda t, coeffs=results[label][0]: coeffs # second argument is necessary to avoid referencing a variable from the outer scope in the lambda function
        #print(interp_functions["u"])
        #print(interp_functions["w"])
        #time.sleep(100)
                
        # define function to update the plots (callback for the slider as well as for manually updating the displayed time)
        def update(time):
            
            line_data = {}
            for partial_label in ["u", "w"]:
                for full_label in ([partial_label] if results_reference is None else [partial_label, partial_label + "_ref"]):
                    coeffs = interp_functions[full_label](time)
                    line_data[full_label] = approx.eval_solution(coeffs, eval_points)
                    der_label = full_label.replace(partial_label, partial_label + "_x")
                    line_data[der_label] = np.gradient(line_data[full_label], eval_points, axis=0)
                    self.max_displacement = max(np.max(np.abs(line_data[full_label])), self.max_displacement)
                    self.max_derivative = max(np.max(np.abs(line_data[der_label])), self.max_derivative)
            
            # plot displacements
            ax_displacements.clear()
            displacement_labels = {"u": ("u", "b"), "w": ("w", "r")}
            for label in displacement_labels:
                ax_displacements.plot(eval_points, line_data[displacement_labels[label][0]], displacement_labels[label][1])
                if results_reference is not None:
                    ax_displacements.plot(eval_points, line_data[displacement_labels[label][0] + "_ref"], "--" + displacement_labels[label][1])
            ax_displacements.set_ylim([-1.1*self.max_displacement, 1.1*self.max_displacement])
            ax_displacements.set_xlabel("x in m")
            ax_displacements.set_ylabel("u, w in m")
            ax_displacements.set_title("displacements")
            ax_displacements.grid(True)
            if results_reference is None:
                ax_displacements.legend([label for label in displacement_labels])
            else:
                ax_displacements.legend(sum([[label, label + " (static)"] for label in displacement_labels], []))
            
            # plot derivatives
            ax_derivatives.clear()
            displacement_labels = {"u_x": ("u_x", "b"), "w_x": ("w_x", "r")}
            for label in displacement_labels:
                ax_derivatives.plot(eval_points, line_data[displacement_labels[label][0]], displacement_labels[label][1])
                if results_reference is not None:
                    ax_derivatives.plot(eval_points, line_data[displacement_labels[label][0] + "_ref"], "--" + displacement_labels[label][1])
            ax_derivatives.set_ylim([-1.1*self.max_derivative, 1.1*self.max_derivative])
            ax_derivatives.set_xlabel("x in m")
            ax_derivatives.set_ylabel("u_x, w_x in m/m")
            ax_derivatives.set_title("derivatives")
            ax_derivatives.grid(True)
            if results_reference is None:
                ax_derivatives.legend([label for label in displacement_labels])
            else:
                ax_derivatives.legend(sum([[label, label + " (static)"] for label in displacement_labels], []))
            
            # plot centerline of the beam
            ax_centerline.clear()
            centerline_labels = {"z": ("centerline", "b")}
            ax_centerline.plot(eval_points + line_data["u"], line_data["w"], centerline_labels["z"][1])
            if results_reference is not None:
                ax_centerline.plot(eval_points + line_data["u_ref"], line_data["w_ref"], "--" + centerline_labels["z"][1])
                ax_centerline.legend(sum([[centerline_labels[label][0], centerline_labels[label][0] + " (static)"] for label in centerline_labels], []))
            else:
                ax_centerline.legend([label for label in centerline_labels])
            ax_centerline.set_ylim([-1.1*self.max_displacement, 1.1*self.max_displacement])
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
        
        
        # ----------energies----------
        fig_energies = plt.figure(self.fig_index + 1, figsize=self.figsizes["energies"])
        ax_energies = fig_energies.add_subplot()
        energy_labels = {"W_ext": ("external work", "b"), "E_k": ("kinetic energy", "r"), "W_i": ("internal energy", "g")}
        if is_dynamic:
            for key in energy_labels:
                ax_energies.plot(results["t"], results[key], "-" + energy_labels[key][1])
                if results_reference is not None:
                    ax_energies.plot(results["t"], results_reference[key], "--" + energy_labels[key][1])
            if results_reference is None:
                ax_energies.legend([energy_labels[key][0] for key in energy_labels])
            else:
                ax_energies.legend(sum([[energy_labels[key][0], energy_labels[key][0] + " (static)"] for key in energy_labels], []))
        else:
            ax_energies.bar([label for label in energy_labels], [float(results["W_ext"]), float(results["E_k"]), float(results["W_i"])])
            ax_energies.set_axisbelow(True)
        ax_energies.set_ylabel("energy in J")
        ax_energies.grid(True)
        
        if blocking:
            plt.show()
        else:
            plt.pause(2)













