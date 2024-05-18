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
        
        self.time_scaling_factor = time_scaling_factor
        # sizes of the figures
        self.figsizes = {"beam": (8, 10),
                         "observation_points": (8, 10),
                         "energies": (8, 6)}
    
    
    def visualize(self, approx, results, results_reference=None, obs_points=None, eval_times=None, num_eval_points=100, blocking=True):
        """"visualizes the results of static and/or dynamic FEM simulations
            
            approx: object of type GalerkinApproximation
            num_eval_points: number of points along the beam, at which the solution is to be evaluated
            results: solution in the form of a dictionary with fields for each quantity, contains the coefficients for the basis according to approx
            results_reference (optional): static reference solutions, which are plottet in case of a dynamic simulation
            obs_points (optional): observation points along the beam, at which displacements and derivatives over time are to be plottet
            eval_times (optional): times for the time series at the observation points
            blocking (optional): whether the function returns or the figures stay open and interactive
        """
        
        is_dynamic = len(results["u"]) > 1 # determine whether there is more than one set of coefficients in the results
        ref_label = "static" # meaning of reference values
        
        # ----------preprocessing (compute evaluation points / get physical displacements from coefficients / interpolate between time steps)----------
        # compute evaluation points
        eval_points = np.linspace(0, approx.domain["l"], num_eval_points)
        # add observation points to eval_points
        if obs_points is not None:
            eval_points = np.sort(np.unique(np.concatenate([eval_points, obs_points])))
        # integrate reference into results
        if results_reference is not None:
            results.update({key + "_ref": results_reference[key] for key in results_reference})
        interp_functions = {}
        for result_key in results: # iterate over fields in results dictionary
            if results[result_key].squeeze().shape == results["t"].squeeze().shape:
                # handle scalar qantities (scalar values for each time step)
                interp_functions[result_key] = results[result_key]
                keys = [result_key]
            else:
                # handle distributed quantities (coefficients for each time step)
                der_key = result_key + "_x" if not "_ref" in result_key else result_key.replace("_ref", "_x_ref")
                keys = [result_key, der_key]
                interp_functions.update({key: [] for key in keys})
                for i in range(len(results[result_key])): # iterate over time steps
                    displacements = approx.eval_solution(results[result_key][i,:], eval_points, "f")
                    derivatives_x = approx.eval_solution(results[result_key][i,:], eval_points, "f_x")
                    #derivatives_x = np.gradient(displacements, eval_points, axis=0)
                    interp_functions[result_key].append(displacements)
                    interp_functions[der_key].append(derivatives_x)
                interp_functions.update({key: np.stack(interp_functions[key]) for key in keys})
            if is_dynamic:
                interp_functions.update({key: scipy.interpolate.interp1d(results["t"], interp_functions[key], kind="linear", axis=0) for key in keys})
            else:
                interp_functions.update({key: lambda t, coeffs=interp_functions[key][0]: coeffs for key in keys}) # second argument is necessary to avoid referencing a variable from the outer scope in the lambda function
        
        
        # ----------plot distributed quantities (displacements, derivatives with respect to x and centerline)----------

        fig, (ax_displacements_dyn, ax_derivatives_dyn, ax_centerline, ax_slider) = plt.subplots(4, 1, figsize=self.figsizes["beam"], width_ratios=[1], height_ratios=[1, 1, 1, 0.25],
                                                                                         gridspec_kw={'hspace': 0.35, "top": 0.97, "bottom": 0})
        
        t_min = results["t"].min()
        t_max = results["t"].max()
        if is_dynamic:
            slider = Slider(ax_slider, 'time in s', t_min, t_max, valinit=t_max)
        else:
            ax_slider.set_visible(False)
        
                
        # define function to update the plots (callback for the slider as well as for manually updating the displayed time)
        def update(t_eval):
            
            # evaluate interpolation functions at the given time (quite efficient)
            line_data = {"u": interp_functions["u"](t_eval),
                         "u_x": interp_functions["u_x"](t_eval),
                         "w": interp_functions["w"](t_eval),
                         "w_x": interp_functions["w_x"](t_eval)}
            if results_reference is not None:
                line_data.update({"u_ref": interp_functions["u_ref"](t_eval),
                                  "u_x_ref": interp_functions["u_x_ref"](t_eval),
                                  "w_ref": interp_functions["w_ref"](t_eval),
                                  "w_x_ref": interp_functions["w_x_ref"](t_eval)})
            
            
            # plot displacements
            ylim_old = ax_displacements_dyn.get_ylim() if not 1. in ax_displacements_dyn.get_ylim() else [-1e-10, 1e-10] # record axes limits before new data is plotted
            ax_displacements_dyn.clear()
            ax_displacements_dyn.plot(eval_points, line_data["u"], "-b")
            ax_displacements_dyn.plot(eval_points, line_data["w"], "-r")
            if results_reference is not None:
                ax_displacements_dyn.plot(eval_points, line_data["u_ref"], "--b")
                ax_displacements_dyn.plot(eval_points, line_data["w_ref"], "--r")
                ax_displacements_dyn.legend(["u", "w", f"u ({ref_label})", f"w ({ref_label})"])
            else:
                ax_displacements_dyn.legend(["u", "w"])
            ylim_new = ax_displacements_dyn.get_ylim()
            ax_displacements_dyn.set_ylim([min(ylim_old[0], ylim_new[0]), max(ylim_old[1], ylim_new[1])])
            ax_displacements_dyn.set_xlabel("x in m")
            ax_displacements_dyn.set_ylabel("u, w in m")
            ax_displacements_dyn.set_title("displacements")
            ax_displacements_dyn.grid(True)
            
            
            # plot derivatives with respect to x
            ylim_old = ax_derivatives_dyn.get_ylim() if not 1. in ax_derivatives_dyn.get_ylim() else [-1e-10, 1e-10] # record axes limits before new data is plotted
            ax_derivatives_dyn.clear()
            ax_derivatives_dyn.plot(eval_points, line_data["u_x"], "-b")
            ax_derivatives_dyn.plot(eval_points, line_data["w_x"], "-r")
            if results_reference is not None:
                ax_derivatives_dyn.plot(eval_points, line_data["u_x_ref"], "--b")
                ax_derivatives_dyn.plot(eval_points, line_data["w_x_ref"], "--r")
                ax_derivatives_dyn.legend(["u_x", "w_x", f"u_x ({ref_label})", f"w_x ({ref_label})"])
            else:
                ax_derivatives_dyn.legend(["u_x", "w_x"])
            ylim_new = ax_derivatives_dyn.get_ylim()
            ax_derivatives_dyn.set_ylim([min(ylim_old[0], ylim_new[0]), max(ylim_old[1], ylim_new[1])])
            ax_derivatives_dyn.set_xlabel("x in m")
            ax_derivatives_dyn.set_ylabel("u_x, w_x in m/m")
            ax_derivatives_dyn.set_title("derivatives with respect to x")
            ax_derivatives_dyn.grid(True)
            
            
            # derivatives with respect to time could be added here
            
            
            # plot centerline of the beam
            ylim_old = ax_centerline.get_ylim() if not 1. in ax_centerline.get_ylim() else [-1e-10, 1e-10] # record axes limits before new data is plotted
            ax_centerline.clear()
            ax_centerline.plot(eval_points + line_data["u"], line_data["w"], "-b")
            if results_reference is not None:
                ax_centerline.plot(eval_points + line_data["u_ref"], line_data["w_ref"], "--b")
                ax_centerline.legend(["centerline", f"centerline ({ref_label})"])
            else:
                ax_centerline.legend(["centerline"])
            ylim_new = ax_centerline.get_ylim()
            ax_centerline.set_ylim([min(ylim_old[0], ylim_new[0]), max(ylim_old[1], ylim_new[1])])
            ax_centerline.set_xlabel("x in m")
            ax_centerline.set_ylabel("z in m")
            ax_centerline.set_title("centerline of the beam")
            ax_centerline.grid(True)
            
            fig.canvas.draw_idle()
        
        # animate motion of the beam
        if is_dynamic:
            t_start = time.time()
            while t_min + (time.time() - t_start)*self.time_scaling_factor < t_max:
                t_scaled = (time.time() - t_start)*self.time_scaling_factor
                update(t_scaled)
                slider.set_val(t_scaled)
                plt.pause(5e-2)
            
            slider.set_val(t_max)
            slider.on_changed(update) # attach the callback function to the slider
            
        update(t_max) # make sure that the last point in time is displayed at the end
        
        
        # ----------plot timeseries at observation points----------
        if is_dynamic and obs_points is not None:
            fig_obs, (ax_displacements_t, ax_derivatives_t) = plt.subplots(2, 1, figsize=self.figsizes["observation_points"], gridspec_kw={'hspace': 0.2, "top": 0.97, "bottom": 0.07})
            if eval_times is None:
                eval_times = np.linspace(results["t"].min(), results["t"].max(), len(eval_points))
            obs_indices = np.argwhere([point in obs_points for point in eval_points])
            
            # plot displacements
            for obs_index in obs_indices:
                l1 = ax_displacements_t.plot(eval_times, interp_functions["u"](eval_times)[:,obs_index])[0]
                l2 = ax_displacements_t.plot(eval_times, interp_functions["w"](eval_times)[:,obs_index])[0]
                if results_reference is not None:
                    ax_displacements_t.plot(eval_times, interp_functions["u_ref"](eval_times)[:,obs_index], linestyle="--", color=l1.get_color())
                    ax_displacements_t.plot(eval_times, interp_functions["w_ref"](eval_times)[:,obs_index], linestyle="--", color=l2.get_color())
            if results_reference is not None:
                ax_displacements_t.legend([f"{quantity}(t;x={obs_point:.2f}){suffix}" for obs_point in obs_points for suffix in ["", f" ({ref_label})"] for quantity in ["u", "w"]])
            else:
                ax_displacements_t.legend([f"{quantity}(t;x={obs_point:.2f})" for obs_point in obs_points for quantity in ["u", "w"]])
            ax_displacements_t.set_xlabel("t in s")
            ax_displacements_t.set_ylabel("u, w in m")
            ax_displacements_t.set_title("displacements")
            ax_displacements_t.grid(True)
            
            # plot derivatives with respect to time
            for obs_index in obs_indices:
                ax_derivatives_t.plot(eval_times, interp_functions["u_d"](eval_times)[:,obs_index])
                ax_derivatives_t.plot(eval_times, interp_functions["w_d"](eval_times)[:,obs_index])
            ax_derivatives_t.legend([f"{quantity}_d(t;x={obs_point})" for obs_point in obs_points for quantity in ["u", "w"]])
            ax_derivatives_t.set_xlabel("t in s")
            ax_derivatives_t.set_ylabel("u_d, w_d in m/s")
            ax_derivatives_t.set_title("derivatives with respect to time")
            ax_derivatives_t.grid(True)
            
            
            # derivatives with respect to x could be added here
            
        
        
        # ----------plot energies----------
        fig_energies, ax_energies = plt.subplots(figsize=self.figsizes["energies"])
        energy_labels = {"W_ext": ("external work", "b"), "E_k": ("kinetic energy", "r"), "W_i": ("internal energy", "g")}
        if is_dynamic:
            for key in energy_labels:
                ax_energies.plot(results["t"], results[key], "-" + energy_labels[key][1])
                if results_reference is not None:
                    ax_energies.plot(results["t"], results_reference[key], "--" + energy_labels[key][1])
            if results_reference is not None:
                ax_energies.legend([energy_labels[key][0]+ suffix for key in energy_labels for suffix in ["", f" ({ref_label})"]])
            else:
                ax_energies.legend([energy_labels[key][0] for key in energy_labels])
        else:
            ax_energies.bar([label for label in energy_labels], [float(results["W_ext"]), float(results["E_k"]), float(results["W_i"])])
            ax_energies.set_axisbelow(True)
        ax_energies.set_ylabel("energy in J")
        ax_energies.grid(True)
        
        if blocking:
            plt.show()
        else:
            update(t_max) # make sure that the last point in time is displayed at the end
            plt.pause(2)













