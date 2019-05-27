/*
 *  Cypress -- C++ Spiking Neural Network Simulation Framework
 *  Copyright (C) 2019 Christoph Ostrau
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @fíle brainscales.hpp
 *
 * Contains the native BrainScaleS backend for Cypress using the high level C++
 * APIs Euter, pymarocco, ...
 *
 * @author Christoph Ostrau
 */

#pragma once

#ifndef CYPRESS_BACKEND_BRAINSCALES_HPP
#define CYPRESS_BACKEND_BRAINSCALES_HPP

#include <stdexcept>
#include <string>

#include <cypress/core/backend.hpp>
#include <cypress/core/network.hpp>
#include <cypress/core/network_base.hpp>
#include <cypress/core/network_base_objects.hpp>
#include <cypress/util/json.hpp>
#include <cypress/util/logger.hpp>

#include <boost/make_shared.hpp>
#include "euter/celltypes.h"
#include "euter/objectstore.h"
#include "euter/population.h"
#include "euter/population_view.h"
#include "pymarocco/PyMarocco.h"
#include "sthal/Wafer.h"

namespace cypress {
/**
 * The BrainScaleS backend directly runs the emulation
 * using PyNN. Must be used on the BrainScaleS server either directly (logged in
 * via ssh) or via NMPI service
 */
class BrainScaleS : public Backend {
private:
	void do_run(NetworkBase &network, Real duration) const override;

public:
	/**
	 * Constructor of the BrainScaleS backend class, takes a PyNN-like setup
	 * JSON object.
	 *
	 * @param setup is a JSON object which supports the following settings:
	 * {
	 *    "neuron_size": 4, // #denmems for a single neuron
	 *    "use_big_capacitor": true, // chose between big and small capacitance
	 *                                                 on the hardware
	 *    "bandwidth": 0.8, // consider input firing rate and
	 *                                     maximally use fraction bandwidth of
	 *                                     0.8
	 *    "synapse_loss": false, // allow loss of synapses for mapping
	 *    "calib_path": "/wang/data/calibration/brainscales/default", // path
	 *                                   to calibration and defects
	 *    "wafer": 33, // which wafer to run on
	 *    "hicann": [297], // hicanns used for the experiment
	 *    "digital_weight" : true //directly set low-level digital weights
	 * }
	 */
	BrainScaleS(const Json &setup = Json());

	/**
	 * Destructor of the BrainScaleS backend class.
	 */
	~BrainScaleS() override = default;

	/**
	 * Returns the neuron types supported by the BrainScaleS backend.
	 */
	std::unordered_set<const NeuronType *> supported_neuron_types()
	    const override;

	/**
	 * Returns the canonical name of the backend.
	 */
	std::string name() const override { return "nmpm1"; }

private:
	// configuration stuff
	size_t m_neuron_size = 4;
	bool m_use_big_capacitor = true;
	double m_bandwidth = 0.8;
	bool m_synapse_loss = false;
	std::string m_calib_path = "/wang/data/calibration/brainscales/default";
	size_t m_wafer = 33;
	Json m_hicann = 297;
	bool m_digital_weight = false;
	bool m_warn_gsyn_emitted = false;

public:
	// Static functions for setting up the network

	static void init_logger();

	/**
	 * @brief For every population in poupulation, this function creates a
	 * bs_population and appends it to the vector.
	 *
	 * @param store global object store for BS simulation
	 * @param populations cypress population vector
	 * @return returns a vector with pointers to BS populations
	 */
	static std::vector<PopulationPtr> create_pops(
	    ObjectStore &store, const std::vector<PopulationBase> &populations);

	/**
	 * Sets parameters of src to bs parameters for the IF_cond_exp model.
	 *
	 * @param tar target BS parameter vector
	 * @oaram src source cypress parameters
	 */
	static inline void set_ifce_params(
	    CellTypeTraits<CellType::IF_cond_exp>::Parameters &tar,
	    const NeuronParameters &src)
	{
		tar.tau_refrac = src[4];
		tar.cm = src[0];
		tar.tau_syn_E = src[2];
		tar.v_rest = src[5];
		tar.tau_syn_I = src[3];
		tar.tau_m = src[1];
		tar.e_rev_E = src[8];
		tar.i_offset = src[10];
		tar.e_rev_I = src[9];
		tar.v_thresh = src[6];
		tar.v_reset = src[7];
	}

	/**
	 * Setts the same parameter set for all populations
	 *
	 * @param vec target BS population parameter vector
	 * @param src source cypress parameters
	 */
	static void set_hom_param(
	    TypedCellParameterVector<CellType::IF_cond_exp> &vec,
	    const NeuronParameters &src);

	/**
	 * Setts different parameters for every neuron
	 *
	 * @param vec target BS population parameter vector
	 * @param pop source cypress population
	 */
	static void set_inhom_param(
	    TypedCellParameterVector<CellType::IF_cond_exp> &vec,
	    const PopulationBase &pop);

	/**
	 * For every population, check its type and whether parameters are
	 * homogeneous, then apply parameters to the respective pop in bs_pop
	 *
	 * @param bs_pop target BS population, mirrored to
	 * @param pop source cypress population
	 */
	static void set_population_parameters(PopulationPtr &bs_pop,
	                                      const PopulationBase &pop);
    
	static bool warn_gsyn_emitted;

	/**
	 * Set recording flags, homogeneous for whole population
	 *
	 * @param params: target bs params of population
	 * @param pop: source cypress population
	 */
	template <typename T>
	static void set_hom_rec(T &params, const PopulationBase &pop)
	{
		std::vector<std::string> signals = pop.type().signal_names;
		for (size_t j = 0; j < signals.size(); j++) {
			if (pop.signals().is_recording(j)) {
				if (signals[j] == "spikes") {
					for (size_t neuron = 0; neuron < params.size(); neuron++) {
						params[neuron].record_spikes = true;
					}
				}
				else if (signals[j] == "v") {
					for (size_t neuron = 0; neuron < params.size(); neuron++) {
						params[neuron].record_v = true;
					}
				}
				else if ((signals[j] == "gsyn_exc") ||
				         (signals[j] == "gsyn_inh")) {
					for (size_t neuron = 0; neuron < params.size(); neuron++) {
						params[neuron].record_gsyn = true;
						if (!BrainScaleS::warn_gsyn_emitted) {
							global_logger().warn(
							    "cypress",
							    "Logging of gsyn is not supported on BS "
							    "and will have no effect!");
							BrainScaleS::warn_gsyn_emitted = true;
						}
					}
				}
				else {
					throw cypress::ExecutionError(
					    "Recording variable " + signals[j] +
					    " is not supported by BrainScaleS!");
				}
			}
		}
	}

	/**
	 * Set recording flags, inhomogeneous for whole population
	 *
	 * @param params: target bs params of population
	 * @param pop: source cypress population
	 */
	template <typename T>
	static void set_inhom_rec(T &params, const PopulationBase &pop)
	{
		std::vector<std::string> signals = pop.type().signal_names;
		for (size_t j = 0; j < signals.size(); j++) {
			for (size_t neuron = 0; neuron < params.size(); neuron++) {
				if (pop[neuron].signals().is_recording(j)) {
					if (signals[j] == "spikes") {
						params[neuron].record_spikes = true;
					}
					else if (signals[j] == "v") {
						params[neuron].record_v = true;
					}
					else if (signals[j] == "gsyn_exc" ||
					         signals[j] == "gsyn_inh") {
						params[neuron].record_gsyn = true;
						if (!BrainScaleS::warn_gsyn_emitted) {
							global_logger().warn(
							    "cypress",
							    "Logging of gsyn is not supported on BS "
							    "and will have no effect!");
							BrainScaleS::warn_gsyn_emitted = true;
						}
					}
					else {
						throw cypress::ExecutionError(
						    "Recording variable " + signals[j] +
						    " is not supported by BrainScaleS!");
					}
				}
			}
		}
	}
	/**
	 * Set record spikes flags, homogeneous for whole population (for types
	 * where only spikes can be recorded)
	 *
	 * @param params: target bs params of population
	 * @param pop: source cypress population
	 */
	template <typename T>
	static void set_hom_rec_spikes(T &params)
	{
		for (size_t neuron = 0; neuron < params.size(); neuron++) {
			params[neuron].record_spikes = true;
		}
	}
	/**
	 * Set record spikes flags, inhomogeneous for whole population
	 *
	 * @param params: target bs params of population
	 * @param pop: source cypress population
	 */
	template <typename T>
	static void set_inhom_rec_spikes(T &params, const PopulationBase &pop)
	{
		for (size_t neuron = 0; neuron < params.size(); neuron++) {
			params[neuron].record_spikes =
			    pop[neuron].signals().is_recording(0);
		}
	}
	/**
	 * Set all record flags for a population, check for homogeneity and
	 * neuron type
	 *
	 * @param bs_pop BS population pointer (target)
	 * @param pop cypress population (source)
	 */
	static void set_population_records(PopulationPtr &bs_pop,
	                                   const PopulationBase &pop);

	/**
	 * Create the appropriate BS connector related to the one in a cypress
	 * ConnectionDescriptor.
	 *
	 * @param conn cypress connection description to be converted
	 * @return pointer to BS connector, directly used in Projection
	 */
	static boost::shared_ptr<::Connector> get_connector(
	    const ConnectionDescriptor &conn);

	/**
	 * Create a BS population view
	 *
	 * @param bs_pop parent BS population
	 * @param start first neuron in view
	 * @param end last +1 neuron in view
	 * @return BS population view
	 */
	static ::PopulationView get_popview(PopulationPtr bs_pop,
	                                    const size_t &start, const size_t &end);
	/**
	 * Trigger manual placement of all populations on a given set of hicanns
	 * @param hican json object being array of ints or ints, representing
	 * hicanns on a wafer
	 * @param marocco current pymarocco instance
	 * @param bs_populations vector of all populations that are manually
	 * placed
	 */
	static void manual_placement(
	    const Json &hicann, boost::shared_ptr<pymarocco::PyMarocco> marocco,
	    std::vector<PopulationPtr> &bs_populations);

	/**
	 * Setting low lever parameter for the BrainScaleS System
	 *
	 * @param wafer pointer to wafer object
	 * @param gmax set to 1023
	 * @param gmax_div set to 1
	 */
	static void set_stahl_params(boost::shared_ptr<sthal::Wafer> wafer,
	                             double gmax, double gmax_div);

	/**
	 * Fetching data after simulation. Check recording flags of pops, and
	 * fetches spikes and voltage traces
	 *
	 * @param populations cypress pops, in which data is put
	 * @param bs_populations BS pops, from which the data is fetched
	 */
	static void fetch_data(const std::vector<PopulationBase> &populations,
	                       const std::vector<PopulationPtr> &bs_populations);
};
}  // namespace cypress

extern "C" {
/**
 * @brief Expose constructor for program dynamically loading this lib
 *
 * @return pointer to BrainScaleS object
 */
cypress::Backend *make_brainscales_backend(const cypress::Json &setup)
{
	return new cypress::BrainScaleS(setup);
}
}
#endif /* CYPRESS_BACKEND_NEST_HPP */
